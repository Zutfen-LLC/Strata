#include "strata/core/portable_ffn.hpp"
#include "strata_vk_native_moe_combine.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cfenv>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <set>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <memory>
#include <limits>
#ifdef STRATA_COMBINE_REAL_VULKAN
#include "ggml-vulkan.h"
#endif
using namespace strata::core;
namespace {
void need(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
std::vector<float> expected_rows,expected_weights,expected_shared,combined;
size_t calls=0,owners=0;
std::set<std::string> receipt_paths;
bool reject=false,nonfinite=false;
void equal(const std::vector<float>& a,const std::vector<float>& b,const char* message) {
    need(a.size()==b.size() && !std::memcmp(a.data(),b.data(),a.size()*4),message);
}
std::vector<float> reference(const std::vector<float>& rows,const std::vector<float>& weights,const std::vector<float>& shared) {
    std::vector<float> result(shared.size());
    for(size_t t=0;t<shared.size()/2560;++t)for(size_t h=0;h<2560;++h) {
        volatile float first=rows[(t*10)*2560+h]*weights[t*10];float sum=first;
        for(size_t k=1;k<10;++k)sum=std::fma(rows[(t*10+k)*2560+h],weights[t*10+k],sum);
        volatile float last=sum+shared[t*2560+h];result[t*2560+h]=last;
    }
    return result;
}
struct Fixture {
    PortableFfnGeometry g{2560,4,1,32,12,10,1e-6f};
    std::vector<std::vector<float>> data;
    std::vector<PortableTensorView> views;
    Fixture() {
        const char* names[]={"hc_ffn_norm.weight","hc_ffn_down.weight","hc_ffn_up.weight","hc_ffn_inject.weight",
            "ffn_gate_inp.weight","ffn_gate_inp_shexp.weight","ffn_gate_shexp.weight","ffn_up_shexp.weight","ffn_down_shexp.weight"};
        const std::vector<std::vector<int64_t>> shapes={{10240},{10240,1},{1,10240},{10240,4},{2560,12},{2560},{2560,32},{2560,32},{32,2560}};
        data.resize(9);
        for(size_t i=0;i<9;++i){size_t count=1;for(auto n:shapes[i])count*=size_t(n);data[i].assign(count,0);}
        std::fill(data[0].begin(),data[0].end(),1.f);
        // Distinct logits reverse the expert IDs. Zero scalar shared gate logit
        // means exactly 0.5, so gating twice/omitting it is observable.
        for(size_t e=0;e<12;++e)data[4][e*2560]=float(e)*0.125f;
        data[6][0]=data[7][0]=1.f;
        for(size_t h=0;h<2560;++h)data[8][h*32]=1.f;
        for(size_t i=0;i<9;++i)views.push_back({std::string("blk.0.")+names[i],shapes[i],0,data[i].data(),data[i].size()*4});
    }
};
}
#ifndef STRATA_COMBINE_REAL_VULKAN
namespace strata::core::portable_detail {
struct StrataVkNativeMoeCombine {};
StrataVkNativeMoeCombine* strata_vk_native_moe_combine_create(ggml_backend_t,const char* path) {
    need(path && *path,"Q8 component: effective shader receipt required");
    const std::filesystem::path receipt(path);
    need(std::filesystem::is_directory(receipt.parent_path()) && !std::filesystem::exists(receipt),"receipt must have a fresh owner namespace");
    need(receipt_paths.insert(path).second,"component receipt path reused");
    std::ofstream out(receipt);out<<"mock effective shader receipt\n";out.close();need(bool(out),"mock receipt write failed");
    ++owners;return new StrataVkNativeMoeCombine;
}
size_t strata_vk_native_moe_combine_bytes(const StrataVkNativeMoeCombine*) {return 512;}
void strata_vk_native_moe_combine_destroy(StrataVkNativeMoeCombine* p) noexcept {--owners;delete p;}
void strata_vk_native_moe_combine_run(StrataVkNativeMoeCombine*,const ggml_tensor* rows,const ggml_tensor* weights,
    const ggml_tensor* shared,ggml_tensor* output,uint32_t tokens) {
    ++calls;need(!reject,"injected combine refusal");
    auto read=[](const ggml_tensor* t){std::vector<float> v(size_t(ggml_nelements(t)));ggml_backend_tensor_get(t,v.data(),0,v.size()*4);return v;};
    equal(read(rows),expected_rows,"row/CPU-miss original slot order changed");
    equal(read(weights),expected_weights,"token/slot weights changed");
    const auto gated=read(shared);need(gated.size()==size_t(tokens)*2560,"shared shape");
    for(size_t i=0;i<gated.size();++i)need(std::fabs(gated[i]-expected_shared[i])<2e-6f,"shared gate must apply exactly once");
    combined=reference(expected_rows,expected_weights,gated);
    if(nonfinite)combined[0]=std::numeric_limits<float>::quiet_NaN();
    ggml_backend_tensor_set(output,combined.data(),0,combined.size()*4);
}
}
#endif
int main(int argc,char** argv){try {
    need(std::fesetround(FE_TONEAREST)==0,"host reference requires RNE");
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(nullptr,ggml_backend_free);
#ifdef STRATA_COMBINE_REAL_VULKAN
    need(argc==7 && std::string(argv[1])=="--device" && std::string(argv[3])=="--expect-device" && std::string(argv[5])=="--receipt-prefix",
        "usage: portable_ffn_native_combine_vulkan --device INDEX --expect-device SUBSTRING --receipt-prefix PRIVATE_PREFIX");
    const std::string index=argv[2];need(!index.empty() && index.find_first_not_of("0123456789")==std::string::npos,"invalid device index");
    const auto device=std::stoull(index);need(device<size_t(ggml_backend_vk_get_device_count()),"device unavailable");
    backend.reset(ggml_backend_vk_init(device));need(bool(backend),"Vulkan unavailable");
    const char* description=ggml_backend_dev_description(ggml_backend_get_device(backend.get()));
    need(std::string(argv[4]).size() && description && std::string(description).find(argv[4])!=std::string::npos,"unexpected physical device");
#else
    (void)argc;(void)argv;backend.reset(ggml_backend_cpu_init());
#endif
#ifndef STRATA_COMBINE_REAL_VULKAN
    struct ReceiptDirectory {
        std::filesystem::path path;
        ReceiptDirectory() {
            const auto seed=std::chrono::steady_clock::now().time_since_epoch().count();
            for(size_t i=0;i<1024;++i) {
                auto candidate=std::filesystem::temp_directory_path()/("strata-combine-test-"+std::to_string(seed)+"-"+std::to_string(i));
                if(std::filesystem::create_directory(candidate)){path=std::move(candidate);return;}
            }
            throw std::runtime_error("test receipt directory unavailable");
        }
        ~ReceiptDirectory(){std::error_code ignored;std::filesystem::remove_all(path,ignored);}
    } private_receipts;
#endif
    Fixture fixture;std::string error;
    PortableFfnOptions opts;opts.accelerator=backend.get();opts.request_accelerator=true;opts.native_moe_combine=true;
#ifdef STRATA_COMBINE_REAL_VULKAN
    opts.native_moe_combine_receipt_prefix=argv[6];
#else
    opts.native_moe_combine_receipt_prefix=(private_receipts.path/"combine").string();
    const auto retained=std::filesystem::path(opts.native_moe_combine_receipt_prefix+".layer-0.t-1.owner-0")/"effective.spv";
    need(std::filesystem::create_directory(retained.parent_path()),"retained receipt fixture setup");
    {std::ofstream out(retained);out<<"retained earlier receipt\n";need(bool(out),"retained receipt fixture write");}
    auto missing_receipt=opts;missing_receipt.native_moe_combine_receipt_prefix.clear();
    need(!PortableFfn::create(fixture.g,0,fixture.views,missing_receipt,error),"empty receipt prefix accepted");
#endif
    for(int64_t T:{1,2,8}) {
        expected_rows.clear();expected_weights.clear();expected_shared.clear();
        std::vector<float> initial(size_t(T)*10240);
        for(size_t t=0;t<size_t(T);++t)for(size_t s=0;s<4;++s)for(size_t h=0;h<2560;++h)
            initial[t*10240+s*2560+h]=0.5f+float((h+t*3)%11)*0.03125f;
        const PortableExpertRowsBatch dispatch=[&](int64_t layer,const std::vector<PortableRoute>& routes,std::vector<float>& rows,std::string&) {
            need(layer==0 && routes.size()==size_t(T),"dispatch association");rows.resize(size_t(T)*10*2560);
            expected_weights.clear();expected_shared.clear();
            for(size_t t=0;t<routes.size();++t) {
                const auto& r=routes[t];need(r.ids.size()==10 && r.ids[0]==11 && r.ids[9]==2,"rank order changed");
                expected_weights.insert(expected_weights.end(),r.weights.begin(),r.weights.end());
                const float x=r.activation[0];const float shared=0.5f*((x/(1.f+std::exp(-x)))*x);
                expected_shared.insert(expected_shared.end(),2560,shared);
                // Original slots deliberately mix two provider origins. Restoring
                // reverse-packed replies must precede this callback boundary.
                for(size_t k=0;k<10;++k)for(size_t h=0;h<2560;++h)
                    rows[(t*10+k)*2560+h]=float(int((t*17+size_t(r.ids[k])*7+h%13)%47)-23)*0x1p-8f;
            }
            expected_rows=rows;
#ifdef STRATA_COMBINE_REAL_VULKAN
            combined=reference(rows,expected_weights,expected_shared);
#endif
            return true;
        };
        auto runner=PortableFfn::create(fixture.g,0,fixture.views,opts,error,T);need(bool(runner),error);
        need(runner->reset(initial,error),error);std::vector<PortableRoute> routes;
        if(T==1) {
            PortableRoute route;
            need(runner->step([&](int64_t l,const PortableRoute& r,std::vector<float>& rows,std::string& e){return dispatch(l,{r},rows,e);},route,error),error);
            routes={route};
        } else need(runner->step_batch(dispatch,routes,error),error);
        const auto result=runner->residual();
        for(size_t t=0;t<size_t(T);++t)for(size_t s=0;s<4;++s)for(size_t h=0;h<2560;++h) {
            const float expected=initial[t*10240+s*2560+h]+combined[t*2560+h];
#ifdef STRATA_COMBINE_REAL_VULKAN
            need(std::isfinite(result[t*10240+s*2560+h]) && std::fabs(result[t*10240+s*2560+h]-expected)<4e-6f,"FFN combined residual mismatch");
#else
            need(result[t*10240+s*2560+h]==expected,"post graph must consume combine once, without host sum");
#endif
        }
        auto legacy_opts=opts;legacy_opts.native_moe_combine=false;
        auto legacy=PortableFfn::create(fixture.g,0,fixture.views,legacy_opts,error,T);need(bool(legacy),error);
        need(runner->memory_usage().device_graph>legacy->memory_usage().device_graph,"staging not charged to device graph");
        need(runner->graph_bytes()>=runner->memory_usage().device_graph,"staging not charged to graph cap");
#ifndef STRATA_COMBINE_REAL_VULKAN
        if(T==1) {
            auto chunk=runner->create_chunk_execution(2,error);need(bool(chunk),error);
            PortableMemoryUsage pair;need(runner->combined_memory_usage(*chunk,pair,error),error);
            need(pair.device_graph==runner->memory_usage().device_graph+chunk->memory_usage().device_graph,
                "chunk combine buffers missing from aggregate ledger");
            need(!runner->create_chunk_execution(3,error),"active chunk lease ignored");
        }
        const auto before=calls;
        need(legacy->reset(initial,error) && legacy->step_batch(dispatch,routes,error),error);
        need(calls==before,"legacy path called native combine");
        for(int fault:{0,1}) {
            reject=fault==0;nonfinite=fault==1;
            if(T==1) {
                PortableRoute route;
                need(!runner->step([&](int64_t l,const PortableRoute& r,std::vector<float>& rows,std::string& e){return dispatch(l,{r},rows,e);},route,error) && route.ids.empty(),
                    "single combine failure must refuse publication");
            } else need(!runner->step_batch(dispatch,routes,error) && routes.empty(),"combine failure must refuse publication");
            equal(runner->residual(),result,"failed combine changed residual");
        }
        reject=nonfinite=false;
#endif
    }
#ifndef STRATA_COMBINE_REAL_VULKAN
    need(owners==0,"combine owner leaked");
    need(receipt_paths.size()==4,"T1/chunk/T2/T8 owners require distinct receipts");
    for(const auto& path:receipt_paths) {
        std::ifstream in(path);std::string line;std::getline(in,line);
        need(line=="mock effective shader receipt","earlier receipt overwritten");
    }
    // A second T1 owner for the same layer/prefix must retain the first file.
    {auto repeat=PortableFfn::create(fixture.g,0,fixture.views,opts,error);need(bool(repeat),error);}
    need(receipt_paths.size()==5,"recreated T1 owner reused receipt");
    {std::ifstream in(retained);std::string line;std::getline(in,line);need(line=="retained earlier receipt","restart collision overwrote an earlier receipt");}
    auto bad=opts;bad.graph_budget=1;need(!PortableFfn::create(fixture.g,0,fixture.views,bad,error),"tiny graph budget accepted");
    bad=opts;bad.allow_cpu_fallback=true;need(!PortableFfn::create(fixture.g,0,fixture.views,bad,error),"fallback accepted");
    auto shape=fixture.g;shape.selected=9;need(!PortableFfn::create(shape,0,fixture.views,opts,error),"non-K10 accepted");
    need(owners==0,"failed construction leaked combine owner");
#endif
    puts("portable FFN native combine integration PASS (observation only; no full model qualification)");return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
