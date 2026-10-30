// Real FFN coordinator, mocked device router/combine boundaries. Not an arithmetic oracle.
#include "strata/core/portable_ffn.hpp"
#include "strata_vk_native_router.hpp"
#include "strata_vk_native_moe_combine.hpp"
#include "portable_router_work.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <set>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
using namespace strata::core;
namespace {
void need(bool ok,const std::string& text){if(!ok)throw std::runtime_error(text);}
size_t owners=0,combine_owners=0,calls=0,combines=0;int fault=0;bool create_failure=false;
std::set<std::string> paths;
std::vector<float> captured_logits;
std::vector<float> read(const ggml_tensor* t){std::vector<float> result(size_t(ggml_nelements(t)));ggml_backend_tensor_get(t,result.data(),0,result.size()*4);return result;}
int32_t selected(size_t token,size_t rank){return int32_t(511-29*rank-3*token);}
float weight(size_t rank){return rank==9?0.f:float(10-rank)*0x1p-6f;}
void equal(const std::vector<float>& a,const std::vector<float>& b){need(a.size()==b.size()&&!std::memcmp(a.data(),b.data(),a.size()*4),"transaction changed residual bits");}
struct Fixture {
    PortableFfnGeometry g{2560,4,1,32,512,10,1e-6f};std::vector<std::vector<float>> data;std::vector<PortableTensorView> views;
    Fixture(){
        const char* names[]={"hc_ffn_norm.weight","hc_ffn_down.weight","hc_ffn_up.weight","hc_ffn_inject.weight","ffn_gate_inp.weight","ffn_gate_inp_shexp.weight","ffn_gate_shexp.weight","ffn_up_shexp.weight","ffn_down_shexp.weight"};
        const std::vector<std::vector<int64_t>> shapes={{10240},{10240,1},{1,10240},{10240,4},{2560,512},{2560},{2560,32},{2560,32},{32,2560}};
        data.resize(9);for(size_t i=0;i<9;++i){size_t n=1;for(auto x:shapes[i])n*=size_t(x);data[i].assign(n,0);}
        std::fill(data[0].begin(),data[0].end(),1.f);for(size_t e=0;e<512;++e)data[4][e*2560]=float(e)*0x1p-9f;
        for(size_t i=0;i<9;++i)views.push_back({std::string("blk.0.")+names[i],shapes[i],0,data[i].data(),data[i].size()*4});
    }
};
void receipt(const char* path){need(path&&*path,"missing router receipt");need(paths.insert(path).second,"receipt reused");
    need(std::filesystem::is_directory(std::filesystem::path(path).parent_path())&&!std::filesystem::exists(path),"receipt namespace not fresh");
    std::ofstream out(path);out<<"mock component receipt\n";need(bool(out),"receipt write");}
}
namespace strata::core::portable_detail {
struct StrataVkNativeRouter{};struct StrataVkNativeMoeCombine{};
StrataVkNativeRouter* strata_vk_native_router_create(ggml_backend_t,const char* path){need(!create_failure,"injected create failure");receipt(path);++owners;return new StrataVkNativeRouter;}
size_t strata_vk_native_router_bytes(const StrataVkNativeRouter*){return 512;}
void strata_vk_native_router_destroy(StrataVkNativeRouter* p) noexcept{--owners;delete p;}
void strata_vk_native_router_run(StrataVkNativeRouter*,const ggml_tensor* logits,ggml_tensor* ids,ggml_tensor* weights,uint32_t tokens){
    ++calls;need(fault!=1,"injected router execution refusal");
    need(logits->type==GGML_TYPE_F32&&ggml_is_contiguous(logits)&&ggml_nelements(logits)==tokens*512,"logits device boundary/shape");
    captured_logits=read(logits);std::vector<int32_t> selected_ids(tokens*10);std::vector<float> selected_weights(tokens*10);
    for(size_t t=0;t<tokens;++t)for(size_t k=0;k<10;++k){selected_ids[t*10+k]=selected(t,k);selected_weights[t*10+k]=weight(k);}
    if(fault==2)selected_ids[0]=512;if(fault==3)selected_ids[1]=selected_ids[0];
    if(fault==4)selected_weights[0]=std::numeric_limits<float>::quiet_NaN();if(fault==5)selected_weights[0]=-1;
    if(fault==6)std::fill(selected_weights.begin(),selected_weights.end(),0);
    ggml_backend_tensor_set(ids,selected_ids.data(),0,selected_ids.size()*4);ggml_backend_tensor_set(weights,selected_weights.data(),0,selected_weights.size()*4);
}
StrataVkNativeMoeCombine* strata_vk_native_moe_combine_create(ggml_backend_t,const char* path){receipt(path);++combine_owners;return new StrataVkNativeMoeCombine;}
size_t strata_vk_native_moe_combine_bytes(const StrataVkNativeMoeCombine*){return 512;}
void strata_vk_native_moe_combine_destroy(StrataVkNativeMoeCombine* p) noexcept{--combine_owners;delete p;}
void strata_vk_native_moe_combine_run(StrataVkNativeMoeCombine*,const ggml_tensor* rows,const ggml_tensor* weights,const ggml_tensor* shared,ggml_tensor* output,uint32_t tokens){
    ++combines;const auto parts=read(rows),w=read(weights),gated=read(shared);auto result=gated;
    for(size_t t=0;t<tokens;++t)for(size_t h=0;h<2560;++h){float sum=0;for(size_t k=0;k<10;++k){need(w[t*10+k]==weight(k),"router weights changed before combine");sum+=parts[(t*10+k)*2560+h]*w[t*10+k];}result[t*2560+h]+=sum;}
    ggml_backend_tensor_set(output,result.data(),0,result.size()*4);
}
}
int main(){try {
    const auto root=std::filesystem::temp_directory_path()/("router-ffn-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    need(std::filesystem::create_directory(root),"test receipt directory");
    struct Cleanup{std::filesystem::path p;~Cleanup(){std::error_code ignored;std::filesystem::remove_all(p,ignored);}} cleanup{root};
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(),ggml_backend_free);need(bool(backend),"CPU mock backend");
    Fixture f;std::string error;PortableFfnOptions opts;opts.accelerator=backend.get();opts.request_accelerator=true;opts.native_router=true;opts.native_router_receipt_prefix=(root/"router").string();
    opts.native_moe_combine_receipt_prefix=(root/"combine").string();
    const auto retained=opts.native_router_receipt_prefix+".layer-0.t-1.owner-0";
    need(std::filesystem::create_directory(retained),"retained receipt collision setup");{std::ofstream out(std::filesystem::path(retained)/"effective.spv");out<<"retained\n";}
    for(bool combined:{false,true})for(int64_t T:{1,2,8}) {
        opts.native_moe_combine=combined;
        auto on=PortableFfn::create(f.g,0,f.views,opts,error,T);need(bool(on),error);
        auto offopts=opts;offopts.native_router=false;offopts.native_moe_combine=false;
        auto off=PortableFfn::create(f.g,0,f.views,offopts,error,T);need(bool(off),error);
        need(on->memory_usage().device_graph>off->memory_usage().device_graph,"router buffers not charged");
        need(on->memory_usage().host_extra_bound>=off->memory_usage().host_extra_bound+(24u<<10),"router host metadata not charged");
        need(on->graph_bytes()>=on->memory_usage().device_graph,"aggregate graph cap omitted router");
        std::vector<float> initial(size_t(T)*10240);for(size_t t=0;t<size_t(T);++t)std::fill(initial.begin()+t*10240,initial.begin()+(t+1)*10240,0.5f+float(t)*0x1p-5f);
        need(on->reset(initial,error),error);std::vector<float> expected=initial;bool provider_called=false;
        const PortableExpertRowsBatch dispatch=[&](int64_t layer,const auto& routes,auto& rows,std::string& e){
            provider_called=true;if(fault==7){e="injected provider failure";return false;}
            need(layer==0&&routes.size()==size_t(T),"route/token association");rows.resize(size_t(T)*2560*10);
            for(size_t t=0;t<size_t(T);++t){const auto& route=routes[t];need(route.native_router&&route.ids.size()==10,"native participation missing");
                for(size_t expert=0;expert<512;++expert)need(captured_logits[t*512+expert]==route.activation[0]*(float(expert)*0x1p-9f),"pre-graph logits/token association changed");
                float sum=0;for(size_t k=0;k<10;++k){need(route.ids[k]==selected(t,k)&&route.weights[k]==weight(k),"IDs/weights reordered or renormalized");
                    const float x=float(route.ids[k]%7-3)*0x1p-7f;sum+=x*route.weights[k];for(size_t h=0;h<2560;++h)rows[(t*10+k)*2560+h]=x;}
                for(size_t s=0;s<4;++s)for(size_t h=0;h<2560;++h)expected[t*10240+s*2560+h]=initial[t*10240+s*2560+h]+sum;
            }
            if(fault==8)rows[0]=std::numeric_limits<float>::infinity();return true;
        };
        std::vector<PortableRoute> routes;
        auto step=[&](){if(T==1){PortableRoute route;const bool ok=on->step([&](auto l,const auto&r,auto& rows,auto&e){return dispatch(l,std::vector<PortableRoute>{r},rows,e);},route,error);routes={route};return ok;}return on->step_batch(dispatch,routes,error);};
        need(step(),error);need(provider_called,"provider not reached");equal(on->residual(),expected);
        need(on->work().native_router_calls==1&&on->work().native_router_rows==uint64_t(T),"completed participation counters");
        const auto preserved=on->residual();for(fault=1;fault<=8;++fault){provider_called=false;need(!step(),"bad component/provider published success");
            need(T==1?routes[0].ids.empty():routes.empty(),"failed route published");equal(on->residual(),preserved);
            need(on->work().native_router_calls==1&&on->work().native_router_rows==uint64_t(T),"failure credited routing work");
            if(fault<=6)need(!provider_called,"malformed native route reached provider");}
        fault=0;need(on->reset(initial,error),error);need(on->work().native_router_calls==1,"reset discarded counters");
        auto before=calls;need(off->reset(initial,error),error);need(off->step_batch([](auto,const auto& r,auto& rows,auto&){for(const auto& route:r)need(!route.native_router,"default route tagged native");rows.assign(r.size()*10*2560,0);return true;},routes,error),error);need(calls==before,"default path called device router");
        if(T==1){auto chunk=on->create_chunk_execution(2,error);need(bool(chunk),error);PortableMemoryUsage pair;need(on->combined_memory_usage(*chunk,pair,error),error);
            need(pair.device_graph==on->memory_usage().device_graph+chunk->memory_usage().device_graph,"shared pair omitted router outputs");need(!on->create_chunk_execution(3,error),"active chunk lease lost");}
    }
    need(!owners&&!combine_owners,"component owner leaked");
    for(int bad=0;bad<5;++bad){auto invalid=opts;auto shape=f.g;if(bad==0)invalid.native_router_receipt_prefix.clear();if(bad==1)invalid.allow_cpu_fallback=true;if(bad==2)shape.experts=256;if(bad==3)shape.selected=9;if(bad==4)invalid.graph_budget=1;
        need(!PortableFfn::create(shape,0,f.views,invalid,error),"invalid options/budget admitted");}
    create_failure=true;need(!PortableFfn::create(f.g,0,f.views,opts,error),"component creation failure accepted");create_failure=false;need(!owners&&!combine_owners,"failed owner leaked");
    {std::ifstream in(std::filesystem::path(retained)/"effective.spv");std::string line;std::getline(in,line);need(line=="retained","receipt overwritten");}
    struct Work{uint64_t native_router_calls=0,native_router_rows=0;} work;
    portable_detail::RouterParticipation completed(true);PortableRoute tagged;tagged.native_router=true;completed.observe(tagged);completed.observe(std::vector<PortableRoute>(2,tagged));need(!work.native_router_calls,"work published early");completed.commit(work);need(work.native_router_calls==2&&work.native_router_rows==3,"model participation staging");
    portable_detail::RouterParticipation legacy(false);legacy.observe(PortableRoute{});legacy.commit(work);need(work.native_router_calls==2,"default routing credited");
    {portable_detail::RouterParticipation aborted(true);aborted.observe(tagged);}
    need(work.native_router_calls==2&&work.native_router_rows==3,"aborted forward credited router work");
    bool refused=false;try{completed.observe(PortableRoute{});}catch(...){refused=true;}need(refused,"silent host fallback credited");
    puts("native-method router FFN/model contracts pass; mocked component arithmetic, no GPU qualification");return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
