// Real FFN/owner code, mocked native device arithmetic. No GPU qualification.
#include "strata/core/portable_ffn.hpp"
#include "strata_vk_native_shared.hpp"
#include "strata_vk_native_stages.hpp"
#include "strata_vk_native_router.hpp"
#include "strata_vk_native_moe_combine.hpp"
#include "portable_shared_work.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <cstdio>
using namespace strata::core;
namespace {
void need(bool b,const std::string& message){if(!b)throw std::runtime_error(message);}
size_t owners=0,stage_owners=0,calls=0,last_shared_budget=0;int fault=0;int creation_fault=0;
std::set<std::string> paths;std::vector<float> mixed;std::vector<int> sequence;std::array<const ggml_tensor*,4> bound_weights{};
std::vector<float> read(const ggml_tensor* t){std::vector<float> v(size_t(ggml_nelements(t)));ggml_backend_tensor_get(t,v.data(),0,v.size()*4);return v;}
void put(ggml_tensor* t,float x){std::vector<float> v(size_t(ggml_nelements(t)),x);ggml_backend_tensor_set(t,v.data(),0,v.size()*4);}
float result(size_t t){return .125f+float(t)*.03125f;}
void receipt(const char* path){need(path&&paths.insert(path).second,"receipt reused");need(!std::filesystem::exists(path),"receipt overwritten");std::ofstream out(path);out<<"mock\n";need(bool(out),"receipt write");}
struct Fixture {
    PortableFfnGeometry g{2560,4,1,32,512,10,1e-6f};std::vector<std::vector<uint8_t>> data;std::vector<PortableTensorView> views;
    Fixture(){
        const char* names[]={"hc_ffn_norm.weight","hc_ffn_down.weight","hc_ffn_up.weight","hc_ffn_inject.weight","ffn_gate_inp.weight","ffn_gate_inp_shexp.weight","ffn_gate_shexp.weight","ffn_up_shexp.weight","ffn_down_shexp.weight"};
        const std::vector<std::vector<int64_t>> shapes={{10240},{10240,1},{1,10240},{10240,4},{2560,512},{2560},{2560,32},{2560,32},{32,2560}};
        data.resize(9);
        for(size_t i=0;i<9;++i){size_t n=1;for(auto d:shapes[i])n*=size_t(d);data[i].resize(i>=6?n/32*34:n*4);
            if(i==0)for(size_t j=0;j<n;++j){const float x=1;std::memcpy(data[i].data()+4*j,&x,4);}
            if(i==5)for(size_t j=0;j<n;++j){const float x=.125f;std::memcpy(data[i].data()+4*j,&x,4);}
            views.push_back({std::string("blk.0.")+names[i],shapes[i],i>=6?8u:0u,data[i].data(),data[i].size()});
        }
    }
};
}
namespace strata::core::portable_detail {
struct StrataVkNativeShared{};struct StrataVkNativeStages{};struct StrataVkNativeRouter{};struct StrataVkNativeMoeCombine{};
StrataVkNativeShared* strata_vk_native_shared_create(ggml_backend_t,const char* p,size_t budget){need(creation_fault!=1,"injected shared creation refusal");need(budget>=512,"shared metadata budget");last_shared_budget=budget;receipt(p);++owners;return new StrataVkNativeShared;}
size_t strata_vk_native_shared_bytes(const StrataVkNativeShared*){return 512;}
void strata_vk_native_shared_destroy(StrataVkNativeShared* p) noexcept{if(p)--owners;delete p;}
StrataVkNativeStages* strata_vk_native_stages_create_quant_rn_observation(ggml_backend_t,const char* p,size_t budget){need(creation_fault!=2,"injected quantizer creation refusal");need(budget>=128,"quant metadata budget");receipt(p);++stage_owners;return new StrataVkNativeStages;}
size_t strata_vk_native_stages_bytes(const StrataVkNativeStages*){return 128;}
void strata_vk_native_stages_destroy(StrataVkNativeStages* p) noexcept{if(p)--stage_owners;delete p;}
void strata_vk_native_stages_run(StrataVkNativeStages*,uint32_t mode,const ggml_tensor* g,const ggml_tensor*,ggml_tensor* h,ggml_tensor* packed,uint32_t width,const int32_t* starts,size_t ns){
    need(fault!=1,"injected quantizer failure");need(ns==2&&starts[0]==0&&starts[1]>=1,"token starts");
    sequence.push_back(mode==1?2:width==2560?0:3);
    if(mode==1)put(h,.25f);else {if(width==2560)mixed=read(g);std::vector<int32_t> q(size_t(ggml_nelements(packed)),0);ggml_backend_tensor_set(packed,q.data(),0,q.size()*4);}
}
void strata_vk_native_shared_run(StrataVkNativeShared*,uint32_t mode,const ggml_tensor* w,const ggml_tensor* x,ggml_tensor* out,ggml_tensor* aux,uint32_t ni,uint32_t no,uint32_t tokens){
    ++calls;need(fault!=2,"injected projection/scalar failure");
    need(w->type==(mode==2?GGML_TYPE_F32:GGML_TYPE_I32),"native weight binding type");
    if(mode==0) {sequence.push_back(no==2560?4:1);if(no!=2560){if(!bound_weights[0])bound_weights[0]=w;else bound_weights[1]=w;}else bound_weights[2]=w;put(out,0);}
    if(mode==1){sequence.push_back(5);bound_weights[3]=w;need(read(x)==mixed,"scalar gate did not consume original mixed activations");std::vector<uint16_t> bf(2560);ggml_backend_tensor_get(w,bf.data(),0,bf.size()*2);for(auto b:bf)need(b==0x3e00,"scalar BF16 binding changed");put(out,0);}
    if(mode==2){sequence.push_back(6);std::vector<float> v(size_t(tokens)*ni);for(size_t t=0;t<tokens;++t)std::fill_n(v.begin()+t*ni,ni,result(t));if(fault==3)v[0]=std::numeric_limits<float>::infinity();if(fault==5)std::fill(v.begin(),v.end(),std::numeric_limits<float>::max());ggml_backend_tensor_set(out,v.data(),0,v.size()*4);put(aux,.5f);}
}
StrataVkNativeRouter* strata_vk_native_router_create(ggml_backend_t,const char* p){receipt(p);return new StrataVkNativeRouter;}
size_t strata_vk_native_router_bytes(const StrataVkNativeRouter*){return 512;}
void strata_vk_native_router_destroy(StrataVkNativeRouter* p) noexcept{delete p;}
void strata_vk_native_router_run(StrataVkNativeRouter*,const ggml_tensor*,ggml_tensor* ids,ggml_tensor* weights,uint32_t t){std::vector<int32_t> v(t*10);for(size_t i=0;i<v.size();++i)v[i]=int(i%10);ggml_backend_tensor_set(ids,v.data(),0,v.size()*4);put(weights,.1f);}
StrataVkNativeMoeCombine* strata_vk_native_moe_combine_create(ggml_backend_t,const char* p){receipt(p);return new StrataVkNativeMoeCombine;}
size_t strata_vk_native_moe_combine_bytes(const StrataVkNativeMoeCombine*){return 512;}
void strata_vk_native_moe_combine_destroy(StrataVkNativeMoeCombine* p) noexcept{delete p;}
void strata_vk_native_moe_combine_run(StrataVkNativeMoeCombine*,const ggml_tensor*,const ggml_tensor*,const ggml_tensor* shared,ggml_tensor* out,uint32_t){auto v=read(shared);ggml_backend_tensor_set(out,v.data(),0,v.size()*4);}
}
int main(){try {
    const auto root=std::filesystem::temp_directory_path()/("shared-ffn-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));need(std::filesystem::create_directory(root),"test root");
    struct Cleanup{std::filesystem::path p;~Cleanup(){std::error_code ignored;std::filesystem::remove_all(p,ignored);}} cleanup{root};
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(),ggml_backend_free);
    Fixture fixture;std::string error;PortableFfnOptions opts;opts.accelerator=backend.get();opts.request_accelerator=true;opts.native_shared=true;
    opts.native_shared_receipt_prefix=(root/"shared").string();opts.native_router_receipt_prefix=(root/"router").string();opts.native_moe_combine_receipt_prefix=(root/"combine").string();
    const auto retained=opts.native_shared_receipt_prefix+".layer-0.t-1.owner-0";need(std::filesystem::create_directory(retained),"retained dir");{std::ofstream out(std::filesystem::path(retained)/"keep");out<<"retained";}
    for(int64_t T:{1,2,8})for(bool combined:{false,true}) {
        opts.native_router=combined;opts.native_moe_combine=combined;sequence.clear();bound_weights={};
        auto on=PortableFfn::create(fixture.g,0,fixture.views,opts,error,T);need(bool(on),error);
        need(on->memory_usage().device_weights>0,"native immutable device weights not accounted");
        need(on->memory_usage().device_graph>0 && on->graph_bytes()>=on->memory_usage().device_graph,"native scratch graph accounting");
        std::vector<float> initial(size_t(T)*10240,.5f);need(on->reset(initial,error),error);std::vector<PortableRoute> routes;
        auto dispatch=[&](auto,const auto& r,auto& rows,std::string& e){if(fault==4){e="provider failed";return false;}for(size_t t=0;t<r.size();++t){need(r[t].native_shared,"missing shared participation");need(r[t].native_router==combined,"router coexistence");need(std::equal(r[t].activation.begin(),r[t].activation.end(),mixed.begin()+t*2560),"original mixed row mismatch");}rows.assign(size_t(T)*10*2560,0);return true;};
        auto step=[&]{if(T==1){PortableRoute r;bool ok=on->step([&](auto l,const auto& r,auto& rows,auto& e){return dispatch(l,std::vector<PortableRoute>{r},rows,e);},r,error);routes={r};return ok;}return on->step_batch(dispatch,routes,error);};
        need(step(),error);need(sequence==std::vector<int>{0,1,1,2,3,4,5,6},"native sequence changed");
        for(size_t t=0;t<size_t(T);++t)for(size_t j=0;j<10240;++j)need(on->residual()[t*10240+j]==.5f+result(t),"native shared output not consumed directly");
        need(on->work().native_shared_calls==1&&on->work().native_shared_rows==uint64_t(T),"completed shared work");
        const auto before=on->residual();for(fault=1;fault<=4;++fault){need(!step(),"failed native/provider call succeeded");need(on->residual()==before,"failed call changed residual");need(on->work().native_shared_calls==1,"failed call credited shared execution");need(T==1?routes[0].ids.empty():routes.empty(),"failed call published routes");}fault=0;
        need(on->reset(initial,error),error);need(on->work().native_shared_calls==1,"reset lost shared counters");
        auto huge=initial;std::fill(huge.begin(),huge.end(),std::numeric_limits<float>::max());need(on->reset(huge,error),error);fault=5;
        need(!step(),"nonfinite post result admitted");need(on->residual()==huge && on->work().native_shared_calls==1,"post failure published state/work");fault=0;
        need(on->reset(initial,error),error);
        if(T==1) {auto* immutable=bound_weights[3];auto chunk=on->create_chunk_execution(2,error);need(bool(chunk),error);need(!on->create_chunk_execution(3,error),"active chunk lease lost");
            PortableMemoryUsage pair;need(on->combined_memory_usage(*chunk,pair,error),error);need(pair.device_weights==on->memory_usage().device_weights,"chunk duplicated immutable device weights");
            need(pair.device_graph==on->memory_usage().device_graph+chunk->memory_usage().device_graph,"chunk scratch ledger mismatch");
            need(chunk->reset(std::vector<float>(20480,.5f),error),error);std::vector<PortableRoute> r;need(chunk->step_batch([](auto,const auto& r,auto& rows,auto&){rows.assign(r.size()*10*2560,0);return true;},r,error),error);need(bound_weights[3]==immutable,"chunk did not borrow shared weights");
        }
    }
    need(!owners&&!stage_owners,"native component owner leaked");opts.native_router=opts.native_moe_combine=false;
    for(int bad=0;bad<6;++bad) {auto invalid=opts;if(bad==0)invalid.native_shared_receipt_prefix.clear();if(bad==1)invalid.allow_cpu_fallback=true;if(bad==2)invalid.graph_budget=1;if(bad==3)invalid.weight_budget=1;if(bad==4)invalid.accelerator=nullptr;if(bad==5)invalid.request_accelerator=false;need(!PortableFfn::create(fixture.g,0,fixture.views,invalid,error),"invalid native shared options/budget accepted");}
    const float arbitrary=1.00001f;std::memcpy(fixture.data[5].data(),&arbitrary,4);need(!PortableFfn::create(fixture.g,0,fixture.views,opts,error),"arbitrary scalar F32 truncated");{auto legacy=opts;legacy.native_shared=false;need(bool(PortableFfn::create(fixture.g,0,fixture.views,legacy,error)),"default scalar F32 support changed");}const float promoted=.125f;std::memcpy(fixture.data[5].data(),&promoted,4);
    auto unsupported=fixture.views;unsupported[6].type=0;need(!PortableFfn::create(fixture.g,0,unsupported,opts,error),"non-Q8 projection admitted");
    std::vector<uint16_t> bf(2560,0x3e00);auto raw=fixture.views;raw[5].type=30;raw[5].data=bf.data();raw[5].bytes=bf.size()*2;{auto on=PortableFfn::create(fixture.g,0,raw,opts,error);need(bool(on),error);need(on->reset(std::vector<float>(10240,.5f),error),error);PortableRoute route;need(on->step([](auto,const auto&,auto& rows,auto&){rows.assign(25600,0);return true;},route,error),error);}
    {auto invalid=opts;invalid.graph_budget=opts.graph_budget-last_shared_budget+512+127;need(!PortableFfn::create(fixture.g,0,fixture.views,invalid,error),"remaining quantizer metadata budget ignored");need(!owners&&!stage_owners,"metadata budget failure leaked owner");}
    for(creation_fault=1;creation_fault<=2;++creation_fault){need(!PortableFfn::create(fixture.g,0,fixture.views,opts,error),"creation failure accepted");need(!owners&&!stage_owners,"failed owner leaked");}creation_fault=0;
    auto offopts=opts;offopts.native_shared=false;auto off=PortableFfn::create(fixture.g,0,fixture.views,offopts,error);need(bool(off),error);auto before=calls;need(off->reset(std::vector<float>(10240,.5f),error),error);PortableRoute r;need(off->step([](auto,const auto&r,auto& rows,auto&){need(!r.native_shared,"default route tagged shared");rows.assign(25600,0);return true;},r,error),error);need(calls==before&&!off->work().native_shared_calls,"default path executed native shared");
    struct Work{uint64_t native_shared_calls=0,native_shared_rows=0;} work;PortableRoute tagged;tagged.native_shared=true;
    {portable_detail::SharedParticipation aborted(true);aborted.observe(tagged);}need(!work.native_shared_calls,"failed model forward credited work");
    portable_detail::SharedParticipation completed(true);completed.observe(tagged);completed.observe(std::vector<PortableRoute>(2,tagged));completed.commit(work);need(work.native_shared_calls==2&&work.native_shared_rows==3,"model shared counters");
    bool refused=false;try{completed.observe(PortableRoute{});}catch(...){refused=true;}need(refused,"model admitted silent shared fallback");
    need(std::filesystem::exists(std::filesystem::path(retained)/"keep"),"old receipt changed");
    puts("native shared FFN/model integration contracts pass; mocked arithmetic, no GPU qualification");return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
