#include "portable_native_ple_grouped_norm.hpp"
#include "portable_native_ple_gate.hpp"
#include "ggml-cpu.h"
#include <cassert>
#include <chrono>
#include <iostream>
#include <limits>

using namespace strata::core::portable_detail;
namespace strata::core::portable_detail {
struct StrataVkNativePleGroupedNorm { size_t bytes; uint64_t calls=0; };
static size_t creates=0;static bool refuse=false;
struct StrataVkNativePleGate {size_t bytes,budget;uint64_t calls=0;};
StrataVkNativePleGate* strata_vk_native_ple_gate_create(ggml_backend_t b,const char*,size_t budget){
    auto c=std::unique_ptr<ggml_context,decltype(&ggml_free)>(ggml_init({4096,nullptr,true}),ggml_free);ggml_new_tensor_1d(c.get(),GGML_TYPE_I32,1);
    return new StrataVkNativePleGate{ggml_backend_alloc_ctx_tensors_from_buft_size(c.get(),ggml_backend_get_default_buffer_type(b)),budget};
}
void strata_vk_native_ple_gate_destroy(StrataVkNativePleGate* p)noexcept{delete p;}
size_t strata_vk_native_ple_gate_bytes(const StrataVkNativePleGate* p){return p->bytes;}
uint64_t strata_vk_native_ple_gate_calls(const StrataVkNativePleGate* p)noexcept{return p->calls;}
uint32_t strata_vk_native_ple_gate_run(StrataVkNativePleGate*p,const ggml_tensor*k,const ggml_tensor*q,ggml_tensor*g,ggml_tensor*d){
    assert(k->buffer==q->buffer&&g->buffer==d->buffer&&k->buffer!=g->buffer);
    if(p->bytes+ggml_backend_buffer_get_size(k->buffer)+ggml_backend_buffer_get_size(g->buffer)>p->budget)throw std::runtime_error("gate aggregate budget");
    float gates[4]={0.5f,0.5f,0.5f,0.5f},diag[16]={};ggml_backend_tensor_set(g,gates,0,sizeof(gates));ggml_backend_tensor_set(d,diag,0,sizeof(diag));++p->calls;return 0;
}
StrataVkNativePleGroupedNorm* strata_vk_native_ple_grouped_norm_create(ggml_backend_t b,const char*,size_t){
    auto c=std::unique_ptr<ggml_context,decltype(&ggml_free)>(ggml_init({4096,nullptr,true}),ggml_free);
    ggml_new_tensor_1d(c.get(),GGML_TYPE_I32,1);++creates;
    return new StrataVkNativePleGroupedNorm{ggml_backend_alloc_ctx_tensors_from_buft_size(c.get(),ggml_backend_get_default_buffer_type(b))};
}
void strata_vk_native_ple_grouped_norm_destroy(StrataVkNativePleGroupedNorm* p)noexcept{delete p;}
size_t strata_vk_native_ple_grouped_norm_bytes(const StrataVkNativePleGroupedNorm* p){return p->bytes;}
uint64_t strata_vk_native_ple_grouped_norm_calls(const StrataVkNativePleGroupedNorm* p)noexcept{return p->calls;}
uint32_t strata_vk_native_ple_grouped_norm_run(StrataVkNativePleGroupedNorm* p,const ggml_tensor* in,const ggml_tensor* gamma,ggml_tensor* out,ggml_tensor*){
    if(refuse)return 1;std::vector<float> a(10240),b(10240);ggml_backend_tensor_get(in,a.data(),0,40960);ggml_backend_tensor_get(gamma,b.data(),0,40960);
    for(size_t i=0;i<a.size();++i)a[i]*=b[i];ggml_backend_tensor_set(out,a.data(),0,40960);++p->calls;return 0;
}
}
int main(int argc,char**argv){
    assert(argc==2);
    auto backend=std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)>(ggml_backend_cpu_init(),ggml_backend_free);
    std::array<std::vector<float>,3> gamma{std::vector<float>(10240,1),std::vector<float>(10240,2),std::vector<float>(10240,3)};
    std::array<const float*,3> roles{gamma[0].data(),gamma[1].data(),gamma[2].data()};
    const auto budget=PortableNativePleGroupedNorm::estimate(backend.get());bool rejected=false;
    try{PortableNativePleGroupedNorm bad(backend.get(),roles,"mock",budget-1);}catch(...){rejected=true;}assert(rejected&&creates==0);
    auto repeated=roles;repeated[1]=repeated[0];rejected=false;
    try{PortableNativePleGroupedNorm bad(backend.get(),repeated,"mock",budget);}catch(...){rejected=true;}assert(rejected&&creates==0);
    gamma[1][0]=std::numeric_limits<float>::infinity();rejected=false;
    try{PortableNativePleGroupedNorm bad(backend.get(),roles,"mock",budget);}catch(...){rejected=true;}assert(rejected&&creates==0);gamma[1][0]=2;
    PortableNativePleGroupedNorm owner(backend.get(),roles,"mock",budget);assert(owner.bytes()==budget&&creates==1);
    const auto dir=std::filesystem::path(argv[1])/("ple-norm-gate-budget-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    assert(std::filesystem::create_directory(dir));const auto gate_budget=PortableNativePleGate::estimate(backend.get());
    const auto borrowed=ggml_backend_buffer_get_size(owner.output(0)->buffer);
    PortableNativePleGate gate(backend.get(),(dir/"exact").string(),gate_budget,borrowed);
    gate.bind_direct(owner.output(0),owner.output(1));gate.run_direct(owner.output(0),owner.output(1));assert(gate.calls()==1);
    PortableNativePleGate short_borrow(backend.get(),(dir/"short").string(),gate_budget,borrowed-1);rejected=false;
    try{short_borrow.bind_direct(owner.output(0),owner.output(1));}catch(...){rejected=true;}assert(rejected);
    auto c=std::unique_ptr<ggml_context,decltype(&ggml_free)>(ggml_init({16384,nullptr,true}),ggml_free);
    std::array<ggml_tensor*,3> producer{},copy{};
    for(size_t i=0;i<3;++i){producer[i]=ggml_new_tensor_2d(c.get(),GGML_TYPE_F32,2560,4);copy[i]=ggml_cpy(c.get(),producer[i],owner.input(i));owner.bind(i,copy[i],producer[i]);}
    auto buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>(ggml_backend_alloc_ctx_tensors(c.get(),backend.get()),ggml_backend_buffer_free);
    std::vector<float> values(10240,4),result(10240);
    for(size_t i=0;i<3;++i){ggml_backend_tensor_set(producer[i],values.data(),0,40960);ggml_backend_tensor_set(owner.input(i),values.data(),0,40960);owner.run(i,copy[i],producer[i]);
        ggml_backend_tensor_get(owner.output(i),result.data(),0,40960);assert(result[0]==4*float(i+1)&&owner.ready(i));
        assert(owner.output(i)!=producer[i]&&owner.gamma(i)!=owner.gamma((i+1)%3));}
    owner.begin();for(size_t i=0;i<3;++i)assert(!owner.ready(i));
    values[0]=5;ggml_backend_tensor_set(owner.input(0),values.data(),0,40960);rejected=false;
    try{owner.run(0,copy[0],producer[0]);}catch(...){rejected=true;}assert(rejected&&!owner.ready(0));
    values[0]=4;ggml_backend_tensor_set(owner.input(0),values.data(),0,40960);refuse=true;rejected=false;
    try{owner.run(0,copy[0],producer[0]);}catch(...){rejected=true;}assert(rejected&&!owner.ready(0));
    refuse=false;owner.run(0,copy[0],producer[0]);assert(owner.ready(0));
    rejected=false;try{owner.run(0,copy[0],producer[1]);}catch(...){rejected=true;}assert(rejected&&!owner.ready(0));
    std::filesystem::remove_all(dir);
    std::cout<<"PLE grouped norm candidate banks, gamma roles, admission and refusal passed with host provider\n";
}
