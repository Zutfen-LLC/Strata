#include "portable_native_ple_grouped_norm.hpp"
#include "ggml-backend.h"
static int copy_computes = 0;
static ggml_status copy_test_compute(ggml_backend_t b, ggml_cgraph* g) { ++copy_computes; return ggml_backend_graph_compute(b,g); }
#define STRATA_PLE_COPY_TEST_COMPUTE copy_test_compute
#include "portable_native_ple_copy_graph.hpp"
#undef STRATA_PLE_COPY_TEST_COMPUTE
#include "portable_native_ple_gate.hpp"
#include "ggml-backend.h"
static bool fail_gated_compute = false;
static ggml_status gated_test_compute(ggml_backend_t backend, ggml_cgraph* graph) {
    return fail_gated_compute ? GGML_STATUS_FAILED : ggml_backend_graph_compute(backend, graph);
}
#define STRATA_PLE_GATED_VALUE_TEST_COMPUTE gated_test_compute
#include "portable_native_ple_gated_value.hpp"
#undef STRATA_PLE_GATED_VALUE_TEST_COMPUTE
#include "ggml-cpu.h"
#include <chrono>
#include <iostream>
#include <limits>

static void check(bool ok){if(!ok)throw std::runtime_error("PLE grouped norm test failed");}

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
    check(k->buffer==q->buffer&&g->buffer==d->buffer&&k->buffer!=g->buffer);
    if(p->bytes+ggml_backend_buffer_get_size(k->buffer)+ggml_backend_buffer_get_size(g->buffer)>p->budget)throw std::runtime_error("gate aggregate budget");
    std::vector<float> gates(ggml_nbytes(g)/4,0.5f),diag(ggml_nbytes(d)/4);ggml_backend_tensor_set(g,gates.data(),0,ggml_nbytes(g));ggml_backend_tensor_set(d,diag.data(),0,ggml_nbytes(d));++p->calls;return 0;
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
    if(refuse)return 1;std::vector<float> a(ggml_nbytes(in)/4),b(10240);ggml_backend_tensor_get(in,a.data(),0,ggml_nbytes(in));ggml_backend_tensor_get(gamma,b.data(),0,40960);
    for(size_t i=0;i<a.size();++i)a[i]*=b[i%10240];ggml_backend_tensor_set(out,a.data(),0,ggml_nbytes(out));++p->calls;return 0;
}
}
int main(int argc,char**argv){
    check(argc==2);
    auto backend=std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)>(ggml_backend_cpu_init(),ggml_backend_free);
    std::array<std::vector<float>,3> gamma{std::vector<float>(10240,1),std::vector<float>(10240,2),std::vector<float>(10240,3)};
    std::array<const float*,3> roles{gamma[0].data(),gamma[1].data(),gamma[2].data()};
    const auto budget=PortableNativePleGroupedNorm::estimate(backend.get());bool rejected=false;
    try{PortableNativePleGroupedNorm bad(backend.get(),roles,"mock",budget-1);}catch(...){rejected=true;}check(rejected&&creates==0);
    auto repeated=roles;repeated[1]=repeated[0];rejected=false;
    try{PortableNativePleGroupedNorm bad(backend.get(),repeated,"mock",budget);}catch(...){rejected=true;}check(rejected&&creates==0);
    gamma[1][0]=std::numeric_limits<float>::infinity();rejected=false;
    try{PortableNativePleGroupedNorm bad(backend.get(),roles,"mock",budget);}catch(...){rejected=true;}check(rejected&&creates==0);gamma[1][0]=2;
    PortableNativePleGroupedNorm owner(backend.get(),roles,"mock",budget);check(owner.bytes()==budget&&creates==1);
    const auto dir=std::filesystem::path(argv[1])/("ple-norm-gate-budget-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const bool created=std::filesystem::create_directory(dir);check(created);const auto gate_budget=PortableNativePleGate::estimate(backend.get());
    const auto borrowed=ggml_backend_buffer_get_size(owner.output(0)->buffer);
    PortableNativePleGate gate(backend.get(),(dir/"exact").string(),gate_budget,borrowed);
    gate.bind_direct(owner.output(0),owner.output(1));gate.run_direct(owner.output(0),owner.output(1));check(gate.calls()==1);
    // A backend failure invalidates even a previously successful gated output.
    {
        auto vc=std::unique_ptr<ggml_context,decltype(&ggml_free)>(ggml_init({4096,nullptr,true}),ggml_free);
        auto* v=ggml_new_tensor_3d(vc.get(),GGML_TYPE_F32,2560,1,1);
        auto vb=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>(ggml_backend_alloc_ctx_tensors(vc.get(),backend.get()),ggml_backend_buffer_free);
        std::vector<float> values(2560,4);ggml_backend_tensor_set(v,values.data(),0,values.size()*4);
        PortableNativePleGatedValue gv(backend.get(),v,gate.gate(),PortableNativePleGatedValue::estimate(backend.get()));
        gv.run(v,gate.gate());check(gv.ready());
        gate.begin();gate.run_direct(owner.output(0),owner.output(1));check(gate.calls()==2);
        fail_gated_compute=true;rejected=false;
        try{gv.run(v,gate.gate());}catch(const std::runtime_error&){rejected=true;}
        fail_gated_compute=false;check(rejected&&!gv.ready());
        rejected=false;try{gv.output();}catch(const std::runtime_error&){rejected=true;}check(rejected);
        gv.reset();rejected=false;try{gv.output();}catch(const std::runtime_error&){rejected=true;}check(rejected);
        gv.run(v,gate.gate());std::vector<float> result(10240);ggml_backend_tensor_get(gv.output(),result.data(),0,result.size()*4);check(gv.ready()&&result[0]==2);
    }
    PortableNativePleGate short_borrow(backend.get(),(dir/"short").string(),gate_budget,borrowed-1);rejected=false;
    try{short_borrow.bind_direct(owner.output(0),owner.output(1));}catch(...){rejected=true;}check(rejected);
    auto c=std::unique_ptr<ggml_context,decltype(&ggml_free)>(ggml_init({1024*1024,nullptr,true}),ggml_free);
    std::array<ggml_tensor*,3> producer{},copy{};
    for(size_t i=0;i<3;++i){producer[i]=ggml_new_tensor_2d(c.get(),GGML_TYPE_F32,2560,4);copy[i]=ggml_cpy(c.get(),producer[i],owner.input(i));owner.bind(i,copy[i],producer[i]);}
    auto* graph=ggml_new_graph(c.get());
    for(auto* node:copy)ggml_build_forward_expand(graph,node);
    auto buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>(ggml_backend_alloc_ctx_tensors(c.get(),backend.get()),ggml_backend_buffer_free);
    std::vector<float> values(10240,4),result(10240);
    for(auto* source:producer)ggml_backend_tensor_set(source,values.data(),0,40960);
    check(ggml_backend_graph_compute(backend.get(),graph)==GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend.get());
    for(size_t i=0;i<3;++i){owner.run(i,copy[i],producer[i]);
        ggml_backend_tensor_get(owner.output(i),result.data(),0,40960);check(result[0]==4*float(i+1)&&owner.ready(i));
        check(owner.output(i)!=producer[i]&&owner.gamma(i)!=owner.gamma((i+1)%3));}
    owner.begin();for(size_t i=0;i<3;++i)check(!owner.ready(i));
    values[0]=5;ggml_backend_tensor_set(owner.input(0),values.data(),0,40960);rejected=false;
    try{owner.run(0,copy[0],producer[0]);}catch(...){rejected=true;}check(rejected&&!owner.ready(0));
    values[0]=4;ggml_backend_tensor_set(owner.input(0),values.data(),0,40960);refuse=true;rejected=false;
    try{owner.run(0,copy[0],producer[0]);}catch(...){rejected=true;}check(rejected&&!owner.ready(0));
    refuse=false;owner.run(0,copy[0],producer[0]);check(owner.ready(0));
    rejected=false;try{owner.run(0,copy[0],producer[1]);}catch(...){rejected=true;}check(rejected&&!owner.ready(0));
    const auto budget2=PortableNativePleGroupedNorm::estimate(backend.get(),2);
    PortableNativePleGroupedNorm batch(backend.get(),roles,"mock",budget2,2);
    auto c2=std::unique_ptr<ggml_context,decltype(&ggml_free)>(ggml_init({1024*1024,nullptr,true}),ggml_free);
    auto* source=ggml_new_tensor_3d(c2.get(),GGML_TYPE_F32,2560,4,2);
    auto* copy2=ggml_cpy(c2.get(),source,batch.input(1));batch.bind(1,copy2,source);
    auto* graph2=ggml_new_graph(c2.get());ggml_build_forward_expand(graph2,copy2);
    auto buf2=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>(ggml_backend_alloc_ctx_tensors(c2.get(),backend.get()),ggml_backend_buffer_free);
    std::vector<float> v2(20480,4),r2(20480);v2[10240]=7;
    ggml_backend_tensor_set(source,v2.data(),0,81920);
    check(ggml_backend_graph_compute(backend.get(),graph2)==GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend.get());
    batch.run(1,copy2,source);ggml_backend_tensor_get(batch.output(1),r2.data(),0,81920);check(r2[0]==8&&r2[10240]==14);
    v2[10240]=9;ggml_backend_tensor_set(batch.input(1),v2.data(),0,81920);rejected=false;
    try{batch.run(1,copy2,source);}catch(...){rejected=true;}check(rejected&&!batch.ready(1));
    // Exercise the production copy owner at both fixed full-T sizes.
    for(size_t rows: {size_t(1),size_t(2)}){
        auto pc=std::unique_ptr<ggml_context,decltype(&ggml_free)>(ggml_init({16384,nullptr,true}),ggml_free);
        std::array<ggml_tensor*,3> sources{};
        for(auto& p:sources)p=ggml_new_tensor_3d(pc.get(),GGML_TYPE_F32,2560,4,rows);
        auto pb=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>(ggml_backend_alloc_ctx_tensors(pc.get(),backend.get()),ggml_backend_buffer_free);
        for(size_t i=0;i<3;++i){std::vector<float> v(10240*rows,float(i+1));v.back()=float(10+i);ggml_backend_tensor_set(sources[i],v.data(),0,v.size()*4);}
        auto amount=PortableNativePleCopyGraph::estimate(backend.get(),rows);
        rejected=false;try{PortableNativePleCopyGraph bad(backend.get(),sources,roles,"mock",amount-1,rows);}catch(...){rejected=true;}check(rejected);
        PortableNativePleCopyGraph cg(backend.get(),sources,roles,"mock",amount,rows);
        check(cg.bytes()==amount);
        if(rows==2){
            auto* saved_data=sources[0]->data;
            auto base=reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(sources[0]->buffer));
            auto capacity=ggml_backend_buffer_get_size(sources[0]->buffer);
            for(auto address : {uintptr_t(0),base+capacity+4,base+capacity-81916}){
                sources[0]->data=reinterpret_cast<void*>(address); auto calls=cg.calls(); auto computes=copy_computes; rejected=false;
                try{cg.run_key_query(sources);}catch(...){rejected=true;}
                check(rejected&&!cg.ready()&&cg.calls()==calls&&copy_computes==computes);
            }
            sources[0]->data=reinterpret_cast<void*>(base+4);
            cg.run_key_query(sources); check(cg.calls()==2);
            sources[0]->data=saved_data; cg.reset();
        }
        const auto initial_calls=cg.calls();
        rejected=false;try{cg.run_gated_value(sources);}catch(...){rejected=true;}check(rejected&&!cg.ready()&&cg.calls()==initial_calls);
        cg.run_key_query(sources);check(!cg.ready()&&cg.calls()==initial_calls+2);
        rejected=false;try{cg.output(2);}catch(...){rejected=true;}check(rejected);
        std::vector<float> later(10240*rows,7);later.back()=19;
        ggml_backend_tensor_set(sources[2],later.data(),0,later.size()*4);
        cg.run_gated_value(sources);check(cg.ready()&&cg.calls()==initial_calls+3);
        for(size_t i=0;i<3;++i){std::vector<float> v(10240*rows);ggml_backend_tensor_get(cg.output(i),v.data(),0,v.size()*4);check(v[0]==(i==2?21.0f:float((i+1)*(i+1)))&&v.back()==(i==2?57.0f:float((10+i)*(i+1))));}
        rejected=false;try{cg.run_gated_value(sources);}catch(...){rejected=true;}check(rejected&&!cg.ready());
        auto wrong=sources;wrong[0]=sources[1];rejected=false;try{cg.run_key_query(wrong);}catch(...){rejected=true;}check(rejected&&!cg.ready());
        auto* node=const_cast<ggml_tensor*>(cg.copy(0));auto* saved=node->src[0];node->src[0]=sources[1];
        rejected=false;try{cg.run_key_query(sources);}catch(...){rejected=true;}check(rejected&&!cg.ready());node->src[0]=saved;
        refuse=true;rejected=false;try{cg.run_key_query(sources);}catch(...){rejected=true;}check(rejected&&!cg.ready());refuse=false;
        rejected=false;try{cg.run_gated_value(sources);}catch(...){rejected=true;}check(rejected);
        cg.run_key_query(sources);refuse=true;rejected=false;try{cg.run_gated_value(sources);}catch(...){rejected=true;}check(rejected&&!cg.ready());refuse=false;
        rejected=false;try{cg.output(0);}catch(...){rejected=true;}check(rejected);
        rejected=false;try{cg.run_gated_value(sources);}catch(...){rejected=true;}check(rejected);
        cg.run_key_query(sources);cg.run_gated_value(sources);check(cg.ready());cg.reset();check(!cg.ready());
    }
    // Full-T GGML gated value must agree with independent single-row graphs.
    std::vector<float> batch_result;
    for (size_t rows : {size_t(2), size_t(1), size_t(1)}) {
        static size_t pass = 0; size_t first = rows == 2 ? 0 : pass++;
        auto vc=std::unique_ptr<ggml_context,decltype(&ggml_free)>(ggml_init({16384,nullptr,true}),ggml_free);
        auto* v=ggml_new_tensor_3d(vc.get(),GGML_TYPE_F32,2560,1,rows);
        auto* g=ggml_new_tensor_3d(vc.get(),GGML_TYPE_F32,1,4,rows);
        auto vb=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>(ggml_backend_alloc_ctx_tensors(vc.get(),backend.get()),ggml_backend_buffer_free);
        std::vector<float> values(rows*2560), gates(rows*4), result(rows*10240);
        for(size_t t=0;t<rows;++t){for(size_t i=0;i<2560;++i)values[t*2560+i]=float((first+t)*17+i%31+1);for(size_t h=0;h<4;++h)gates[t*4+h]=float((first+t)*4+h+1)/8;}
        ggml_backend_tensor_set(v,values.data(),0,values.size()*4);ggml_backend_tensor_set(g,gates.data(),0,gates.size()*4);
        auto amount=PortableNativePleGatedValue::estimate(backend.get(),rows);
        rejected=false;try{PortableNativePleGatedValue bad(backend.get(),v,g,amount-1,rows);}catch(...){rejected=true;}check(rejected);
        PortableNativePleGatedValue gv(backend.get(),v,g,amount,rows);
        rejected=false;try{gv.output();}catch(...){rejected=true;}check(rejected);
        rejected=false;try{gv.run(g,v);}catch(...){rejected=true;}check(rejected&&!gv.ready());
        gv.run(v,g);ggml_backend_tensor_get(gv.output(),result.data(),0,result.size()*4);
        for(size_t t=0;t<rows;++t)for(size_t h=0;h<4;++h)for(size_t i=0;i<2560;++i)check(result[t*10240+h*2560+i]==values[t*2560+i]*gates[t*4+h]);
        if(rows==2)batch_result=result;else for(size_t i=0;i<10240;++i)check(result[i]==batch_result[first*10240+i]);
        auto* saved=v->data;v->data=reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(v->buffer))+ggml_backend_buffer_get_size(v->buffer));
        rejected=false;try{gv.run(v,g);}catch(...){rejected=true;}check(rejected&&!gv.ready());v->data=saved;
        auto* gate_data=g->data;g->data=v->data;
        rejected=false;try{gv.run(v,g);}catch(...){rejected=true;}check(rejected&&!gv.ready());g->data=gate_data;
        auto old_ne=g->ne[1];g->ne[1]=3;
        rejected=false;try{gv.run(v,g);}catch(...){rejected=true;}check(rejected&&!gv.ready());g->ne[1]=old_ne;
        gv.run(v,g);gv.reset();check(!gv.ready());
    }
    auto gate_budget2=PortableNativePleGate::estimate(backend.get(),2);
    PortableNativePleGate gate2(backend.get(),(dir/"batch").string(),gate_budget2,ggml_backend_buffer_get_size(batch.output(0)->buffer),2);
    gate2.bind_direct(batch.output(0),batch.output(1));gate2.run_direct(batch.output(0),batch.output(1));check(gate2.calls()==1&&ggml_nbytes(gate2.gate())==32);
    std::cout<<"PLE grouped norm candidate banks, gamma roles, admission and refusal passed with host provider\n";
}
