#include "ggml-backend.h"
static bool fail_gated_compute = false;
static unsigned gated_failures = 0;
static ggml_status gated_test_compute(ggml_backend_t backend, ggml_cgraph* graph) {
    if (fail_gated_compute) { ++gated_failures; return GGML_STATUS_FAILED; }
    return ggml_backend_graph_compute(backend, graph);
}
#define STRATA_PLE_GATED_VALUE_TEST_COMPUTE gated_test_compute
#include "portable_native_ple_executor.hpp"
#undef STRATA_PLE_GATED_VALUE_TEST_COMPUTE
#include "portable_native_ple_composition.hpp"
#include "native_component_bound_fake.hpp"
#include <unordered_map>
#include "portable_ple_fixture.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
using namespace strata::core;
using namespace strata::core::portable_detail;
static void check(bool b,const char* s){if(!b)throw std::runtime_error(s);}
static unsigned quant=0,dot=0,value=0,norm=0,gate=0,conv=0;
static unsigned executor_creates=0;
static bool factory_probe=false;
static size_t factory_fail=0;
static std::vector<std::string> factory_calls;
static std::vector<unsigned> factory_destroyed;
static std::unordered_map<const void*,size_t> factory_owners;
static size_t factory_enter(const char* name){if(!factory_probe)return SIZE_MAX;size_t i=factory_calls.size();factory_calls.emplace_back(name);factory_destroyed.push_back(0);if(i==factory_fail)throw std::runtime_error("injected factory failure");return i;}
template<class T> static T* factory_return(T* p,size_t i){if(i!=SIZE_MAX)factory_owners.emplace(p,i);return p;}
static void factory_destroy(const void* p){auto it=factory_owners.find(p);if(it!=factory_owners.end()){++factory_destroyed[it->second];factory_owners.erase(it);}}
static std::string fail_at;
static int norm_call=0;
static void inject(const char* name){if(fail_at==name) throw std::runtime_error(std::string("injected ")+name);}
static std::vector<float> expected_embedding;
static std::array<int,3> hc_creates{}, hc_runs{};
static std::array<size_t,3> hc_budgets{};
static std::array<std::vector<unsigned char>,4> hc_roles;
static void verify_hc_role(const ggml_tensor* t,size_t i){std::vector<unsigned char> v(ggml_nbytes(t));ggml_backend_tensor_get(t,v.data(),0,v.size());check(v==hc_roles[i],"HC copied role");}
static void fill_hc(ggml_tensor* t,float x){std::vector<float> v(ggml_nelements(t),x);ggml_backend_tensor_set(t,v.data(),0,v.size()*sizeof(float));}
static unsigned candidate_writes=0;
static void write_then_throw(const char* point, std::initializer_list<ggml_tensor*> tensors, float value) {
 if(fail_at!=point) return;
 for(auto* t:tensors){fill_hc(t,value);std::vector<float> actual(ggml_nelements(t));ggml_backend_tensor_get(t,actual.data(),0,actual.size()*sizeof(float));check(!actual.empty()&&actual==std::vector<float>(actual.size(),value),"candidate write missing");value+=1.f;}
 ++candidate_writes;throw std::runtime_error(std::string("injected after write ")+point);
}
namespace strata::core::portable_detail {
struct StrataVkNativeHcPending{}; struct StrataVkNativeHc{}; struct StrataVkNativeHcMix{};
StrataVkNativeHcPending* strata_vk_native_hc_pending_create(ggml_backend_t,const char*,size_t b){++hc_creates[0];hc_budgets[0]=b;auto i=factory_enter("hc pending");return factory_return(new StrataVkNativeHcPending,i);}
size_t strata_vk_native_hc_pending_bytes(const StrataVkNativeHcPending*){return 113;}
void strata_vk_native_hc_pending_destroy(StrataVkNativeHcPending*p)noexcept{factory_destroy(p);delete p;}
void strata_vk_native_hc_pending_run(StrataVkNativeHcPending*,const ggml_tensor*,const ggml_tensor*,const ggml_tensor*,ggml_tensor*,ggml_tensor*,ggml_tensor*){++hc_runs[0];throw std::runtime_error("unexpected pending");}
StrataVkNativeHc* strata_vk_native_hc_create(ggml_backend_t,const char*,size_t b){++hc_creates[1];hc_budgets[1]=b;auto i=factory_enter("hc norm");return factory_return(new StrataVkNativeHc,i);}
size_t strata_vk_native_hc_bytes(const StrataVkNativeHc*){return 227;}
void strata_vk_native_hc_destroy(StrataVkNativeHc*p)noexcept{factory_destroy(p);delete p;}
void strata_vk_native_hc_run(StrataVkNativeHc*,const ggml_tensor*x,const ggml_tensor*w,const ggml_tensor*iw,ggml_tensor*out,ggml_tensor*rs,ggml_tensor*inject,uint32_t n){++hc_runs[1];write_then_throw("hc norm written",{out,rs,inject},101.f);::inject("hc norm");check(n==1||n==2,"HC norm rows supported");verify_hc_role(w,0);verify_hc_role(iw,1);check(ggml_nelements(x)==n*10240,"HC norm rows");std::vector<float> v(n*10240);ggml_backend_tensor_get(x,v.data(),0,v.size()*4);for(auto& f:v)f+=10;ggml_backend_tensor_set(out,v.data(),0,v.size()*4);fill_hc(rs,2);fill_hc(inject,3);}
StrataVkNativeHcMix* strata_vk_native_hc_mix_create(ggml_backend_t,const char*,size_t b){++hc_creates[2];hc_budgets[2]=b;auto i=factory_enter("hc mix");return factory_return(new StrataVkNativeHcMix,i);}
size_t strata_vk_native_hc_mix_bytes(const StrataVkNativeHcMix*){return 331;}
void strata_vk_native_hc_mix_destroy(StrataVkNativeHcMix*p)noexcept{factory_destroy(p);delete p;}
void strata_vk_native_hc_mix_run(StrataVkNativeHcMix*,const ggml_tensor*x,const ggml_tensor*d,const ggml_tensor*u,ggml_tensor*lp,ggml_tensor*low,ggml_tensor*le,ggml_tensor*up,ggml_tensor*g,ggml_tensor*ue,ggml_tensor*mixed){++hc_runs[2];write_then_throw("hc mix written",{lp,low,le,up,g,ue,mixed},201.f);inject("hc mix");verify_hc_role(d,2);verify_hc_role(u,3);size_t n=ggml_nelements(mixed)/2560;check(n==1||n==2,"HC mix rows supported");std::vector<float> v(n*10240,-1234.f),o(n*2560,-1234.f);ggml_backend_tensor_get(x,v.data(),0,v.size()*4);for(size_t r=0;r<n;++r)for(size_t i=0;i<2560;++i)o[r*2560+i]=v[r*10240+i]+20;for(auto*t:{lp,low,le,up,g,ue})fill_hc(t,4);ggml_backend_tensor_set(mixed,o.data(),0,o.size()*4);}
struct StrataVkNativeStages{}; struct StrataVkQ8Wo{}; struct StrataVkNativeShared{};
StrataVkNativeStages* strata_vk_native_stages_create_quant_rcp_observation(ggml_backend_t,const char* p,size_t){++executor_creates;auto i=factory_enter("key quant");std::ofstream(p)<<"mock";return factory_return(new StrataVkNativeStages,i);}
size_t strata_vk_native_stages_bytes(const StrataVkNativeStages*){return 512;}
void strata_vk_native_stages_destroy(StrataVkNativeStages*p)noexcept{factory_destroy(p);delete p;}
StrataVkQ8Wo* strata_vk_head_vocab_create(ggml_backend_t,const char*p,size_t){++executor_creates;auto i=factory_enter("key dot");std::ofstream(p)<<"mock";return factory_return(new StrataVkQ8Wo,i);}
size_t strata_vk_q8_wo_bytes(const StrataVkQ8Wo*){return 512;}
void strata_vk_q8_wo_destroy(StrataVkQ8Wo*p)noexcept{factory_destroy(p);delete p;}
void strata_vk_ple_key_quantize_bounded(StrataVkNativeStages*,const ggml_tensor*x,ggml_tensor*q,size_t t){inject("quant");check(t==1||t==2,"quant rows");std::vector<float>a(t*2560);ggml_backend_tensor_get(x,a.data(),0,a.size()*4);check(a==expected_embedding,"gather embedding");std::vector<uint8_t> z(t*2880);ggml_backend_tensor_set(q,z.data(),0,z.size());++quant;}
void strata_vk_ple_key_run_bounded(StrataVkQ8Wo*,const ggml_tensor*,const ggml_tensor*,ggml_tensor*out,size_t rows,size_t t){inject("dot");check((t==1||t==2)&&rows==10240,"key rows");std::vector<float> a(t*rows,-1234.f);for(size_t r=0;r<t;++r)for(size_t i=0;i<rows;++i)a[r*rows+i]=1.f+expected_embedding[r*2560];ggml_backend_tensor_set(out,a.data(),0,a.size()*4);++dot;}
StrataVkNativeShared* strata_vk_ple_value_bf16_create(ggml_backend_t,const char*,size_t){++executor_creates;auto i=factory_enter("value");return factory_return(new StrataVkNativeShared,i);}
void strata_vk_native_shared_destroy(StrataVkNativeShared*p)noexcept{factory_destroy(p);delete p;}
size_t strata_vk_native_shared_bytes(const StrataVkNativeShared*){return 0;}
void strata_vk_native_shared_run(StrataVkNativeShared*,uint32_t mode,const ggml_tensor*weight_for_value,const ggml_tensor*x,ggml_tensor*out,ggml_tensor*,uint32_t ni,uint32_t no,uint32_t t){inject("value");check(mode==1&&ni==2560&&no==2560&&(t==1||t==2),"value rows");std::vector<float>a(t*ni);ggml_backend_tensor_get(x,a.data(),0,a.size()*4);check(a==expected_embedding,"value gather");std::vector<uint16_t>w(size_t(ni)*no);ggml_backend_tensor_get(weight_for_value,w.data(),0,w.size()*2);std::vector<float>v;for(size_t r=0;r<t;++r){auto row=native_bf16_matrix_control(a.data()+r*ni,w.data(),ni);v.insert(v.end(),row.begin(),row.end());}ggml_backend_tensor_set(out,v.data(),0,v.size()*4);++value;}
struct StrataVkNativePleGroupedNorm{size_t bytes;uint64_t calls=0;};
struct StrataVkNativePleGate{size_t bytes,budget;uint64_t calls=0;};
static size_t mock_bytes(ggml_backend_t b){auto c=std::unique_ptr<ggml_context,decltype(&ggml_free)>(ggml_init({4096,nullptr,true}),ggml_free);ggml_new_tensor_1d(c.get(),GGML_TYPE_I32,1);return ggml_backend_alloc_ctx_tensors_from_buft_size(c.get(),ggml_backend_get_default_buffer_type(b));}
StrataVkNativePleGroupedNorm* strata_vk_native_ple_grouped_norm_create(ggml_backend_t b,const char*,size_t){++executor_creates;auto i=factory_enter("grouped norm");return factory_return(new StrataVkNativePleGroupedNorm{mock_bytes(b)},i);}
void strata_vk_native_ple_grouped_norm_destroy(StrataVkNativePleGroupedNorm*p)noexcept{factory_destroy(p);delete p;}
size_t strata_vk_native_ple_grouped_norm_bytes(const StrataVkNativePleGroupedNorm*p){return p->bytes;}
uint64_t strata_vk_native_ple_grouped_norm_calls(const StrataVkNativePleGroupedNorm*p)noexcept{return p->calls;}
uint32_t strata_vk_native_ple_grouped_norm_run(StrataVkNativePleGroupedNorm*p,const ggml_tensor*in,const ggml_tensor*,ggml_tensor*out,ggml_tensor*){++norm_call; inject(norm_call==1?"key norm":norm_call==2?"query norm":"value norm");size_t count=ggml_nelements(in);check(count==10240||count==2*10240,"norm rows");std::vector<float>a(count);ggml_backend_tensor_get(in,a.data(),0,a.size()*4);ggml_backend_tensor_set(out,a.data(),0,a.size()*4);++norm;++p->calls;return 0;}
StrataVkNativePleGate* strata_vk_native_ple_gate_create(ggml_backend_t b,const char*,size_t budget){++executor_creates;auto i=factory_enter("gate");return factory_return(new StrataVkNativePleGate{mock_bytes(b),budget},i);}
void strata_vk_native_ple_gate_destroy(StrataVkNativePleGate*p)noexcept{factory_destroy(p);delete p;}
size_t strata_vk_native_ple_gate_bytes(const StrataVkNativePleGate*p){return p->bytes;}
uint64_t strata_vk_native_ple_gate_calls(const StrataVkNativePleGate*p)noexcept{return p->calls;}
uint32_t strata_vk_native_ple_gate_run(StrataVkNativePleGate*p,const ggml_tensor*,const ggml_tensor*,ggml_tensor*g,ggml_tensor*d){inject("gate");size_t count=ggml_nelements(g);check(count==4||count==8,"gate rows");std::vector<float>a(count,0.5f),z(ggml_nbytes(d)/4);ggml_backend_tensor_set(g,a.data(),0,a.size()*4);ggml_backend_tensor_set(d,z.data(),0,z.size()*4);++gate;++p->calls;return 0;}
struct StrataVkNativePleConv{size_t bytes;};
StrataVkNativePleConv* strata_vk_native_ple_conv_create(ggml_backend_t b,const char*,size_t){++executor_creates;auto i=factory_enter("conv");return factory_return(new StrataVkNativePleConv{mock_bytes(b)},i);}
void strata_vk_native_ple_conv_destroy(StrataVkNativePleConv*p)noexcept{factory_destroy(p);delete p;}
size_t strata_vk_native_ple_conv_bytes(const StrataVkNativePleConv*p){return p->bytes;}
void strata_vk_native_ple_conv_run(StrataVkNativePleConv*,ggml_tensor*const t[10]){write_then_throw("conv written",{t[5],t[6],t[7],t[8],t[9]},301.f);inject("conv");size_t count=ggml_nelements(t[1]);check(count==10240||count==20480,"conv rows");size_t rows=count/10240;std::vector<float>h(92160),n(count),x(count),g(count),next(92160,-1234.f),c(count,-1234.f),y(count,-1234.f),pre(count,-1234.f),e(count,-1234.f);ggml_backend_tensor_get(t[0],h.data(),0,h.size()*4);ggml_backend_tensor_get(t[1],n.data(),0,n.size()*4);ggml_backend_tensor_get(t[3],x.data(),0,x.size()*4);ggml_backend_tensor_get(t[4],g.data(),0,g.size()*4);for(size_t k=0;k<10240;++k){for(size_t r=0;r<rows;++r){size_t i=r*10240+k;pre[i]=((h[9*k+r]+h[9*k+r+3])+h[9*k+r+6])+n[i];e[i]=std::exp(-pre[i]);c[i]=pre[i]/(1+e[i]);y[i]=x[i]+(g[i]+c[i]);}for(size_t j=0;j<9-rows;++j)next[9*k+j]=h[9*k+j+rows];for(size_t r=0;r<rows;++r)next[9*k+9-rows+r]=n[r*10240+k];}ggml_backend_tensor_set(t[5],next.data(),0,next.size()*4);std::vector<float>*o[]={&c,&y,&pre,&e};for(size_t i=6;i<10;++i)ggml_backend_tensor_set(t[i],o[i-6]->data(),0,o[i-6]->size()*4);++conv;}
}
int main(){try{
 std::string error;auto table=ple_fixture::small_table(error);check(bool(table),error.c_str());std::vector<std::array<uint32_t,16>> rows;std::vector<std::array<int32_t,2>> predecessors{{3,5},{5,7}};check(table->gather_batch({7,9},predecessors,rows,expected_embedding,error),error.c_str());
 auto b=std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)>(ggml_backend_cpu_init(),ggml_backend_free);auto c=std::unique_ptr<ggml_context,decltype(&ggml_free)>(ggml_init({16384,nullptr,true}),ggml_free);auto*k=ggml_new_tensor_2d(c.get(),GGML_TYPE_Q8_0,2560,10240);auto*v=ggml_new_tensor_1d(c.get(),GGML_TYPE_I32,2560*2560/2);auto buf=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>(ggml_backend_alloc_ctx_tensors(c.get(),b.get()),ggml_backend_buffer_free);std::vector<uint8_t> kr(ggml_nbytes(k));std::vector<uint16_t> vr(2560*2560),cw(40960,0x3c00);vr[0]=0x3f80;vr[2560]=0x4000;ggml_backend_tensor_set(k,kr.data(),0,kr.size());ggml_backend_tensor_set(v,vr.data(),0,vr.size()*2);
 std::array<std::vector<float>,3> gam{std::vector<float>(10240,1),std::vector<float>(10240,1),std::vector<float>(10240,1)};auto root=std::filesystem::temp_directory_path()/("ple-integration-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));PortableNativePleExecutor::Roles roles{b.get(),table.get(),k,kr.data(),kr.size(),v,vr.data(),vr.size(),{gam[0].data(),gam[1].data(),gam[2].data()},cw.data(),cw.size()*2,1,root.string()};PortableNativePleExecutor exec(roles,PortableNativePleExecutor::estimate(b.get()));using Tx=PortableNativePleTransaction;Tx tx(10240,10240,2000000,[&](const Tx::Request&r,const Tx::OutputView& o){check(r.predecessors==predecessors,"transaction predecessors");norm_call=0;exec(r,o);},{3,5});std::vector<float> input(20480);for(size_t i=0;i<input.size();++i)input[i]=i<10240?0.25f:0.75f;auto candidate=tx.stage(0,{7,9},input);check(candidate.predecessors()==predecessors,"candidate predecessors");check(quant==1&&dot==1&&value==1&&norm==3&&gate==1&&conv==1,"native stage counts");auto out=candidate.residual(),hist=candidate.history();for(size_t r=0;r<2;++r){for(size_t i=0;i<10240;++i){size_t col=i%2560;float n=0.5f*expected_embedding[r*2560]*(col==0?1.f:col==1?2.f:0.f),want=input[r*10240+i]+n+n/(1+std::exp(-n));check(std::fabs(out[r*10240+i]-want)<1e-6f,"row output");check(hist[9*i+7+r]==n,"history append");}}check(tx.position()==0&&tx.history()[7]==0,"candidate mutated committed state");
 using HC=PortableNativeHcComposed; auto hcctx=std::unique_ptr<ggml_context,decltype(&ggml_free)>(ggml_init({8192,nullptr,true}),ggml_free);std::array<ggml_tensor*,4> src={ggml_new_tensor_1d(hcctx.get(),GGML_TYPE_F32,10240),ggml_new_tensor_1d(hcctx.get(),GGML_TYPE_I32,20480),ggml_new_tensor_1d(hcctx.get(),GGML_TYPE_I32,1638400),ggml_new_tensor_1d(hcctx.get(),GGML_TYPE_I32,1638400)};auto hcbuf=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>(ggml_backend_alloc_ctx_tensors(hcctx.get(),b.get()),ggml_backend_buffer_free);check(bool(hcbuf),"HC source allocation");for(size_t i=0;i<4;++i){hc_roles[i].assign(ggml_nbytes(src[i]),0x31+i);ggml_backend_tensor_set(src[i],hc_roles[i].data(),0,hc_roles[i].size());}
 // Exercise real composition and its empty-Dispatch factory wiring. Each probe
 // gets a fresh directory because constructors intentionally preserve receipts.
 {
  const std::vector<std::string> order={"key quant","key dot","value","gate","grouped norm","conv","value","gate","grouped norm","conv","hc pending","hc norm","hc mix"};
  std::array<const ggml_tensor*,4> sources{src[0],src[1],src[2],src[3]};
  for(size_t failure=0;failure<order.size();++failure){
   auto probe_roles=roles;probe_roles.directory=root.string()+"-factory-"+std::to_string(failure);
   factory_calls.clear();factory_destroyed.clear();check(factory_owners.empty(),"stale factory owner");factory_fail=failure;factory_probe=true;
   bool propagated=false;
   try{PortableNativePleComposition rejected(probe_roles,sources,b.get(),{113,227,331},SIZE_MAX,"factory-hc");}
   catch(const std::runtime_error& e){propagated=std::string(e.what())=="injected factory failure";}
   factory_probe=false;
   check(propagated,"factory exception not propagated");
   check(factory_calls==std::vector<std::string>(order.begin(),order.begin()+failure+1),"factory prefix mismatch");
   check(factory_owners.empty(),"factory owner leaked");
   for(size_t i=0;i<factory_destroyed.size();++i)check(factory_destroyed[i]==(i<failure?1u:0u),"factory destruction count");
  }
  hc_creates={};
 }
 constexpr size_t hc_budget=20*1024*1024;HC hc(b.get(),src[0],src[1],src[2],src[3],hc_budget,"mock-receipt");check(hc_creates==std::array<int,3>{1,1,1},"HC factory calls");size_t scratch=hc.bytes()-hc.role_bank()->bytes()-113-227-331;check(hc_budgets[0]==hc_budget-hc.role_bank()->bytes()-scratch&&hc_budgets[1]==hc_budgets[0]-113&&hc_budgets[2]==hc_budgets[1]-227,"HC budgets");
 // Independent executors, transactions and empty-Dispatch HC instances.
 auto roles_batch=roles, roles_serial=roles; roles_batch.directory=root.string()+"-batch"; roles_serial.directory=root.string()+"-serial";
 PortableNativePleExecutor batch_exec(roles_batch,PortableNativePleExecutor::estimate(b.get())), serial_exec(roles_serial,PortableNativePleExecutor::estimate(b.get()));
 HC batch_hc(b.get(),src[0],src[1],src[2],src[3],hc_budget,"mock-batch"), serial_hc(b.get(),src[0],src[1],src[2],src[3],hc_budget,"mock-serial");
 struct Results {std::vector<float> ple,normalized,mixed;std::vector<std::array<int32_t,2>> prev;};
 auto exercise=[&](PortableNativePleExecutor& executor,HC& consumer,bool batched){
   Results result; Tx transaction(10240,10240,2000000,[&](const Tx::Request& r,const Tx::OutputView& out){std::vector<std::array<uint32_t,16>> indices;std::string err;check(table->gather_batch(r.tokens,r.predecessors,indices,expected_embedding,err),err.c_str());norm_call=0;executor(r,out);for(size_t row=0;row<r.tokens.size();++row)for(size_t c=0;c<10240;++c){float sum=((r.history[9*c+row]+r.history[9*c+row+3])+r.history[9*c+row+6])+out.normalized[row*10240+c];float expected=r.residual_input[row*10240+c]+(out.normalized[row*10240+c]+sum/(1+std::exp(-sum)));check(std::isfinite(out.residual[row*10240+c])&&std::fabs(out.residual[row*10240+c]-expected)<1e-6f,"independent convolution oracle");}},{3,5});
   for(int32_t token:std::vector<int32_t>{1,1,1,1,1,1,1,3,5}){auto warm=transaction.stage(transaction.position(),{token},std::vector<float>(10240,0.125f));transaction.commit(warm,true);}check(transaction.previous()==std::array<int32_t,2>{3,5},"warmup predecessors");bool nonzero=false;for(size_t c=0;c<10240;++c)for(size_t j:{size_t(0),size_t(1),size_t(3),size_t(4),size_t(6),size_t(7)})nonzero|=transaction.history()[9*c+j]!=0;check(nonzero,"warmup history taps");
   auto step=[&](std::vector<int32_t> tokens){size_t start=transaction.position(), count=tokens.size();std::vector<float> in(count*10240,-987.f);for(size_t r=0;r<count;++r)for(size_t i=0;i<10240;++i)in[r*10240+i]=0.25f+0.25f*(start+r)+float(i%13)*0.001f;
     auto q=quant, d=dot, vcount=value, n=norm, g=gate, cv=conv;auto hc_before=hc_runs;
     auto candidate=transaction.stage(start,tokens,in);auto p=candidate.predecessors();result.prev.insert(result.prev.end(),p.begin(),p.end());auto outrow=candidate.residual();result.ple.insert(result.ple.end(),outrow.begin(),outrow.end());
     check(quant==q+1&&dot==d+1&&value==vcount+1&&norm==n+3&&gate==g+1&&conv==cv+1,"per-batch PLE stage calls");
     auto collect=[&](const HC::View& view,uint32_t rows){check(rows==count&&view.rows()==count,"comparison HC rows");check(view.read(HC::View::Field::residual)==outrow,"comparison HC residual");auto a=view.read(HC::View::Field::normalized),m=view.read(HC::View::Field::mixed);for(size_t r=0;r<count;++r)for(size_t i=0;i<2560;++i){check(a[r*10240+i]==outrow[r*10240+i]+10.f,"comparison HC norm");check(m[r*2560+i]==a[r*10240+i]+20.f,"comparison HC mix");}result.normalized.insert(result.normalized.end(),a.begin(),a.end());result.mixed.insert(result.mixed.end(),m.begin(),m.end());};
     consumer.run(1,start,transaction,candidate,collect,[](const HC::View&,uint32_t){});check(hc_runs[0]==hc_before[0]&&hc_runs[1]==hc_before[1]+1&&hc_runs[2]==hc_before[2]+1,"per-batch HC stage calls");
   };
   if(batched){step({7,9});step({11});}else{step({7});step({9});step({11});}
   check(transaction.position()==12&&transaction.previous()==std::array<int32_t,2>{9,11},"comparison final position");
   return std::make_pair(result,transaction.history());
 };
 auto batch=exercise(batch_exec,batch_hc,true), serial=exercise(serial_exec,serial_hc,false);
 check(batch.first.prev==std::vector<std::array<int32_t,2>>{{3,5},{5,7},{7,9}}&&batch.first.prev==serial.first.prev,"comparison predecessors");
 check(batch.first.ple==serial.first.ple&&batch.first.normalized==serial.first.normalized&&batch.first.mixed==serial.first.mixed,"ordered PLE and HC outputs");
 check(batch.second==serial.second&&batch.second.size()==9*10240,"final channel-major history");
 hcbuf.reset();hcctx.reset();
 check(table->gather_batch({7,9},predecessors,rows,expected_embedding,error),error.c_str());
 auto capture=[&](const HC::View& view,uint32_t n){check(n==2&&view.rows()==2,"HC rows");auto residual=view.read(HC::View::Field::residual),normalized=view.read(HC::View::Field::normalized),mixed=view.read(HC::View::Field::mixed);check(residual==out,"HC materialized residual");for(size_t r=0;r<2;++r)for(size_t i=0;i<2560;++i){check(normalized[r*10240+i]==out[r*10240+i]+10,"HC normalized");check(mixed[r*2560+i]==normalized[r*10240+i]+20,"HC mixed");}};auto noop=[](const HC::View&,uint32_t){};
 bool failed=false;try{hc.run(0,0,tx,candidate,capture,noop);}catch(const std::runtime_error&){failed=true;}check(failed&&tx.position()==0&&hc.rows()==0,"wrong boundary publication");hc.run(1,0,tx,candidate,capture,noop);check(tx.position()==2&&tx.history()==hist&&hc.rows()==2&&hc_runs==std::array<int,3>{0,6,6},"HC commit and dispatch");
 check(tx.previous()==std::array<int32_t,2>{7,9}&&!tx.live(candidate),"committed token identity and candidate expiry");
 // Retain an actual publication and check that failures never overwrite it.
 std::shared_ptr<HC::View> prior;
 tx.reset({3,5}); auto seed=tx.stage(0,{7,9},input);
 hc.run(1,0,tx,seed,[&](const HC::View& v,uint32_t){prior=std::make_shared<HC::View>(v);},noop);
 auto published=prior->read(HC::View::Field::mixed);
 auto published_residual=prior->read(HC::View::Field::residual), published_normalized=prior->read(HC::View::Field::normalized);
 auto committed=tx.history(); auto prev=tx.previous();
 auto unchanged=[&](){check(tx.position()==2&&tx.previous()==prev&&tx.history()==committed,"failure changed committed PLE");check(hc.rows()==2&&prior->read(HC::View::Field::mixed)==published&&prior->read(HC::View::Field::residual)==published_residual&&prior->read(HC::View::Field::normalized)==published_normalized,"failure changed HC publication");};
 auto throws=[&](auto f){bool threw=false;try{f();}catch(const std::exception&){threw=true;}check(threw,"expected failure");};
 // The next request uses distinct predecessor rows and exercises both row counts.
 auto next_execute=[&](const Tx::Request& r,const Tx::OutputView& out){std::vector<std::array<uint32_t,16>> indices;std::string err;check(table->gather_batch(r.tokens,r.predecessors,indices,expected_embedding,err),err.c_str());norm_call=0;exec(r,out);};
 // Use a fresh transaction with the same committed state established through the real executor.
 Tx negative(10240,10240,2000000,next_execute,{3,5}); auto initial=negative.stage(0,{7,9},input);negative.commit(initial,true);
 auto neg_unchanged=[&](){check(negative.position()==2&&negative.previous()==prev&&negative.history()==committed,"negative PLE state");unchanged();};
 for(const char* point:{"quant","dot","value","key norm","query norm","gate","value norm","conv"}) for(size_t count:{size_t(1),size_t(2)}) {
   fail_at=point;throws([&]{negative.stage(2,count==1?std::vector<int32_t>{11}:std::vector<int32_t>{11,13},std::vector<float>(count*10240,0.5f));});fail_at.clear();neg_unchanged();
 }
 // Convolution writes all candidate outputs before failing, for both row counts.
 for(size_t count:{size_t(1),size_t(2)}) {
   auto before=candidate_writes; fail_at="conv written";
   throws([&]{negative.stage(2,count==1?std::vector<int32_t>{11}:std::vector<int32_t>{11,13},std::vector<float>(count*10240,0.5f));});
   fail_at.clear(); check(candidate_writes==before+1,"convolution candidate write injection"); neg_unchanged();
   auto retry=negative.stage(2,count==1?std::vector<int32_t>{11}:std::vector<int32_t>{11,13},std::vector<float>(count*10240,0.5f));
   check(negative.live(retry),"convolution failure retry"); negative.abort(retry);
   throws([&]{retry.history();}); neg_unchanged();
 }
 // Fail the actual gated owner's backend compute after a successful gate, for both row sets.
 for(size_t count:{size_t(1),size_t(2)}) {
   auto gates_before=gate, norms_before=norm, conv_before=conv, failures_before=gated_failures;
   fail_gated_compute=true;
   throws([&]{negative.stage(2,count==1?std::vector<int32_t>{11}:std::vector<int32_t>{11,13},std::vector<float>(count*10240,0.5f));});
   fail_gated_compute=false;
   check(gated_failures==failures_before+1&&gate==gates_before+1&&norm==norms_before+2&&conv==conv_before,"gated compute failure stage ordering");
   neg_unchanged();
   auto retry=negative.stage(2,count==1?std::vector<int32_t>{11}:std::vector<int32_t>{11,13},std::vector<float>(count*10240,0.5f));
   check(negative.live(retry),"gated failure retry");negative.abort(retry);neg_unchanged();
 }
 auto bad=input;bad[10240]=std::numeric_limits<float>::quiet_NaN();
 auto dispatch_before=quant;throws([&]{negative.stage(2,{11,-1},input);});throws([&]{negative.stage(2,{11,13},bad);});throws([&]{negative.stage(2,{11,13},std::vector<float>(10240));});check(quant==dispatch_before,"malformed input dispatched");neg_unchanged();
 auto budget_roles=roles;budget_roles.directory=root.string()+"-budget";throws([&]{PortableNativePleExecutor rejected(budget_roles,PortableNativePleExecutor::estimate(b.get())-1);});throws([&]{Tx rejected(10240,10240,1,next_execute,{3,5});});check(quant==dispatch_before&&!std::filesystem::exists(budget_roles.directory),"budget dispatched or created directory");
 auto invalid_roles=budget_roles; invalid_roles.conv_bytes=0;
 throws([&]{PortableNativePleExecutor rejected(invalid_roles,SIZE_MAX);});
 invalid_roles=budget_roles; invalid_roles.key_bytes=0;
 throws([&]{PortableNativePleExecutor rejected(invalid_roles,SIZE_MAX);});
 check(quant==dispatch_before&&!std::filesystem::exists(budget_roles.directory),"invalid descriptors dispatched or created directory");
 auto pending=negative.stage(2,{11,13},input);check(pending.predecessors()==std::vector<std::array<int32_t,2>>{{7,9},{9,11}},"staged predecessor rows");auto staged_history=pending.history();neg_unchanged();
 throws([&]{hc.run(0,2,negative,pending,noop,noop);});throws([&]{hc.run(1,3,negative,pending,noop,noop);});throws([&]{hc.run(1,2,tx,pending,noop,noop);});
 for(const char* point:{"hc norm","hc mix"}){fail_at=point;throws([&]{hc.run(1,2,negative,pending,noop,noop);});fail_at.clear();check(negative.live(pending)&&pending.history()==staged_history,"HC failure lost candidate");neg_unchanged();}
 for(const char* point:{"hc norm written","hc mix written"}) {
   auto before=candidate_writes; int calls=0; fail_at=point;
   throws([&]{hc.run(1,2,negative,pending,[&](const HC::View&,uint32_t){++calls;},noop);});
   fail_at.clear(); check(candidate_writes==before+1 && calls==0,"HC candidate write injection/early publication");
   check(negative.live(pending)&&pending.history()==staged_history,"HC written failure lost candidate"); neg_unchanged();
   // Dispatch failure permits retry; consumer failure invalidates its view and poisons HC.
   std::shared_ptr<HC::View> failed_view;
   throws([&]{hc.run(1,2,negative,pending,[&](const HC::View& v,uint32_t){failed_view=std::make_shared<HC::View>(v);throw std::runtime_error("consumer");},noop);});
   check(bool(failed_view),"HC retry did not reach consumer"); throws([&]{failed_view->read(HC::View::Field::mixed);}); neg_unchanged();
   throws([&]{hc.run(1,2,negative,pending,noop,noop);}); hc.reset();
   throws([&]{prior->read(HC::View::Field::mixed);}); check(hc.rows()==0,"HC reset after consumer failure");
   tx.reset({3,5}); check(table->gather_batch({7,9},predecessors,rows,expected_embedding,error),error.c_str());
   auto restore=tx.stage(0,{7,9},input); hc.run(1,0,tx,restore,[&](const HC::View& v,uint32_t){prior=std::make_shared<HC::View>(v);},noop);
   published=prior->read(HC::View::Field::mixed); committed=tx.history(); prev=tx.previous(); neg_unchanged();
 }
 for(int which=0;which<2;++which){int effects=0;auto first=[&](const HC::View&,uint32_t){++effects;if(which==0)throw std::runtime_error("first consumer");};auto second=[&](const HC::View&,uint32_t){++effects;throw std::runtime_error("second consumer");};throws([&]{hc.run(1,2,negative,pending,first,second);});check(effects==which+1,"consumer side effects");neg_unchanged();throws([&]{hc.run(1,2,negative,pending,noop,noop);});hc.reset();check(hc.rows()==0,"HC reset");
 // Restore a publication after reset without consuming the pending candidate.
 tx.reset({3,5});check(table->gather_batch({7,9},predecessors,rows,expected_embedding,error),error.c_str());auto restore=tx.stage(0,{7,9},input);hc.run(1,0,tx,restore,[&](const HC::View& v,uint32_t){prior=std::make_shared<HC::View>(v);},noop);published=prior->read(HC::View::Field::mixed);committed=tx.history();prev=tx.previous();
 }
 negative.abort(pending);throws([&]{hc.run(1,2,negative,pending,noop,noop);});negative.reset({3,5});auto replay=negative.stage(0,{7,9},input);check(replay.history()==hist&&replay.residual()==out&&replay.predecessors()==predecessors,"reset replay differs");negative.commit(replay,true);check(negative.history()==hist&&negative.previous()==std::array<int32_t,2>{7,9},"replay commit");
 tx.reset({3,5});check(tx.position()==0&&tx.previous()==std::array<int32_t,2>{3,5},"reset position and predecessors");
 for(float x:tx.history())check(x==0.0f,"reset history");
 std::cout<<"PLE T2+T1 versus serial integration passed\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
