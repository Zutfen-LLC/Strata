#include "portable_native_ffn_rows.hpp"
#include "ggml-cpu.h"
#include <cstdio>
#include <limits>
#include <chrono>
#include <filesystem>
#include <fstream>
using namespace strata::core::portable_detail;
static void check(bool ok,const char* m){if(!ok)throw std::runtime_error(m);}
namespace strata::core::portable_detail {
struct StrataVkNativeRouter{};struct StrataVkNativeShared{};struct StrataVkNativeStages{};struct StrataVkNativeMoeCombine{};
static size_t router_runs=0,shared_runs=0,combine_runs=0;
static void receipt(const char* p){std::ofstream f(p);check(bool(f),"mock receipt");f<<"mock";}
static void fill(ggml_tensor* t,float x){std::vector<float> v(size_t(ggml_nelements(t)),x);ggml_backend_tensor_set(t,v.data(),0,v.size()*4);}
StrataVkNativeRouter* strata_vk_native_router_create(ggml_backend_t,const char* p){receipt(p);return new StrataVkNativeRouter;}
size_t strata_vk_native_router_bytes(const StrataVkNativeRouter*){return 128;}
void strata_vk_native_router_destroy(StrataVkNativeRouter* p)noexcept{delete p;}
void strata_vk_native_router_run(StrataVkNativeRouter*,const ggml_tensor* logits,ggml_tensor* ids,ggml_tensor* weights,uint32_t n){++router_runs;check(logits->ne[1]==n,"native adapter columns");std::vector<int32_t> v(n*10);for(size_t i=0;i<v.size();++i)v[i]=int32_t(9-i%10);ggml_backend_tensor_set(ids,v.data(),0,v.size()*4);fill(weights,.1f);}
StrataVkNativeShared* strata_vk_native_shared_create(ggml_backend_t,const char* p,size_t){receipt(p);return new StrataVkNativeShared;}
size_t strata_vk_native_shared_bytes(const StrataVkNativeShared*){return 128;}
void strata_vk_native_shared_destroy(StrataVkNativeShared* p)noexcept{delete p;}
StrataVkNativeStages* strata_vk_native_stages_create_quant_rn_observation(ggml_backend_t,const char* p,size_t){receipt(p);return new StrataVkNativeStages;}
size_t strata_vk_native_stages_bytes(const StrataVkNativeStages*){return 128;}
void strata_vk_native_stages_destroy(StrataVkNativeStages* p)noexcept{delete p;}
void strata_vk_native_stages_run(StrataVkNativeStages*,uint32_t mode,const ggml_tensor*,const ggml_tensor*,ggml_tensor* hidden,ggml_tensor* packed,uint32_t,const int32_t* starts,size_t count){check(count==2&&starts[0]==0&&(starts[1]==1||starts[1]==2),"native shared rows");if(mode==1)fill(hidden,0);else {std::vector<int32_t> zero(size_t(ggml_nelements(packed)));ggml_backend_tensor_set(packed,zero.data(),0,zero.size()*4);}}
void strata_vk_native_shared_run(StrataVkNativeShared*,uint32_t mode,const ggml_tensor*,const ggml_tensor*,ggml_tensor* out,ggml_tensor* aux,uint32_t,uint32_t,uint32_t){++shared_runs;fill(out,mode==2?1.f:0.f);if(mode==2)fill(aux,.5f);}
StrataVkNativeMoeCombine* strata_vk_native_moe_combine_create(ggml_backend_t,const char* p){receipt(p);return new StrataVkNativeMoeCombine;}
size_t strata_vk_native_moe_combine_bytes(const StrataVkNativeMoeCombine*){return 128;}
void strata_vk_native_moe_combine_destroy(StrataVkNativeMoeCombine* p)noexcept{delete p;}
void strata_vk_native_moe_combine_run(StrataVkNativeMoeCombine*,const ggml_tensor* rows,const ggml_tensor* weights,const ggml_tensor* shared,ggml_tensor* output,uint32_t n){++combine_runs;std::vector<float> r(n*10*2560),w(n*10),v(n*2560);ggml_backend_tensor_get(rows,r.data(),0,r.size()*4);ggml_backend_tensor_get(weights,w.data(),0,w.size()*4);ggml_backend_tensor_get(shared,v.data(),0,v.size()*4);for(size_t t=0;t<n;++t)for(size_t k=0;k<10;++k)for(size_t h=0;h<2560;++h)v[t*2560+h]+=r[(t*10+k)*2560+h]*w[t*10+k];ggml_backend_tensor_set(output,v.data(),0,v.size()*4);}
}
int main(){try{
 std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(),ggml_backend_free);
 std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({8192,nullptr,true}),ggml_free);
 auto* backing=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,2560,513);
 auto* role=ggml_view_2d(ctx.get(),backing,2560,512,backing->nb[1],2560*sizeof(float));
 std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()),ggml_backend_buffer_free);
 check(role->data && role->view_offs==2560*sizeof(float),"view setup");
 auto malformed=*role;malformed.data=nullptr;bool refused=false;
 try{PortableNativeFfnProjection bad(backend.get(),&malformed,16u<<20);}catch(...){refused=true;}check(refused,"null data admitted");
 std::unique_ptr<ggml_context,decltype(&ggml_free)> tiny_ctx(ggml_init({4096,nullptr,true}),ggml_free);
 auto* tiny=ggml_new_tensor_1d(tiny_ctx.get(),GGML_TYPE_F32,1);
 std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> tiny_buf(ggml_backend_alloc_ctx_tensors(tiny_ctx.get(),backend.get()),ggml_backend_buffer_free);
 malformed=*role;malformed.buffer=tiny->buffer;malformed.data=tiny->data;malformed.view_src=nullptr;refused=false;
 try{PortableNativeFfnProjection bad(backend.get(),&malformed,16u<<20);}catch(...){refused=true;}check(refused,"short buffer admitted");
 malformed=*role;malformed.view_src=nullptr;malformed.data=static_cast<char*>(ggml_backend_buffer_get_base(buffer.get()))+ggml_backend_buffer_get_size(buffer.get())-4;refused=false;
 try{PortableNativeFfnProjection bad(backend.get(),&malformed,16u<<20);}catch(...){refused=true;}check(refused,"offset overrun admitted");
 malformed=*role;malformed.view_offs=0;refused=false;
 try{PortableNativeFfnProjection bad(backend.get(),&malformed,16u<<20);}catch(...){refused=true;}check(refused,"inconsistent view admitted");
 std::vector<float> weights(2560*512);for(size_t j=0;j<512;++j){weights[j*2560]=float(j+1);weights[j*2560+1]=2.f;}
 ggml_backend_tensor_set(role,weights.data(),0,weights.size()*4);
 bool short_budget=false;try{PortableNativeFfnProjection no(backend.get(),role,1);}catch(...){short_budget=true;}check(short_budget,"budget accepted");
 PortableNativeFfnProjection p(backend.get(),role,16u<<20);
 std::fill(weights.begin(),weights.end(),0.f);ggml_backend_tensor_set(role,weights.data(),0,weights.size()*4);
 std::vector<float> mixed(5120);mixed[0]=2;mixed[1]=3;mixed[2560]=5;mixed[2561]=-1;
 p.run(mixed,2);auto* out=p.output(2);check(out->ne[0]==512&&out->ne[1]==2&&out->op==GGML_OP_MUL_MAT,"not one two-column matrix");
 std::vector<float> v(1024);ggml_backend_tensor_get(out,v.data(),0,v.size()*4);
 for(size_t j=0;j<512;++j)check(v[j]==2*float(j+1)+6&&v[512+j]==5*float(j+1)-2,"T2 projection mismatch");
 mixed.resize(2560);p.run(mixed,1);v.resize(512);ggml_backend_tensor_get(p.output(1),v.data(),0,v.size()*4);
 for(size_t j=0;j<512;++j)check(v[j]==2*float(j+1)+6,"tail mismatch");
 mixed[0]=std::numeric_limits<float>::infinity();bool failed=false;try{p.run(mixed,1);}catch(...){failed=true;}check(failed,"nonfinite input accepted");
 failed=false;try{p.output(1);}catch(...){failed=true;}check(failed,"failed output remained accessible");
 mixed.assign(5120,0);mixed[0]=2;mixed[2560]=5;size_t calls=0, router_calls=0;
 PortableNativeFfnRows::Dispatch dispatch;
 dispatch.router=[&](size_t n,const ggml_tensor* logits,std::vector<strata::core::PortableRoute>& routes){++router_calls;check(logits->ne[1]==int64_t(n),"router columns");for(size_t t=0;t<n;++t){routes[t].native_router=true;for(int k=0;k<10;++k){routes[t].ids.push_back((9-k)+int(t)*10);routes[t].weights.push_back(float(k+1)/10);}}};
 dispatch.shared=[](size_t n,const ggml_tensor* input){check(input->ne[1]==int64_t(n),"shared columns");return std::vector<float>(n*2560,1.f);};
 dispatch.combine=[](size_t n,const std::vector<float>& rows,const std::vector<float>& weights,const std::vector<float>& shared){auto out=shared;for(size_t t=0;t<n;++t)for(size_t k=0;k<10;++k)for(size_t h=0;h<2560;++h)out[t*2560+h]+=rows[(t*10+k)*2560+h]*weights[t*10+k];return out;};
 PortableNativeFfnRows core(2,p,dispatch);
 strata::core::PortableExpertRowsBatch provider=[&](int64_t layer,const auto& routes,auto& rows,auto&){++calls;check(layer==2&&routes.size()==2&&routes[0].ids[0]==9&&routes[1].ids[0]==19&&routes[1].activation[0]==5,"provider routes");rows.resize(2*10*2560);for(size_t t=0;t<2;++t)for(size_t k=0;k<10;++k)std::fill_n(rows.begin()+(t*10+k)*2560,2560,float(routes[t].ids[k]));return true;};
 auto result=core.run(mixed,2,provider);check(calls==1&&router_calls==1&&result.block.size()==5120,"batch result");
 for(size_t t=0;t<2;++t){float expected=1;for(size_t k=0;k<10;++k)expected+=float(9-k+t*10)*float(k+1)/10;check(std::abs(result.block[t*2560]-expected)<1e-5f,"original-slot weighted result");}
 provider=[](int64_t,const auto&,auto&,auto&){return false;};bool rejected=false;try{core.run(mixed,2,provider);}catch(...){rejected=true;}check(rejected&&core.poisoned(),"provider failure did not poison");
 rejected=false;try{core.run(mixed,2,provider);}catch(...){rejected=true;}check(rejected,"poisoned continuation");core.reset();check(!core.poisoned(),"reset");
 mixed.resize(2560);provider=[&](int64_t,const auto& routes,auto& rows,auto&){++calls;check(routes.size()==1&&routes[0].activation[0]==2,"tail route");rows.assign(10*2560,1.f);return true;};
 auto tail=core.run(mixed,1,provider);check(tail.block.size()==2560&&std::abs(tail.block[0]-6.5f)<1e-5f&&calls==2,"tail result");
 auto root=std::filesystem::temp_directory_path()/("ffn-row-owners-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));check(std::filesystem::create_directory(root),"owner root");
 std::array<std::unique_ptr<PortableNativeRouter>,2> routers;std::array<std::unique_ptr<PortableNativeShared>,2> shared;std::array<std::unique_ptr<PortableNativeMoeCombine>,2> combines;
 std::array<ggml_tensor*,4> shared_weights{tiny,tiny,tiny,tiny};
 for(size_t i=0;i<2;++i){routers[i]=std::make_unique<PortableNativeRouter>(backend.get(),i+1,16u<<20,(root/"router").string());shared[i]=std::make_unique<PortableNativeShared>(backend.get(),2560,32,i+1,shared_weights,16u<<20,(root/"shared").string());combines[i]=std::make_unique<PortableNativeMoeCombine>(backend.get(),i+1,16u<<20,(root/"combine").string());}
 PortableNativeFfnRows real(2,p,PortableNativeFfnRows::native_dispatch({routers[0].get(),routers[1].get()},{shared[0].get(),shared[1].get()},{combines[0].get(),combines[1].get()}));
 size_t native_batches=0;strata::core::PortableExpertRowsBatch native_provider=[&](int64_t,const auto& routes,auto& rows,auto&){++native_batches;for(const auto& r:routes)check(r.native_router&&r.native_shared&&r.ids[0]==9,"native owner route");rows.assign(routes.size()*10*2560,2.f);return true;};
 mixed.resize(5120);mixed[2560]=5;auto actual=real.run(mixed,2,native_provider);check(actual.block.size()==5120&&std::abs(actual.block[0]-3.f)<1e-5f&&native_batches==1&&router_runs==1&&shared_runs==5&&combine_runs==1,"T2 native owner adapter");
 mixed.resize(2560);actual=real.run(mixed,1,native_provider);check(actual.block.size()==2560&&std::abs(actual.block[0]-3.f)<1e-5f&&native_batches==2&&router_runs==2&&shared_runs==10&&combine_runs==2,"T1 native owner adapter");
 std::puts("FFN projection and row core CPU graph/native-owner mocks passed");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
