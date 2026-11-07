#include "portable_native_gdn_core_rows.hpp"
#include "ggml-cpu.h"
#include <chrono>
#include <cstdio>
#include <fstream>
using namespace strata::core::portable_detail;
namespace {
void need(bool b,const char*m){if(!b)throw std::runtime_error(m);}
int fail=0;size_t conv_calls=0,ab_calls=0,recur_calls=0;
std::vector<float> get(const ggml_tensor*t){std::vector<float>v(ggml_nbytes(t)/4);ggml_backend_tensor_get(t,v.data(),0,v.size()*4);return v;}
void put(ggml_tensor*t,const std::vector<float>&v){need(ggml_nbytes(t)==v.size()*4,"mock span");ggml_backend_tensor_set(t,v.data(),0,v.size()*4);}
}
namespace strata::core::portable_detail {
struct StrataVkNativeGdnConvRows{};struct StrataVkNativeGdnAbRows{};struct StrataVkNativeGdnRows{};
StrataVkNativeGdnConvRows* strata_vk_native_gdn_conv_rows_create(ggml_backend_t,const char*p,size_t){std::ofstream(p)<<"mock";if(fail==1)throw std::runtime_error("conv factory");return new StrataVkNativeGdnConvRows;}
void strata_vk_native_gdn_conv_rows_destroy(StrataVkNativeGdnConvRows*p)noexcept{delete p;}size_t strata_vk_native_gdn_conv_rows_bytes(const StrataVkNativeGdnConvRows*){return 512;}
StrataVkNativeGdnAbRows* strata_vk_native_gdn_ab_rows_create(ggml_backend_t,const char*p,size_t){std::ofstream(p)<<"mock";if(fail==2)throw std::runtime_error("AB factory");return new StrataVkNativeGdnAbRows;}
void strata_vk_native_gdn_ab_rows_destroy(StrataVkNativeGdnAbRows*p)noexcept{delete p;}size_t strata_vk_native_gdn_ab_rows_bytes(const StrataVkNativeGdnAbRows*){return 512;}
StrataVkNativeGdnRows* strata_vk_native_gdn_rows_create(ggml_backend_t,const char*p,size_t){std::ofstream(p)<<"mock";if(fail==3)throw std::runtime_error("recurrence factory");return new StrataVkNativeGdnRows;}
void strata_vk_native_gdn_rows_destroy(StrataVkNativeGdnRows*p)noexcept{delete p;}size_t strata_vk_native_gdn_rows_bytes(const StrataVkNativeGdnRows*){return 512;}
void strata_vk_native_gdn_conv_rows_run(StrataVkNativeGdnConvRows*,ggml_tensor*const t[10],uint32_t n){++conv_calls;auto h=get(t[0]),q=get(t[1]);std::vector<float>out(n*10240);for(size_t r=0;r<n;++r)for(size_t c=0;c<10240;++c){out[r*10240+c]=q[r*10240+c]+h[c*3];h[c*3]=h[c*3+1];h[c*3+1]=h[c*3+2];h[c*3+2]=q[r*10240+c];}put(t[3],h);put(t[4],out);if(fail==4)throw std::runtime_error("conv wrote then threw");}
uint32_t strata_vk_native_gdn_ab_rows_run(StrataVkNativeGdnAbRows*,ggml_tensor*const t[8],uint32_t n){++ab_calls;auto mixed=get(t[0]);std::vector<float>gate(n*48),beta(n*48);for(size_t r=0;r<n;++r)for(size_t i=0;i<48;++i){gate[r*48+i]=mixed[r*2560];beta[r*48+i]=1;}put(t[5],gate);put(t[6],beta);return fail==5?1:0;}
void strata_vk_native_gdn_rows_run(StrataVkNativeGdnRows*,ggml_tensor*const t[13],ggml_tensor*,uint32_t n){++recur_calls;auto state=get(t[0]),h=get(t[1]),gate=get(t[2]),z=get(t[4]);std::vector<float>y(n*6144);for(size_t r=0;r<n;++r)for(size_t i=0;i<6144;++i){state[i]+=h[r*10240+i]+gate[r*48]+z[r*6144+i];y[r*6144+i]=state[i];}put(t[6],state);put(t[7],y);if(fail==6)throw std::runtime_error("recurrence wrote then threw");}
}
int main(int argc,char**argv){try{
 need(argc==2,"scratch");std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)>backend(ggml_backend_cpu_init(),ggml_backend_free);std::unique_ptr<ggml_context,decltype(&ggml_free)>ctx(ggml_init({16384,nullptr,true}),ggml_free);
 std::array<ggml_tensor*,6>role{};const size_t counts[]={40960,61440,61440,48,48,128};for(size_t i=0;i<6;++i)role[i]=ggml_new_tensor_1d(ctx.get(),(i==1||i==2)?GGML_TYPE_I32:GGML_TYPE_F32,counts[i]);
 auto*q2=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,20480);auto*z2=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,12288);
 std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>buf(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()),ggml_backend_buffer_free);need(bool(buf),"allocation");
 auto root=std::filesystem::path(argv[1])/("gdn-core-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));std::filesystem::create_directory(root);
 PortableNativeGdnCoreRows::Caps caps{512,512,512};auto budget=PortableNativeGdnCoreRows::estimate(backend.get(),caps);
 for(int f=1;f<=3;++f){fail=f;bool rejected=false;try{PortableNativeGdnCoreRows bad(backend.get(),role,caps,budget,(root/std::to_string(f)).string());}catch(...){rejected=true;}need(rejected,"factory failure");}
 fail=0;PortableNativeGdnCoreRows a(backend.get(),role,caps,budget,(root/"a").string());
 std::vector<float>seed(786432,0.25f),hist(30720,0.5f);a.reset(seed,hist);
 std::vector<float>q(20480),z(12288),mixed(5120);for(size_t r=0;r<2;++r){std::fill(q.begin()+r*10240,q.begin()+(r+1)*10240,float(1+r));std::fill(z.begin()+r*6144,z.begin()+(r+1)*6144,float(3+r));mixed[r*2560]=float(5+r);}put(q2,q);put(z2,z);
 a.stage(mixed,q2,z2,2);auto out=a.output();need(a.committed_state()==seed&&a.committed_history()==hist&&conv_calls==1&&ab_calls==1&&recur_calls==1,"T2 staged or dispatch count");
 a.reset(seed,hist);ggml_tensor outside=*q2;outside.data=reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(q2->buffer))+ggml_backend_buffer_get_size(q2->buffer)-4);bool rejected_span=false;try{a.stage(mixed,&outside,z2,2);}catch(...){rejected_span=true;}need(rejected_span&&!a.poisoned()&&a.committed_state()==seed,"out-of-allocation span accepted");
 auto saved_state=a.committed_state(),saved_history=a.committed_history();for(int f=4;f<=6;++f){a.reset(saved_state,saved_history);fail=f;bool rejected=false;try{a.stage(mixed,q2,z2,2);}catch(...){rejected=true;}need(rejected&&!a.candidate()&&a.poisoned()&&a.committed_state()==saved_state&&a.committed_history()==saved_history,"failed stage changed committed state");}fail=0;a.reset(seed,hist);a.stage(mixed,q2,z2,2);need(a.output()==out,"reset A/B/A");
 std::puts("GDN core rows factory mock and staged banks passed");return 0;
}catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
