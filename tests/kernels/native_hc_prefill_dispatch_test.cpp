// Actual prefill wrapper with transport-only mocks; the shader does not run.
#include "ggml.h"
#include "ggml-backend-impl.h"
#include "strata_vk_native_hc.hpp"
#include "native_hc_layout.hpp"
#include <array>
#include <cfenv>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>
struct MockDevice {struct {struct {uint32_t maxComputeWorkGroupCount[3]={65535,65535,65535},maxComputeWorkGroupSize[3]={1024,1024,64},maxComputeWorkGroupInvocations=1024,maxComputeSharedMemorySize=32768;} limits;} properties;};
struct ggml_backend_vk_context {std::shared_ptr<MockDevice> device=std::make_shared<MockDevice>();};
struct ggml_backend_vk_buffer_context {std::weak_ptr<MockDevice> device;};
namespace strata::core::portable_detail {
static const uint32_t strata_native_hc_prefill_spv[]={1};
enum class NativeComponentAddressing {descriptor};
struct StrataVkLogitsGuard {ggml_backend_t backend;ggml_tensor* status=nullptr;size_t bytes=512;int pipeline=1;};
struct Call {std::vector<const ggml_tensor*> tensors;std::array<uint32_t,9> params;std::array<uint32_t,3> grid;};
static std::vector<Call> calls;static uint32_t fake_status=0;static const ggml_tensor* status_tensor=nullptr;
static std::unique_ptr<StrataVkLogitsGuard> strata_vk_q8_component_owner(ggml_backend_t b,const uint32_t*,size_t,NativeComponentAddressing,const char*,uint32_t n,uint32_t words,const char*,size_t budget){if(n!=7||words!=9||budget<512)throw std::invalid_argument("layout/budget");return std::make_unique<StrataVkLogitsGuard>(StrataVkLogitsGuard{b});}
static bool ggml_backend_buffer_is_vk(ggml_backend_buffer_t){return true;}
static void* mock_base(ggml_backend_buffer_t){return nullptr;}
static size_t mock_size(ggml_backend_buffer_t b){return b->size;}
static uint32_t get_misalign_bytes(ggml_backend_vk_context*,const ggml_tensor*t){return uintptr_t(t->data)%256;}
static void ggml_backend_vk_synchronize(ggml_backend_t){}
static void ggml_pipeline_request_descriptor_sets(ggml_backend_vk_context*,int,int){}
static std::shared_ptr<int> ggml_vk_get_compute_ctx(ggml_backend_vk_context*){return std::make_shared<int>(1);}
static void ggml_vk_sync_buffers(ggml_backend_vk_context*,const std::shared_ptr<int>&){}
static const ggml_tensor* ggml_vk_tensor_subbuffer(ggml_backend_vk_context*,const ggml_tensor*t,bool){return t;}
static void ggml_vk_dispatch_pipeline(ggml_backend_vk_context*,const std::shared_ptr<int>&,int,std::initializer_list<const ggml_tensor*> b,const std::array<uint32_t,9>&p,const std::array<uint32_t,3>&g){calls.push_back({b,p,g});}
static void mock_get(const ggml_tensor*t,void*out,size_t,size_t n){std::memset(out,0,n);if(t==status_tensor)std::memcpy(out,&fake_status,4);}
static void mock_set(ggml_tensor*t,const void*data,size_t offset,size_t n){native_hc_need(t==status_tensor&&offset==0&&n==4&&*static_cast<const uint32_t*>(data)==0,"status reset");}
}
#define STRATA_HC_PREFILL_MOCK_SPV
#define ggml_backend_buffer_get_base mock_base
#define ggml_backend_buffer_get_size mock_size
#define ggml_backend_tensor_get mock_get
#define ggml_backend_tensor_set mock_set
#include "strata_vk_native_hc_prefill.inc"
#undef ggml_backend_tensor_get
#undef ggml_backend_tensor_set
#undef ggml_backend_buffer_get_base
#undef ggml_backend_buffer_get_size
using namespace strata::core::portable_detail;
template<class F> static void reject(F f){size_t n=calls.size();bool caught=false;try{f();}catch(const std::exception&){caught=true;}native_hc_need(caught&&calls.size()==n,"expected pre-dispatch refusal");}
int main(){try{
 ggml_backend_vk_context vk,foreign;ggml_backend backend{};backend.context=&vk;ggml_backend_vk_buffer_context bc,fc;bc.device=vk.device;fc.device=foreign.device;ggml_backend_buffer buffer{},other{};buffer.context=&bc;other.context=&fc;buffer.size=0x20000000;other.size=0x20000000;
 std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({32768,nullptr,true}),ggml_free);
 auto make=[&](ggml_type type,int64_t n,int id){auto*t=ggml_new_tensor_1d(ctx.get(),type,n);t->buffer=&buffer;t->data=(void*)(0x1000000ull*(id+1)+4*(id+1));return t;};
 std::array<ggml_tensor*,6> norm{make(GGML_TYPE_F32,20480,0),make(GGML_TYPE_F32,10240,1),make(GGML_TYPE_F32,20480,2),make(GGML_TYPE_F32,20480,3),make(GGML_TYPE_I32,10240,4),make(GGML_TYPE_F32,8,5)};
 std::array<ggml_tensor*,6> silu{make(GGML_TYPE_F32,640,6),nullptr,make(GGML_TYPE_F32,640,7),make(GGML_TYPE_F32,640,8),make(GGML_TYPE_I32,320,9),nullptr};
 std::array<ggml_tensor*,6> mix{norm[2],make(GGML_TYPE_F32,20480,10),make(GGML_TYPE_F32,5120,11),nullptr,make(GGML_TYPE_I32,2560,12),make(GGML_TYPE_I32,2560,13)};
 auto*s=make(GGML_TYPE_F32,1,14);status_tensor=s;
 reject([&]{auto*p=strata_vk_native_hc_prefill_create(&backend,"mock",511);strata_vk_native_hc_prefill_destroy(p);});
 std::unique_ptr<StrataVkNativeHcPrefill,decltype(&strata_vk_native_hc_prefill_destroy)> p(strata_vk_native_hc_prefill_create(&backend,"mock",512),strata_vk_native_hc_prefill_destroy);p->owner->status=s;
 for(uint32_t mode=0;mode<3;++mode){auto&v=mode==0?norm:mode==1?silu:mix;strata_vk_native_hc_prefill_run(p.get(),mode,2,v);auto&call=calls.back();native_hc_need(call.grid==std::array<uint32_t,3>{mode==0?4u:mode==1?1u:5u,2,1}&&call.params[0]==mode&&call.params[1]==2&&call.tensors[6]==s,"grid/mode/status");for(size_t j=0;j<6;++j)native_hc_need(call.tensors[j]==(v[j]?v[j]:v[0])&&call.params[2+j]==(v[j]?get_misalign_bytes(&vk,v[j])/4:0),"descriptor or offset");
  for(size_t j=0;j<6;++j)if(v[j]){auto n=v[j]->ne[0];v[j]->ne[0]=n-1;reject([&]{strata_vk_native_hc_prefill_run(p.get(),mode,2,v);});v[j]->ne[0]=n;}}
 auto old=norm[2]->data;norm[2]->data=(void*)(uintptr_t(norm[0]->data)+40960);reject([&]{strata_vk_native_hc_prefill_run(p.get(),0,2,norm);});norm[2]->data=old;
 norm[1]->buffer=&other;reject([&]{strata_vk_native_hc_prefill_run(p.get(),0,2,norm);});norm[1]->buffer=&buffer;
 vk.device->properties.limits.maxComputeWorkGroupCount[1]=1;reject([&]{strata_vk_native_hc_prefill_run(p.get(),0,2,norm);});vk.device->properties.limits.maxComputeWorkGroupCount[1]=65535;
 fake_status=1;bool caught=false;try{strata_vk_native_hc_prefill_run(p.get(),0,2,norm);}catch(...){caught=true;}native_hc_need(caught,"device status");fake_status=0;
 for(size_t i:{0u,2u,3u,4u,5u})norm[i]->ne[0]/=2;strata_vk_native_hc_prefill_run(p.get(),0,1,norm);native_hc_need(calls.back().grid==std::array<uint32_t,3>{4,1,1},"T1 grid");
 auto save=norm[0]->data;norm[0]->data=(void*)(buffer.size-4);reject([&]{strata_vk_native_hc_prefill_run(p.get(),0,1,norm);});norm[0]->data=save;
 puts("HC prefill wrapper transport controls passed; no shader execution");return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
