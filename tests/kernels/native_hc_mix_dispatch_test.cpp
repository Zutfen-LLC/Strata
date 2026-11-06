// Exercise the real wrapper with transport-only Vulkan mocks; no shader execution.
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

struct MockDevice { struct { struct { uint32_t maxComputeWorkGroupCount[3]={65535,65535,65535}, maxComputeWorkGroupSize[3]={1024,1024,64}, maxComputeWorkGroupInvocations=1024, maxComputeSharedMemorySize=32768; } limits; } properties; };
struct ggml_backend_vk_context { std::shared_ptr<MockDevice> device=std::make_shared<MockDevice>(); };
struct ggml_backend_vk_buffer_context { std::weak_ptr<MockDevice> device; };
namespace strata::core::portable_detail {
static const uint32_t strata_native_hc_mix_spv[]={1}, strata_native_head_hc_mix_rn_spv[]={2};
enum class NativeComponentAddressing { descriptor };
struct StrataVkLogitsGuard { ggml_backend_t backend; ggml_tensor* status=nullptr; size_t bytes=512; int pipeline=1; };
struct Dispatch { std::vector<const ggml_tensor*> buffers; std::array<uint32_t,9> params; std::array<uint32_t,3> grid; };
static std::vector<Dispatch> calls;
static uint32_t fake_status=0; static const ggml_tensor* status_tensor=nullptr;
static std::unique_ptr<StrataVkLogitsGuard> strata_vk_q8_component_owner(ggml_backend_t b,const uint32_t*,size_t,NativeComponentAddressing,const char*,uint32_t n,uint32_t words,const char*,size_t) { if(n!=7||words!=9) throw std::runtime_error("layout"); return std::make_unique<StrataVkLogitsGuard>(StrataVkLogitsGuard{b}); }
static bool ggml_backend_buffer_is_vk(ggml_backend_buffer_t){return true;}
static uint32_t get_misalign_bytes(ggml_backend_vk_context*,const ggml_tensor* t){return uintptr_t(t->data)%256;}
static void ggml_backend_vk_synchronize(ggml_backend_t){}
static void ggml_pipeline_request_descriptor_sets(ggml_backend_vk_context*,int,int){}
static std::shared_ptr<int> ggml_vk_get_compute_ctx(ggml_backend_vk_context*){return std::make_shared<int>(1);}
static void ggml_vk_sync_buffers(ggml_backend_vk_context*,const std::shared_ptr<int>&){}
static const ggml_tensor* ggml_vk_tensor_subbuffer(ggml_backend_vk_context*,const ggml_tensor* t,bool){return t;}
static void ggml_vk_dispatch_pipeline(ggml_backend_vk_context*,const std::shared_ptr<int>&,int,std::initializer_list<const ggml_tensor*> b,const std::array<uint32_t,9>& p,const std::array<uint32_t,3>& g){calls.push_back({b,p,g});}
static void mock_get(const ggml_tensor* t,void* out,size_t,size_t n){std::memset(out,0,n);if(t==status_tensor)std::memcpy(out,&fake_status,4);}
static void mock_set(ggml_tensor* t,const void* data,size_t offset,size_t n){native_hc_need(t==status_tensor&&offset==0&&n==4&&*static_cast<const uint32_t*>(data)==0,"status reset");}
}
#define STRATA_HC_MIX_MOCK_SPV
#define ggml_backend_tensor_get mock_get
#define ggml_backend_tensor_set mock_set
#include "strata_vk_native_hc_mix.inc"
#undef ggml_backend_tensor_get
#undef ggml_backend_tensor_set
using namespace strata::core::portable_detail;
template<class F> static void reject_before(F f){auto before=calls.size();bool caught=false;try{f();}catch(const std::invalid_argument&){caught=true;}native_hc_need(caught&&calls.size()==before,"expected rejection before dispatch");}
int main(){try {
 ggml_backend_vk_context vk; ggml_backend backend{};backend.context=&vk;ggml_backend_vk_buffer_context bc;bc.device=vk.device;ggml_backend_buffer buffer{};buffer.context=&bc;buffer.size=200000000;
 std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({32768,nullptr,true}),ggml_free);
 const int64_t counts[]={20480,1638400,1638400,640,640,640,20480,20480,20480,5120};
 std::array<ggml_tensor*,10> t{};
 for(size_t i=0;i<t.size();++i){t[i]=ggml_new_tensor_1d(ctx.get(),i==1||i==2?GGML_TYPE_I32:GGML_TYPE_F32,counts[i]);t[i]->buffer=&buffer;t[i]->data=(void*)(0x1000000ull*(i+1)+4*(i+1));}
 auto* s=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,1);s->buffer=&buffer;s->data=(void*)0xb00002c;status_tensor=s;
 std::unique_ptr<StrataVkNativeHcMix,decltype(&strata_vk_native_hc_mix_destroy)> p(strata_vk_native_hc_mix_create(&backend,"mock",512),strata_vk_native_hc_mix_destroy);p->owner->status=s;
 auto run=[&]{strata_vk_native_hc_mix_run(p.get(),t[0],t[1],t[2],t[3],t[4],t[5],t[6],t[7],t[8],t[9]);};
 auto verify=[&](size_t start,uint32_t rows){
  native_hc_need(calls.size()==start+3,"three dispatches required");
  const std::array<std::array<uint32_t,3>,3> grids={{{320,rows,1},{10240,rows,1},{10,rows,1}}};
  const std::array<std::array<uint32_t,9>,3> params={{{0,rows,1,1,2,4,5,6,11},{1,rows,5,5,3,7,8,9,11},{2,rows,1,8,2,4,5,10,11}}};
  const std::array<std::vector<const ggml_tensor*>,3> buffers={{{t[0],t[0],t[1],t[3],t[4],t[5],s},{t[4],t[4],t[2],t[6],t[7],t[8],s},{t[0],t[7],t[1],t[3],t[4],t[9],s}}};
  for(size_t i=0;i<3;++i)native_hc_need(calls[start+i].grid==grids[i]&&calls[start+i].params==params[i]&&calls[start+i].buffers==buffers[i],"grid, params or descriptor order");
 };
 run();verify(0,2);
 // Every bank and both shared weight matrices must have their full span.
 for(auto* x:t){auto n=x->ne[0];x->ne[0]=n-1;reject_before(run);x->ne[0]=n;}
 // Overlap only the second input row, not its first row.
 auto old=t[9]->data;t[9]->data=(void*)(uintptr_t(t[0]->data)+40960);reject_before(run);t[9]->data=old;
 vk.device->properties.limits.maxComputeWorkGroupCount[1]=1;reject_before(run);vk.device->properties.limits.maxComputeWorkGroupCount[1]=65535;
 fake_status=0x25;auto before=calls.size();bool refused=false;try{run();}catch(const NativeHcMixDomainRefusal& e){refused=e.status==0x25;}native_hc_need(refused,"aggregate status accepted or lost");verify(before,2);fake_status=0;
 for(size_t i:{0u,3u,4u,5u,6u,7u,8u,9u})t[i]->ne[0]/=2;
 before=calls.size();run();verify(before,1);
 std::puts("actual HC mix wrapper dispatch/validation passed (mock Vulkan; no shader execution)");return 0;
 }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
