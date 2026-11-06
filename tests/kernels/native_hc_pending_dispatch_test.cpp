// Actual wrapper, with only Vulkan transport and SPIR-V replaced by a host mock.
#include "ggml.h"
#include "ggml-backend-impl.h"
#include <array>
#include <cfenv>
#include <memory>
#include <stdexcept>
#include <vector>
#include <cstdio>
#include <cstring>
struct MockDevice { struct { struct { uint32_t maxComputeWorkGroupCount[3]={65535,65535,65535}, maxComputeWorkGroupSize[3]={1024,1024,64}, maxComputeWorkGroupInvocations=1024, maxComputeSharedMemorySize=32768; } limits; } properties; };
struct ggml_backend_vk_context { std::shared_ptr<MockDevice> device=std::make_shared<MockDevice>(); };
struct ggml_backend_vk_buffer_context { std::weak_ptr<MockDevice> device; };
namespace strata::core::portable_detail {
static const uint32_t strata_native_hc_pending_spv[]={1}, strata_native_hc_write_spv[]={2};
enum class NativeComponentAddressing { descriptor };
struct StrataVkLogitsGuard { ggml_backend_t backend; ggml_tensor* status=nullptr; size_t bytes=512; int pipeline=1; };
static unsigned submissions=0; static uint32_t fake_status=0; static const ggml_tensor* status_tensor=nullptr;
static std::array<uint32_t,7> observed_offsets; static std::array<uint32_t,3> observed_grid; static std::vector<const ggml_tensor*> observed_buffers;
static std::unique_ptr<StrataVkLogitsGuard> strata_vk_q8_component_owner(ggml_backend_t b,const uint32_t*,size_t,NativeComponentAddressing,const char*,uint32_t n,uint32_t words,const char*,size_t) { if(n!=7||words!=7) throw std::runtime_error("layout"); return std::make_unique<StrataVkLogitsGuard>(StrataVkLogitsGuard{b}); }
static bool ggml_backend_buffer_is_vk(ggml_backend_buffer_t){return true;}
static uint32_t get_misalign_bytes(ggml_backend_vk_context*,const ggml_tensor* t){return uintptr_t(t->data)%256;}
static void ggml_backend_vk_synchronize(ggml_backend_t){}
static void ggml_pipeline_request_descriptor_sets(ggml_backend_vk_context*,int,int){}
static std::shared_ptr<int> ggml_vk_get_compute_ctx(ggml_backend_vk_context*){return std::make_shared<int>(1);}
static void ggml_vk_sync_buffers(ggml_backend_vk_context*,const std::shared_ptr<int>&){}
static const ggml_tensor* ggml_vk_tensor_subbuffer(ggml_backend_vk_context*,const ggml_tensor* t,bool){return t;}
static void ggml_vk_dispatch_pipeline(ggml_backend_vk_context*,const std::shared_ptr<int>&,int,std::initializer_list<const ggml_tensor*> b,const std::array<uint32_t,7>& o,const std::array<uint32_t,3>& g){++submissions; observed_buffers=b; observed_offsets=o; observed_grid=g;}
static void mock_get(const ggml_tensor* t,void* out,size_t,size_t n){std::memset(out,0,n);if(t==status_tensor)std::memcpy(out,&fake_status,4);}
static void mock_set(ggml_tensor*,const void*,size_t,size_t){}
}
#define STRATA_HC_PENDING_MOCK_SPV
#define ggml_backend_tensor_get mock_get
#define ggml_backend_tensor_set mock_set
#include "strata_vk_native_hc_pending.inc"
#undef ggml_backend_tensor_get
#undef ggml_backend_tensor_set
using namespace strata::core::portable_detail;
template<class F> static void reject_before(F f){auto before=submissions;bool caught=false;try{f();}catch(const std::invalid_argument&){caught=true;}native_hc_need(caught&&submissions==before,"expected rejection before dispatch");}
int main(){try {
 ggml_backend_vk_context vk; ggml_backend backend{}; backend.context=&vk; ggml_backend_vk_buffer_context bc;bc.device=vk.device;ggml_backend_buffer buffer{};buffer.context=&bc;buffer.size=1000000;
 std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({32768,nullptr,true}),ggml_free);
 auto tensor=[&](int64_t n,uintptr_t a){auto*t=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,n);t->buffer=&buffer;t->data=(void*)a;return t;};
 auto*r=tensor(20480,0x100004);auto*b=tensor(5120,0x200008);auto*i=tensor(8,0x30000c);auto*c=tensor(20480,0x400010);auto*g=tensor(8,0x500014);auto*e=tensor(8,0x600018);auto*s=tensor(1,0x70001c);status_tensor=s;
 std::unique_ptr<StrataVkNativeHcPending,decltype(&strata_vk_native_hc_pending_destroy)> p(strata_vk_native_hc_pending_create(&backend,"mock",512),strata_vk_native_hc_pending_destroy);p->owner->status=s;
 auto run=[&]{strata_vk_native_hc_pending_run(p.get(),r,b,i,c,g,e);};
 auto verify=[&](std::array<uint32_t,3> grid){native_hc_need(observed_grid==grid && observed_offsets==std::array<uint32_t,7>{1,2,3,4,5,6,7} && observed_buffers==std::vector<const ggml_tensor*>{r,b,i,c,g,e,s},"dispatch grid, offsets or descriptor order");};
 run();native_hc_need(submissions==1,"T2 dispatch count");verify({4,2,1});
 // Each full-bank span must be validated, including all outputs and row 1.
 for(auto*t:{b,i,c,g,e}){auto n=t->ne[0];t->ne[0]=n/2;reject_before(run);t->ne[0]=n;}
 auto old=c->data;c->data=(void*)(uintptr_t(r->data)+40960);reject_before(run);c->data=old;
 vk.device->properties.limits.maxComputeWorkGroupCount[1]=1;reject_before(run);vk.device->properties.limits.maxComputeWorkGroupCount[1]=65535;
 fake_status=2;auto before=submissions;bool refused=false;try{run();}catch(const std::invalid_argument&){refused=true;}native_hc_need(refused&&submissions==before+1,"aggregate status accepted");fake_status=0;
 for(auto*t:{r,b,i,c,g,e})t->ne[0]/=2;
 run();verify({4,1,1});native_hc_need(submissions==3,"tail dispatch count");
 std::puts("actual HC pending wrapper dispatch/validation passed (mock Vulkan; no shader execution)");return 0;
 }catch(const std::exception& ex){std::fprintf(stderr,"%s\n",ex.what());return 1;}}
