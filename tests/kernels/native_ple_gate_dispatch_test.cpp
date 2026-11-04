#include "ggml.h"
#include "ggml-backend-impl.h"
#include <array>
#include <cfenv>
#include <memory>
#include <stdexcept>
#include <vector>
#include <cstdio>
#include <cstring>
struct MockDevice{struct{struct{uint32_t maxComputeWorkGroupCount[3]={65535,65535,65535};uint32_t maxComputeWorkGroupSize[3]={1024,1024,64};uint32_t maxComputeWorkGroupInvocations=1024,maxComputeSharedMemorySize=32768,maxPerStageDescriptorStorageBuffers=16,maxDescriptorSetStorageBuffers=16,maxStorageBufferRange=131072;}limits;}properties;};
struct ggml_backend_vk_context{std::shared_ptr<MockDevice> device=std::make_shared<MockDevice>();};
struct ggml_backend_vk_buffer_context{std::weak_ptr<MockDevice> device;};
namespace strata::core::portable_detail {
enum class NativeComponentAddressing {descriptor,physical64};
struct StrataVkLogitsGuard{ggml_backend_t backend;ggml_tensor* status=nullptr;size_t bytes=512;int pipeline=1;};
static bool is_vk=true,bad_input=false,input_alias=false;static size_t submissions=0;static uint32_t fake_status=0;static const ggml_tensor* status_tensor=nullptr;
static bool ggml_backend_is_vk(ggml_backend_t){return is_vk;}
static bool ggml_backend_buffer_is_vk(ggml_backend_buffer_t){return is_vk;}
static std::unique_ptr<StrataVkLogitsGuard> strata_vk_q8_component_owner(ggml_backend_t b,const uint32_t*,size_t,NativeComponentAddressing,const char*,uint32_t n,uint32_t words,const char*,size_t,size_t){if(n!=5||words!=5)throw std::runtime_error("gate pipeline layout");return std::make_unique<StrataVkLogitsGuard>(StrataVkLogitsGuard{b});}
static uint32_t get_misalign_bytes(ggml_backend_vk_context*,const ggml_tensor* t){return uintptr_t(t->data)%256;}
static void ggml_backend_vk_synchronize(ggml_backend_t){}
static void ggml_pipeline_request_descriptor_sets(ggml_backend_vk_context*,int,int){}
static std::shared_ptr<int> ggml_vk_get_compute_ctx(ggml_backend_vk_context*){return std::make_shared<int>(1);}
static void ggml_vk_sync_buffers(ggml_backend_vk_context*,const std::shared_ptr<int>&){}
static int ggml_vk_tensor_subbuffer(ggml_backend_vk_context*,const ggml_tensor*,bool){return 0;}
static void ggml_vk_dispatch_pipeline(ggml_backend_vk_context*,const std::shared_ptr<int>&,int,std::initializer_list<int>,const std::array<uint32_t,5>& off,const std::array<uint32_t,3>& grid){if(off!=std::array<uint32_t,5>{2,input_alias?2u:3u,4,5,6}||grid!=std::array<uint32_t,3>{4,1,1})throw std::runtime_error("gate bias/grid");++submissions;}
static void mock_get(const ggml_tensor* t,void* out,size_t,size_t n){std::memset(out,0,n);if(t==status_tensor)std::memcpy(out,&fake_status,4);else if(bad_input){uint32_t v=0x7f800000;std::memcpy(out,&v,4);}}
static void mock_set(ggml_tensor*,const void*,size_t,size_t){}
}
#include "native_hc_layout.hpp"
#define ggml_backend_tensor_get mock_get
#define ggml_backend_tensor_set mock_set
#include "strata_vk_native_ple_gate.inc"
#undef ggml_backend_tensor_get
#undef ggml_backend_tensor_set
using namespace strata::core::portable_detail;
template<class F>void reject(F f){auto before=submissions;bool caught=false;try{f();}catch(const std::invalid_argument&){caught=true;}native_hc_need(caught&&before==submissions,"gate refusal submitted work");}
int main(){try{
 ggml_backend_vk_context vk;ggml_backend backend{};backend.context=&vk;ggml_backend_vk_buffer_context bc;bc.device=vk.device;ggml_backend_buffer buffer{};buffer.context=&bc;buffer.size=200000;
 std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({32768,nullptr,true}),ggml_free);
 auto tensor=[&](ggml_type type,int64_t n,uintptr_t addr){auto*t=ggml_new_tensor_1d(ctx.get(),type,n);t->buffer=&buffer;t->data=(void*)addr;return t;};
 auto*k=tensor(GGML_TYPE_F32,10240,0x100008);auto*q=tensor(GGML_TYPE_F32,10240,0x20000c);auto*g=tensor(GGML_TYPE_F32,4,0x300010);auto*d=tensor(GGML_TYPE_F32,16,0x400014);auto*s=tensor(GGML_TYPE_I32,1,0x500018);status_tensor=s;
 std::unique_ptr<StrataVkNativePleGate,decltype(&strata_vk_native_ple_gate_destroy)> p(strata_vk_native_ple_gate_create(&backend,"mock.spv",300000),strata_vk_native_ple_gate_destroy);p->owner->status=s;
 auto run=[&]{return strata_vk_native_ple_gate_run(p.get(),k,q,g,d);};native_hc_need(run()==0&&submissions==1&&strata_vk_native_ple_gate_calls(p.get())==1,"gate first dispatch");
 auto same=q->data;q->data=k->data;input_alias=true;native_hc_need(run()==0,"read-only input alias refused");input_alias=false;q->data=same;
 is_vk=false;reject(run);is_vk=true;auto count=k->ne[0];k->ne[0]=10239;reject(run);k->ne[0]=count;
 auto stride=k->nb[0];k->nb[0]=8;reject(run);k->nb[0]=stride;
 auto old=d->data;d->data=k->data;reject(run);d->data=old;auto type=q->type;q->type=GGML_TYPE_I32;reject(run);q->type=type;
 old=q->data;q->data=(void*)0x20000d;reject(run);q->data=old;bad_input=true;reject(run);bad_input=false;
 old=s->data;s->data=(void*)0x500019;reject(run);s->data=old;
 vk.device->properties.limits.maxStorageBufferRange=40960;reject(run);vk.device->properties.limits.maxStorageBufferRange=131072;
 ggml_backend_vk_buffer_context other;other.device=std::make_shared<MockDevice>();ggml_backend_buffer foreign{};foreign.context=&other;foreign.size=100;auto* saved=q->buffer;q->buffer=&foreign;reject(run);q->buffer=saved;
 p->budget=200511;reject(run);p->budget=200512;native_hc_need(run()==0,"exact budget refused");p->budget=300000;fake_status=1;native_hc_need(run()==1&&submissions==4&&strata_vk_native_ple_gate_calls(p.get())==3,"gate device refusal accepted");
 fake_status=0;native_hc_need(run()==0&&strata_vk_native_ple_gate_calls(p.get())==4,"gate retry differs");
 vk.device->properties.limits.maxComputeWorkGroupInvocations=256;reject([&]{strata_vk_native_ple_gate_create(&backend,"mock.spv",300000);});
 std::puts("PLE gate actual dispatch mapping, domain, device, budget, status and retry passed with mock backend");return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
