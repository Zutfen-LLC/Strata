// Actual GDN row convolution wrapper with mocked Vulkan submission. No GPU arithmetic.
#include "ggml.h"
#include "ggml-backend-impl.h"
#include <array>
#include <memory>
#include <vector>
#include <cstdio>
#include <cstring>
#include <stdexcept>
struct MockDevice { struct { struct { uint32_t maxComputeWorkGroupCount[3]={65535,65535,65535},maxComputeWorkGroupSize[3]={1024,1024,64}; uint32_t maxComputeSharedMemorySize=32768,maxComputeWorkGroupInvocations=1024,maxPerStageDescriptorStorageBuffers=16,maxDescriptorSetStorageBuffers=16,maxStorageBufferRange=1u<<24; } limits; } properties; };
struct ggml_backend_vk_context { std::shared_ptr<MockDevice> device=std::make_shared<MockDevice>(); };
struct ggml_backend_vk_buffer_context { std::weak_ptr<MockDevice> device; };
namespace strata::core::portable_detail {
enum class NativeComponentAddressing { descriptor,physical64 };
struct StrataVkLogitsGuard { ggml_backend_t backend; ggml_tensor* status=nullptr; size_t bytes=512; int pipeline=1; };
static size_t submissions=0; static std::array<uint32_t,15> offsets; static std::array<uint32_t,3> grid; static std::array<const ggml_tensor*,11> descriptors; static size_t di=0; static uint32_t fake_status=0; static const ggml_tensor* status_tensor=nullptr;
static bool ggml_backend_is_vk(ggml_backend_t){return true;}
static bool ggml_backend_buffer_is_vk(ggml_backend_buffer_t){return true;}
static std::unique_ptr<StrataVkLogitsGuard> strata_vk_q8_component_owner(ggml_backend_t b,const uint32_t*,size_t,NativeComponentAddressing,const char*,uint32_t n,uint32_t words,const char*,size_t,size_t){if(n!=11||words!=12)throw std::runtime_error("pipeline layout");return std::make_unique<StrataVkLogitsGuard>(StrataVkLogitsGuard{b});}
static uint32_t get_misalign_bytes(ggml_backend_vk_context*,const ggml_tensor*t){return uintptr_t(t->data)%256;}
static void ggml_backend_vk_synchronize(ggml_backend_t){}
static void ggml_pipeline_request_descriptor_sets(ggml_backend_vk_context*,int,int){di=0;}
static std::shared_ptr<int> ggml_vk_get_compute_ctx(ggml_backend_vk_context*){return std::make_shared<int>(1);}
static void ggml_vk_sync_buffers(ggml_backend_vk_context*,const std::shared_ptr<int>&){}
static int ggml_vk_tensor_subbuffer(ggml_backend_vk_context*,const ggml_tensor*t,bool){descriptors.at(di)=t;return int(di++);}
static void ggml_vk_dispatch_pipeline(ggml_backend_vk_context*,const std::shared_ptr<int>&,int,std::initializer_list<int> d,const std::array<uint32_t,15>&p,const std::array<uint32_t,3>&g){if(d.size()!=11||di!=11)throw std::runtime_error("descriptors");++submissions;offsets=p;grid=g;}
static const ggml_tensor* bad_input=nullptr;
static void mock_get(const ggml_tensor*t,void*out,size_t,size_t n){std::memset(out,0,n);if(t==status_tensor)std::memcpy(out,&fake_status,4);if(t==bad_input&&n>=192*2){uint32_t bad=0x3f800000u;std::memcpy(static_cast<char*>(out)+192,&bad,4);}}
static void mock_set(ggml_tensor*t,const void* p,size_t,size_t n){if(t!=status_tensor||n!=4||*static_cast<const uint32_t*>(p)!=0)throw std::runtime_error("status reset");}
static void* mock_base(ggml_backend_buffer_t){return reinterpret_cast<void*>(0x100000);}
}
#include "native_gdn_domain.hpp"
#define ggml_backend_tensor_get mock_get
#define ggml_backend_tensor_set mock_set
#define ggml_backend_buffer_get_base mock_base
namespace strata::core::portable_detail {
struct StrataVkNativeGdnRows { std::unique_ptr<StrataVkLogitsGuard> owner;size_t budget=0; };
#include "strata_vk_native_gdn_rows_run.inc"
}
#undef ggml_backend_tensor_get
#undef ggml_backend_tensor_set
#undef ggml_backend_buffer_get_base
using namespace strata::core::portable_detail;
static void need(bool b){if(!b)throw std::runtime_error("conv mock contract");}
template<class F> static void reject(F f){auto before=submissions;bool caught=false;try{f();}catch(const std::invalid_argument&){caught=true;}need(caught&&before==submissions);}
int main(){try{
 ggml_backend_vk_context vk;ggml_backend backend{};backend.context=&vk;ggml_backend_vk_buffer_context bc;bc.device=vk.device;ggml_backend_buffer buffer{};buffer.context=&bc;buffer.size=16u<<20;
 std::unique_ptr<ggml_context,decltype(&ggml_free)> c(ggml_init({32768,nullptr,true}),ggml_free);
 std::array<ggml_tensor*,13> t{};const size_t counts[]={786432,20480,96,96,12288,128,786432,12288,96,12288,98,12288,12288};
 buffer.size=64u<<20;
 auto tensor=[&](ggml_type type,size_t n,uintptr_t address){auto*x=ggml_new_tensor_1d(c.get(),type,n);x->buffer=&buffer;x->data=reinterpret_cast<void*>(address);return x;};
 for(size_t i=0;i<13;++i)t[i]=tensor(GGML_TYPE_F32,counts[i],0x100000+i*0x400000+4*(i+1));
 auto*aux=tensor(GGML_TYPE_F32,96+98+12288+12288,0x3900008);uintptr_t at=uintptr_t(aux->data);
 for(size_t i:{size_t(8),size_t(10),size_t(11),size_t(12)}){t[i]->data=reinterpret_cast<void*>(at);at+=counts[i]*4;}
 StrataVkNativeGdnRows owner;owner.budget=buffer.size+512;owner.owner=std::make_unique<StrataVkLogitsGuard>(StrataVkLogitsGuard{&backend});auto*p=&owner;
 auto*status=tensor(GGML_TYPE_I32,1,0x3f0002c);status_tensor=status;p->owner->status=status;
 uint32_t rows=2;auto run=[&]{strata_vk_native_gdn_rows_run(p,t.data(),aux,rows);};run();need(grid==std::array<uint32_t,3>{48,1,1}&&offsets[14]==2&&descriptors[8]==aux&&descriptors[10]==status);
 need(offsets[10]==offsets[8]+96&&offsets[11]==offsets[10]+98&&offsets[12]==offsets[11]+12288);
 bad_input=t[2];reject(run);bad_input=nullptr;
 {auto old=t[6]->data;t[6]->data=t[0]->data;reject(run);t[6]->data=old;}
 {auto old=t[1]->data;t[1]->data=nullptr;reject(run);t[1]->data=old;}
 {auto old=t[1]->ne[0];--t[1]->ne[0];reject(run);t[1]->ne[0]=old;}
 {auto old=t[10]->data;t[10]->data=reinterpret_cast<void*>(at);reject(run);t[10]->data=old;}
 {auto old=t[0]->data;t[0]->data=reinterpret_cast<void*>(0x100000+buffer.size-4);reject(run);t[0]->data=old;}
 ggml_backend_vk_buffer_context foreign;auto dev=std::make_shared<MockDevice>();foreign.device=dev;ggml_backend_buffer fb{};fb.context=&foreign;fb.size=buffer.size;auto saved=t[1]->buffer;t[1]->buffer=&fb;reject(run);t[1]->buffer=saved;
 {auto before=submissions;fake_status=1;bool caught=false;try{run();}catch(const std::invalid_argument&){caught=true;}need(caught&&submissions==before+1);fake_status=0;}
 rows=0;reject(run);rows=3;reject(run);rows=1;
 for(size_t i:{size_t(1),size_t(2),size_t(3),size_t(4),size_t(7),size_t(8),size_t(9),size_t(10),size_t(11),size_t(12)})t[i]->ne[0]/=2;
 aux->ne[0]/=2;at=uintptr_t(aux->data);for(size_t i:{size_t(8),size_t(10),size_t(11),size_t(12)}){t[i]->data=reinterpret_cast<void*>(at);at+=ggml_nbytes(t[i]);}
 run();need(grid==std::array<uint32_t,3>{48,1,1}&&offsets[14]==1);
 puts("actual GDN recurrence rows wrapper mock T2/T1 preflight and single dispatch passed; no GPU execution");return 0;
}catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
