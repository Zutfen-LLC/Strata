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
static size_t submissions=0; static std::array<uint32_t,9> offsets; static std::array<uint32_t,3> grid; static std::array<const ggml_tensor*,9> descriptors; static size_t di=0; static uint32_t fake_status=0; static const ggml_tensor* status_tensor=nullptr;
static bool ggml_backend_is_vk(ggml_backend_t){return true;}
static bool ggml_backend_buffer_is_vk(ggml_backend_buffer_t){return true;}
static std::unique_ptr<StrataVkLogitsGuard> strata_vk_q8_component_owner(ggml_backend_t b,const uint32_t*,size_t,NativeComponentAddressing,const char*,uint32_t n,uint32_t words,const char*,size_t,size_t){if(n!=11||words!=12)throw std::runtime_error("pipeline layout");return std::make_unique<StrataVkLogitsGuard>(StrataVkLogitsGuard{b});}
static uint32_t get_misalign_bytes(ggml_backend_vk_context*,const ggml_tensor*t){return uintptr_t(t->data)%256;}
static void ggml_backend_vk_synchronize(ggml_backend_t){}
static void ggml_pipeline_request_descriptor_sets(ggml_backend_vk_context*,int,int){di=0;}
static std::shared_ptr<int> ggml_vk_get_compute_ctx(ggml_backend_vk_context*){return std::make_shared<int>(1);}
static void ggml_vk_sync_buffers(ggml_backend_vk_context*,const std::shared_ptr<int>&){}
static int ggml_vk_tensor_subbuffer(ggml_backend_vk_context*,const ggml_tensor*t,bool){descriptors.at(di)=t;return int(di++);}
static void ggml_vk_dispatch_pipeline(ggml_backend_vk_context*,const std::shared_ptr<int>&,int,std::initializer_list<int> d,const std::array<uint32_t,9>&p,const std::array<uint32_t,3>&g){if(d.size()!=9||di!=9)throw std::runtime_error("descriptors");++submissions;offsets=p;grid=g;}
static const ggml_tensor* bad_input=nullptr;
static void mock_get(const ggml_tensor*t,void*out,size_t,size_t n){std::memset(out,0,n);if(t==status_tensor)std::memcpy(out,&fake_status,4);if(t==bad_input&&n>=10244){uint32_t bad=0x7f800000u;std::memcpy(static_cast<char*>(out)+10240,&bad,4);}}
static void mock_set(ggml_tensor*t,const void* p,size_t,size_t n){if(t!=status_tensor||n!=4||*static_cast<const uint32_t*>(p)!=0)throw std::runtime_error("status reset");}
static void* mock_base(ggml_backend_buffer_t){return reinterpret_cast<void*>(0x100000);}
}
#include "native_hc_layout.hpp"
#define ggml_backend_tensor_get mock_get
#define ggml_backend_tensor_set mock_set
#define ggml_backend_buffer_get_base mock_base
namespace strata::core::portable_detail {
struct StrataVkNativeGdnAbRows { std::unique_ptr<StrataVkLogitsGuard> owner;size_t budget=0; };
#include "strata_vk_native_gdn_ab_rows_run.inc"
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
 std::array<ggml_tensor*,8> t{}; const size_t counts[]={5120,61440,61440,48,48,96,96,672};
 for(size_t i=0;i<8;++i){t[i]=ggml_new_tensor_1d(c.get(),(i==1||i==2)?GGML_TYPE_I32:GGML_TYPE_F32,counts[i]);t[i]->buffer=&buffer;t[i]->data=reinterpret_cast<void*>(0x100000+i*0x100000+4*(i+1));}
 StrataVkNativeGdnAbRows owner;owner.budget=buffer.size+512;owner.owner=std::make_unique<StrataVkLogitsGuard>(StrataVkLogitsGuard{&backend});auto* p=&owner;
 auto* status=ggml_new_tensor_1d(c.get(),GGML_TYPE_I32,1);status->buffer=&buffer;status->data=reinterpret_cast<void*>(0xb0002c);status_tensor=status;p->owner->status=status;
 uint32_t rows=2;auto run=[&]{strata_vk_native_gdn_ab_rows_run(p,t.data(),rows);};run();need(descriptors[8]==status&&offsets[8]==11);need(grid==std::array<uint32_t,3>{48,2,1});for(size_t i=0;i<8;++i){need(offsets[i]==i+1&&descriptors[i]==t[i]&&ggml_nbytes(descriptors[i])==counts[i]*4);}

 for(size_t i=0;i<8;++i)for(size_t j=0;j<i;++j){auto old=t[i]->data;t[i]->data=t[j]->data;reject(run);t[i]->data=old;}
 {auto* old=t[1];t[1]=nullptr;reject(run);t[1]=old;}
 {auto old=t[0]->data;t[0]->data=nullptr;reject(run);t[0]->data=old;}
 {auto old=status->data;status->data=t[1]->data;reject(run);status->data=old;}
 rows=0;reject(run);rows=3;reject(run);rows=2;
 {auto before=submissions;fake_status=0x80;need(strata_vk_native_gdn_ab_rows_run(p,t.data(),rows)==0x80&&submissions==before+1);fake_status=0;}
 bad_input=t[0];reject(run);bad_input=nullptr;
 {auto old=t[0]->data;t[0]->data=reinterpret_cast<void*>(0x0ffffc);reject(run);t[0]->data=reinterpret_cast<void*>(0x100000+buffer.size-4);reject(run);t[0]->data=old;}
 {auto old=buffer.size;buffer.size=4;reject(run);buffer.size=old;}
 auto n=t[1]->ne[0];--t[1]->ne[0];reject(run);t[1]->ne[0]=n;
 ggml_backend_vk_buffer_context other;auto dev=std::make_shared<MockDevice>();other.device=dev;ggml_backend_buffer foreign{};foreign.context=&other;foreign.size=buffer.size;auto saved=t[2]->buffer;t[2]->buffer=&foreign;reject(run);t[2]->buffer=saved;
 vk.device->properties.limits.maxComputeWorkGroupCount[1]=1;reject(run);vk.device->properties.limits.maxComputeWorkGroupCount[1]=65535;
 --p->budget;reject(run);++p->budget;
 for(size_t i:{size_t(0),size_t(5),size_t(6),size_t(7)})t[i]->ne[0]/=2;rows=1;run();need(grid==std::array<uint32_t,3>{48,1,1});
 puts("actual GDN AB rows wrapper mock preflight and T2/T1 dispatch passed; no GPU execution");return 0;
}catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
