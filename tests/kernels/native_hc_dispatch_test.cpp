#include "ggml.h"
#include "ggml-backend-impl.h"
#include <array>
#include <memory>
#include <vector>
#include <cstdio>
#include <cstring>
struct MockDevice{struct{struct{uint32_t maxComputeWorkGroupCount[3]={65535,65535,65535};uint32_t maxComputeSharedMemorySize=32768;}limits;}properties;};
struct ggml_backend_vk_context{std::shared_ptr<MockDevice> device=std::make_shared<MockDevice>();};
struct ggml_backend_vk_buffer_context{std::weak_ptr<MockDevice> device;};
namespace strata::core::portable_detail {
enum class NativeComponentAddressing {descriptor,physical64};
struct StrataVkLogitsGuard{ggml_backend_t backend;ggml_tensor* status=nullptr;size_t bytes=512;int pipeline=1;};
static bool is_vk=true;static size_t submissions=0;static uint32_t fake_status=0;static const ggml_tensor* status_tensor=nullptr;static std::vector<std::array<uint32_t,3>> grids;
static std::unique_ptr<StrataVkLogitsGuard> strata_vk_q8_component_owner(ggml_backend_t b,const uint32_t*,size_t,NativeComponentAddressing,const char*,uint32_t,uint32_t,const char*,size_t){return std::make_unique<StrataVkLogitsGuard>(StrataVkLogitsGuard{b});}
static bool ggml_backend_buffer_is_vk(ggml_backend_buffer_t){return is_vk;}
static uint32_t get_misalign_bytes(ggml_backend_vk_context*,const ggml_tensor* t){return uintptr_t(t->data)%256;}
static void ggml_backend_vk_synchronize(ggml_backend_t){}
static void ggml_pipeline_request_descriptor_sets(ggml_backend_vk_context*,int,int){}
static std::shared_ptr<int> ggml_vk_get_compute_ctx(ggml_backend_vk_context*){return std::make_shared<int>(1);}
static void ggml_vk_sync_buffers(ggml_backend_vk_context*,const std::shared_ptr<int>&){}
static int ggml_vk_tensor_subbuffer(ggml_backend_vk_context*,const ggml_tensor*,bool){return 0;}
static void ggml_vk_dispatch_pipeline(ggml_backend_vk_context*,const std::shared_ptr<int>&,int,std::initializer_list<int>,const std::array<uint32_t,9>& p,const std::array<uint32_t,3>& grid){++submissions;grids.push_back(grid);if(p!=std::array<uint32_t,9>{uint32_t(submissions%2==1?0:1),8,2,3,4,5,6,7,0})throw std::runtime_error("HC descriptor bias mapping");}
static void mock_get(const ggml_tensor* t,void* out,size_t,size_t size){std::memset(out,0,size);if(t==status_tensor)std::memcpy(out,&fake_status,4);}
static void mock_set(ggml_tensor*,const void*,size_t,size_t){}
}
#define ggml_backend_tensor_get mock_get
#define ggml_backend_tensor_set mock_set
#include "strata_vk_native_hc.inc"
#undef ggml_backend_tensor_get
#undef ggml_backend_tensor_set
using namespace strata::core::portable_detail;
template<class F>void reject(F f){auto before=submissions;bool caught=false;try{f();}catch(const std::invalid_argument&){caught=true;}native_hc_need(caught&&before==submissions,"native HC: refusal submitted work");}
int main(){try{ggml_backend_vk_context vk;ggml_backend backend{};backend.context=&vk;ggml_backend_vk_buffer_context bc;bc.device=vk.device;ggml_backend_buffer buffer{};buffer.context=&bc;
 std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({32768,nullptr,true}),ggml_free);auto tensor=[&](ggml_type type,int64_t n,uintptr_t address){auto* t=ggml_new_tensor_1d(ctx.get(),type,n);t->buffer=&buffer;t->data=(void*)address;return t;};
 auto* x=tensor(GGML_TYPE_F32,8*10240,0x100008);auto* w=tensor(GGML_TYPE_F32,10240,0x30000c);auto* wi=tensor(GGML_TYPE_I32,20480,0x500010);auto* xn=tensor(GGML_TYPE_F32,8*10240,0x600014);auto* rs=tensor(GGML_TYPE_F32,32,0x700018);auto* inj=tensor(GGML_TYPE_F32,32,0x80001c);auto* status=tensor(GGML_TYPE_I32,128,0x900000);status_tensor=status;
 StrataVkNativeHc hc;hc.owner=std::make_unique<StrataVkLogitsGuard>(StrataVkLogitsGuard{&backend,status});auto run=[&]{strata_vk_native_hc_run(&hc,x,w,wi,xn,rs,inj,8);};run();native_hc_need(submissions==2&&grids[0]==std::array<uint32_t,3>{1,8,1}&&grids[1]==std::array<uint32_t,3>{4,8,1},"HC mode grids");
 reject([&]{strata_vk_native_hc_run(nullptr,x,w,wi,xn,rs,inj,8);});reject([&]{strata_vk_native_hc_run(&hc,x,w,wi,xn,rs,inj,9);});
 is_vk=false;reject(run);is_vk=true;auto stride=x->nb[0];x->nb[0]=8;reject(run);x->nb[0]=stride;
 auto old=wi->type;wi->type=GGML_TYPE_F32;reject(run);wi->type=old;auto* address=xn->data;xn->data=x->data;reject(run);xn->data=address;
 address=x->data;x->data=(void*)0x100009;reject(run);x->data=address;auto ne=rs->ne[0];rs->ne[0]=31;reject(run);rs->ne[0]=ne;
 ggml_backend_vk_buffer_context other;other.device=std::make_shared<MockDevice>();ggml_backend_buffer foreign{};foreign.context=&other;w->buffer=&foreign;reject(run);w->buffer=&buffer;
 vk.device->properties.limits.maxComputeSharedMemorySize=4096;reject(run);vk.device->properties.limits.maxComputeSharedMemorySize=32768;
 auto rounding=std::fegetround();std::fesetround(FE_DOWNWARD);reject(run);std::fesetround(rounding);
 fake_status=2;bool caught=false;try{run();}catch(const std::invalid_argument&){caught=true;}native_hc_need(caught&&submissions==4,"native HC: bad device arithmetic accepted");puts("native HC actual dispatch bias/grid/domain/status refusal (mock submission) passed");return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
