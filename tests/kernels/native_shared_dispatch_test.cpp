// Execute the actual dispatch validation with mocked Vulkan submission only.
// This is a host contract test, never arithmetic/hardware qualification.
#include "ggml.h"
#include "ggml-backend-impl.h"
#include <array>
#include <memory>
#include <vector>
#include <stdexcept>
#include <cstdio>
struct MockDevice {
    struct {struct {uint32_t maxComputeWorkGroupCount[3]={65535,65535,65535};uint32_t maxComputeSharedMemorySize=32768;} limits;} properties;
};
struct ggml_backend_vk_context {std::shared_ptr<MockDevice> device=std::make_shared<MockDevice>();};
struct ggml_backend_vk_buffer_context {std::weak_ptr<MockDevice> device;};
namespace strata::core::portable_detail {
struct StrataVkLogitsGuard {ggml_backend_t backend;size_t bytes=512;int pipeline=1;};
enum class NativeComponentAddressing {descriptor};
static size_t submissions=0;
static bool fail_submission=false;
static std::array<uint32_t,9> last_params{};
static std::array<uint32_t,3> last_grid{};
static std::unique_ptr<StrataVkLogitsGuard> strata_vk_q8_component_owner(ggml_backend_t backend,const uint32_t*,size_t,NativeComponentAddressing,
    const char*,uint32_t,uint32_t,const char*,size_t) {return std::make_unique<StrataVkLogitsGuard>(StrataVkLogitsGuard{backend});}
static bool ggml_backend_buffer_is_vk(ggml_backend_buffer_t) {return true;}
static uint32_t get_misalign_bytes(ggml_backend_vk_context*,const ggml_tensor* t) {return uint32_t(reinterpret_cast<uintptr_t>(t->data)%256);}
static void ggml_backend_vk_synchronize(ggml_backend_t) {}
static void ggml_pipeline_request_descriptor_sets(ggml_backend_vk_context*,int,int) {}
static std::shared_ptr<int> ggml_vk_get_compute_ctx(ggml_backend_vk_context*) {return std::make_shared<int>(1);}
static void ggml_vk_sync_buffers(ggml_backend_vk_context*,const std::shared_ptr<int>&) {}
static int ggml_vk_tensor_subbuffer(ggml_backend_vk_context*,const ggml_tensor*,bool) {return 0;}
static void ggml_vk_dispatch_pipeline(ggml_backend_vk_context*,const std::shared_ptr<int>&,int,std::initializer_list<int>,
    const std::array<uint32_t,9>& params,const std::array<uint32_t,3>& grid) {++submissions;if(fail_submission)throw std::runtime_error("injected submission failure");last_params=params;last_grid=grid;}
}
#include "strata_vk_native_shared.inc"
using namespace strata::core::portable_detail;
void need(bool ok) {if(!ok)throw std::runtime_error("shared dispatch contract failed");}
template<class F> void rejects(F fn) {const auto before=submissions;bool caught=false;try{fn();}catch(const std::invalid_argument&){caught=true;}need(caught && before==submissions);}
int main() {try {
    ggml_backend_vk_context vkctx;ggml_backend backend{};backend.context=&vkctx;
    ggml_backend_vk_buffer_context bufferctx;bufferctx.device=vkctx.device;
    ggml_backend_buffer buffer{};buffer.context=&bufferctx;
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({32768,nullptr,true}),ggml_free);
    auto tensor=[&](ggml_type type,int64_t n,uintptr_t address){auto* t=ggml_new_tensor_1d(ctx.get(),type,n);t->buffer=&buffer;t->data=reinterpret_cast<void*>(address);return t;};
    auto* w=tensor(GGML_TYPE_I32,2560/32*640*34/4,0x100008);
    auto* x=tensor(GGML_TYPE_I32,8*2560/32*9,0x400010);
    auto* y=tensor(GGML_TYPE_F32,8*640,0x500018);
    auto* aux=tensor(GGML_TYPE_F32,1,0x600020);
    StrataVkNativeShared component;component.owner=std::make_unique<StrataVkLogitsGuard>(StrataVkLogitsGuard{&backend});
    auto run=[&]{strata_vk_native_shared_run(&component,0,w,x,y,aux,2560,640,8);};
    run();need(submissions==1 && last_grid==std::array<uint32_t,3>{640,8,1});
    need(last_params==std::array<uint32_t,9>{0,2560,640,8,2,4,6,8,256});
    rejects([&]{strata_vk_native_shared_run(&component,0,w,x,y,aux,2560,640,9);});
    rejects([&]{strata_vk_native_shared_run(&component,3,w,x,y,aux,2560,640,8);});
    rejects([&]{strata_vk_native_shared_run(nullptr,0,w,x,y,aux,2560,640,8);});
    const auto old_type=x->type;x->type=GGML_TYPE_F32;rejects(run);x->type=old_type;
    auto old_ne=y->ne[0];y->ne[0]--;rejects(run);y->ne[0]=old_ne;
    auto* old_data=y->data;y->data=x->data;rejects(run);y->data=old_data;
    old_data=w->data;w->data=reinterpret_cast<void*>(0x100009);rejects(run);w->data=old_data;
    ggml_backend_vk_buffer_context otherctx;otherctx.device=std::make_shared<MockDevice>();
    ggml_backend_buffer otherbuffer{};otherbuffer.context=&otherctx;w->buffer=&otherbuffer;rejects(run);w->buffer=&buffer;
    vkctx.device->properties.limits.maxComputeWorkGroupCount[0]=639;rejects(run);vkctx.device->properties.limits.maxComputeWorkGroupCount[0]=65535;
    auto* ws=tensor(GGML_TYPE_I32,2560/2,0x700008);
    auto* xf=tensor(GGML_TYPE_F32,8*2560,0x800010);
    auto* raw=tensor(GGML_TYPE_F32,8,0x900018);
    strata_vk_native_shared_run(&component,1,ws,xf,raw,aux,2560,1,8);
    need(last_grid==std::array<uint32_t,3>{1,8,1} && last_params[8]==256);
    auto* out=tensor(GGML_TYPE_F32,8*2560,0xa00008);
    auto* sig=tensor(GGML_TYPE_F32,8,0xb00010);
    strata_vk_native_shared_run(&component,2,raw,xf,out,sig,2560,1,8);
    need(last_grid==std::array<uint32_t,3>{10,8,1});
    rejects([&]{strata_vk_native_shared_run(&component,1,ws,xf,raw,aux,2560,2,8);});
    // The separate matrix pipeline must not widen the scalar API.
    StrataVkNativeShared matrix;matrix.owner=std::make_unique<StrataVkLogitsGuard>(StrataVkLogitsGuard{&backend});matrix.matrix=true;matrix.budget=8u<<20;buffer.size=6u<<20;
    auto* mw=tensor(GGML_TYPE_I32,2560*128/2,0x100000);
    auto* mx=tensor(GGML_TYPE_F32,2560,0x500000);
    auto* my=tensor(GGML_TYPE_F32,128,0x600000);
    auto* ma=tensor(GGML_TYPE_F32,1,0x700000);
    strata_vk_native_shared_run(&matrix,1,mw,mx,my,ma,2560,128,1);
    need(last_grid==std::array<uint32_t,3>{128,1,1} && last_params[8]==256);
    need(strata_vk_native_shared_submitted(&matrix)==1 && strata_vk_native_shared_completed(&matrix)==1 && strata_vk_native_shared_uncertain(&matrix)==0);
    auto* qw=tensor(GGML_TYPE_I32,2560*512/2,0x100000);
    auto* qy=tensor(GGML_TYPE_F32,512,0x600000);
    strata_vk_native_shared_run(&matrix,1,qw,mx,qy,ma,2560,512,1);
    need(last_grid==std::array<uint32_t,3>{512,1,1});
    fail_submission=true;bool failed=false;try{strata_vk_native_shared_run(&matrix,1,mw,mx,my,ma,2560,128,1);}catch(const std::runtime_error&){failed=true;}fail_submission=false;
    need(failed&&strata_vk_native_shared_submitted(&matrix)==2&&strata_vk_native_shared_completed(&matrix)==2&&strata_vk_native_shared_uncertain(&matrix)==1);
    rejects([&]{strata_vk_native_shared_run(&matrix,1,mw,mx,my,ma,2560,127,1);});
    rejects([&]{strata_vk_native_shared_run(&matrix,0,mw,mx,my,ma,2560,128,1);});
    auto* saved=my->data;my->data=mx->data;rejects([&]{strata_vk_native_shared_run(&matrix,1,mw,mx,my,ma,2560,128,1);});my->data=saved;
    matrix.budget=buffer.size+matrix.owner->bytes;strata_vk_native_shared_run(&matrix,1,mw,mx,my,ma,2560,128,1);
    --matrix.budget;rejects([&]{strata_vk_native_shared_run(&matrix,1,mw,mx,my,ma,2560,128,1);});
    puts("native shared actual dispatch validation passed (mock submission)");return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
