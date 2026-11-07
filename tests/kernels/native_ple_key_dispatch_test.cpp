// Actual wrapper preflight and dispatch, with Vulkan calls mocked; no GPU arithmetic.
#include "ggml.h"
#include "ggml-backend-impl.h"
#include <array>
#include <memory>
#include <stdexcept>
#include <cstdio>
struct MockDevice { struct { struct { uint32_t maxComputeWorkGroupCount[3]={65535,65535,65535}; uint32_t maxStorageBufferRange=1u<<30; } limits; } properties; };
struct ggml_backend_vk_context { std::shared_ptr<MockDevice> device=std::make_shared<MockDevice>(); };
struct ggml_backend_vk_buffer_context { std::weak_ptr<MockDevice> device; };
namespace strata::core::portable_detail {
struct StrataVkLogitsGuard { ggml_backend_t backend; size_t bytes=512; int pipeline=1; };
static size_t submissions=0;
static std::array<uint32_t,4> params_seen{};
static std::array<uint32_t,3> grid_seen{};
static void logits_guard_need(bool b,const char* msg) { if(!b) throw std::runtime_error(msg); }
static bool ggml_backend_buffer_is_vk(ggml_backend_buffer_t) { return true; }
static size_t ggml_vk_tensor_buffer_offset(ggml_backend_vk_context*,const ggml_tensor* t) { return uintptr_t(t->data)-0x1000; }
static uint32_t get_misalign_bytes(ggml_backend_vk_context*,const ggml_tensor* t) { return uintptr_t(t->data)%256; }
static void ggml_backend_vk_synchronize(ggml_backend_t) {}
static void ggml_pipeline_request_descriptor_sets(ggml_backend_vk_context*,int,int) {}
static std::shared_ptr<int> ggml_vk_get_compute_ctx(ggml_backend_vk_context*) { return std::make_shared<int>(1); }
static void ggml_vk_sync_buffers(ggml_backend_vk_context*,const std::shared_ptr<int>&) {}
static int ggml_vk_tensor_subbuffer(ggml_backend_vk_context*,const ggml_tensor*,bool) { return 0; }
static void ggml_vk_dispatch_pipeline(ggml_backend_vk_context*,const std::shared_ptr<int>&,int,std::initializer_list<int>,const std::array<uint32_t,4>& p,const std::array<uint32_t,3>& g) { ++submissions; params_seen=p; grid_seen=g; }
}
#include "strata_vk_q8_wo.inc"
using namespace strata::core::portable_detail;
static void need(bool b) { if(!b) throw std::runtime_error("PLE key dispatch contract"); }
template<class F> static void rejects(F f) { auto before=submissions; bool caught=false; try { f(); } catch(const std::exception&) { caught=true; } need(caught && before==submissions); }
int main() { try {
    ggml_backend_vk_context vk; ggml_backend backend{}; backend.context=&vk;
    ggml_backend_vk_buffer_context bc; bc.device=vk.device; ggml_backend_buffer buffer{}; buffer.context=&bc; buffer.size=64u<<20;
    std::unique_ptr<ggml_context,decltype(&ggml_free)> c(ggml_init({16384,nullptr,true}),ggml_free);
    auto tensor=[&](ggml_type type,size_t n,uintptr_t address) { auto* t=ggml_new_tensor_1d(c.get(),type,n); t->buffer=&buffer; t->data=reinterpret_cast<void*>(address); return t; };
    constexpr size_t rows=10240;
    auto* w=tensor(GGML_TYPE_I32,rows*2720/4,0x100008);
    auto* x=tensor(GGML_TYPE_I32,2*2880/4,0x2000010);
    auto* y=tensor(GGML_TYPE_F32,2*rows,0x2100018);
    StrataVkQ8Wo s; s.owner=std::make_unique<StrataVkLogitsGuard>(StrataVkLogitsGuard{&backend}); s.vocab=true; s.budget=buffer.size+512;
    auto run=[&](size_t t) { strata_vk_ple_key_run_bounded(&s,w,x,y,rows,t); };
    run(2); need(grid_seen==std::array<uint32_t,3>{rows,2,1} && params_seen==std::array<uint32_t,4>{rows,2,4,6});
    // Shader's word offsets for both rows, including descriptor biases.
    need(params_seen[2]+720==724 && params_seen[3]+rows==10246);
    rejects([&]{run(0);}); rejects([&]{run(3);}); rejects([&]{run(1);});
    auto* old=y->data; y->data=x->data; rejects([&]{run(2);}); y->data=reinterpret_cast<void*>(0x5000000); rejects([&]{run(2);}); y->data=old;
    vk.device->properties.limits.maxComputeWorkGroupCount[1]=1; rejects([&]{run(2);}); vk.device->properties.limits.maxComputeWorkGroupCount[1]=65535;
    ggml_backend_vk_buffer_context foreign; foreign.device=std::make_shared<MockDevice>(); ggml_backend_buffer fb{}; fb.context=&foreign; w->buffer=&fb; rejects([&]{run(2);}); w->buffer=&buffer;
    --s.budget; rejects([&]{run(2);}); ++s.budget;
    x->ne[0]/=2; y->ne[0]/=2; run(1); need(grid_seen==std::array<uint32_t,3>{rows,1,1});
    strata_vk_q8_wo_run(&s,w,x,y,rows); need(grid_seen==std::array<uint32_t,3>{rows,1,1});
    x->ne[0]*=2; y->ne[0]*=2; rejects([&]{strata_vk_q8_wo_run(&s,w,x,y,rows);});
    puts("PLE key wrapper mock dispatch passed"); return 0;
} catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; } }
