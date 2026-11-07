// Actual wrapper preflight and dispatch, with Vulkan calls mocked; no GPU arithmetic.
#include "ggml.h"
#include "ggml-backend-impl.h"
#include <array>
#include <memory>
#include <stdexcept>
#include <string>
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
template<class F> static void rejects_reason(F f,const char* reason) {auto before=submissions;bool caught=false;try{f();}catch(const std::exception&e){caught=std::string(e.what())==reason;}need(caught&&before==submissions);}
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
    // GDN is a distinct component/domain, not a PLE flag. Both QKV and z
    // consume the same packed rows and each uses one T-row dispatch.
    s.vocab=false; s.gdn=true;
    auto gdn=[&](size_t n,size_t t){strata_vk_gdn_input_run_bounded(&s,w,x,y,n,t);};
    auto before=submissions; gdn(10240,2);
    need(submissions==before+1 && grid_seen==std::array<uint32_t,3>{10240,2,1});
    rejects([&]{strata_vk_ple_key_run_bounded(&s,w,x,y,rows,2);});
    rejects([&]{gdn(10240,0);}); rejects([&]{gdn(10240,3);});
    rejects([&]{gdn(6144,2);}); rejects([&]{gdn(10240,1);});
    rejects([&]{strata_vk_gdn_input_run_bounded(&s,nullptr,x,y,rows,2);});
    old=y->data; y->data=x->data; rejects([&]{gdn(10240,2);});
    y->data=reinterpret_cast<void*>(0x5000000); rejects([&]{gdn(10240,2);}); y->data=old;
    w->buffer=&fb; rejects([&]{gdn(10240,2);}); w->buffer=&buffer;
    w->ne[0]=6144*2720/4; y->ne[0]=2*6144; gdn(6144,2);
    need(grid_seen==std::array<uint32_t,3>{6144,2,1} && params_seen[2]==4 && params_seen[3]==6);
    x->ne[0]/=2; y->ne[0]/=2; gdn(6144,1);
    need(grid_seen==std::array<uint32_t,3>{6144,1,1});
    // GDN output uses its own 6144-input shader and domain, not QSA/PLE.
    s.gdn=false; s.gdn_output=true;
    w->ne[0]=2560*6528/4; x->ne[0]=2*6912/4; y->ne[0]=2*2560;
    auto gout=[&](size_t n){strata_vk_gdn_output_run_bounded(&s,w,x,y,n);};gout(2);
    need(grid_seen==std::array<uint32_t,3>{2560,2,1}&&params_seen[0]==2560&&params_seen[2]==4&&params_seen[3]==6);
    rejects([&]{gout(0);});rejects([&]{gout(1);});rejects([&]{gout(3);});
    old=y->data;y->data=x->data;rejects([&]{gout(2);});y->data=old;
    x->ne[0]/=2;y->ne[0]/=2;gout(1);need(grid_seen==std::array<uint32_t,3>{2560,1,1});
    // QSA has explicit bounded modes and cannot cross into GDN or serial.
    s.gdn_output=false;s.qsa_rows=true;
    x->data=reinterpret_cast<void*>(0x2400010);y->data=reinterpret_cast<void*>(0x2500018);
    w->ne[0]=12288*2720/4;x->ne[0]=2*2880/4;y->ne[0]=2*12288;
    auto qkv=[&](size_t n,size_t t){strata_vk_qsa_qkv_run_bounded(&s,w,x,y,n,t);};
    qkv(12288,2);need(grid_seen==std::array<uint32_t,3>{12288,2,1}&&params_seen[2]==4&&params_seen[3]==6);
    rejects([&]{strata_vk_gdn_input_run_bounded(&s,w,x,y,12288,2);});rejects([&]{strata_vk_q8_wo_run(&s,w,x,y,12288);});
    rejects([&]{qkv(12288,0);});rejects([&]{qkv(12288,3);});rejects([&]{qkv(12288,1);});
    old=y->data;y->data=x->data;rejects([&]{qkv(12288,2);});y->data=reinterpret_cast<void*>(0x5000000);rejects([&]{qkv(12288,2);});y->data=old;
    w->buffer=&fb;rejects([&]{qkv(12288,2);});w->buffer=&buffer;--s.budget;rejects([&]{qkv(12288,2);});++s.budget;
    vk.device->properties.limits.maxComputeWorkGroupCount[1]=1;rejects([&]{qkv(12288,2);});vk.device->properties.limits.maxComputeWorkGroupCount[1]=65535;
    w->ne[0]=512*2720/4;y->ne[0]=2*512;qkv(512,2);x->ne[0]/=2;y->ne[0]/=2;qkv(512,1);
    rejects_reason([&]{strata_vk_q8_wo_run(&s,w,x,y,512);},"Q8 WO: bounded component requires bounded entrypoint");
    s.qsa_rows=false;s.qkv=true;strata_vk_q8_wo_run(&s,w,x,y,512);s.qkv=false;s.qsa_rows=true;
    s.qsa_rows=false;s.qsa_wo_rows=true;w->ne[0]=2560*6528/4;x->ne[0]=2*6912/4;y->ne[0]=2*2560;
    auto qwo=[&](size_t t){strata_vk_qsa_wo_run_bounded(&s,w,x,y,t);};qwo(2);need(grid_seen==std::array<uint32_t,3>{2560,2,1});
    rejects([&]{strata_vk_gdn_output_run_bounded(&s,w,x,y,2);});rejects([&]{strata_vk_q8_wo_run(&s,w,x,y,2560);});rejects([&]{qwo(1);});
    x->ne[0]/=2;y->ne[0]/=2;qwo(1);
    rejects_reason([&]{strata_vk_q8_wo_run(&s,w,x,y,2560);},"Q8 WO: bounded component requires bounded entrypoint");
    s.qsa_wo_rows=false;strata_vk_q8_wo_run(&s,w,x,y,2560);
    s.gdn_output=true;gout(1);rejects_reason([&]{qwo(1);},"native QSA WO rows: unsupported Q8 span");
    s.gdn_output=false;s.qsa_wo_rows=true;qwo(1);rejects_reason([&]{gout(1);},"native GDN output rows: unsupported Q8 span");
    puts("PLE, GDN and QSA bounded wrapper mock dispatch passed"); return 0;
} catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; } }
