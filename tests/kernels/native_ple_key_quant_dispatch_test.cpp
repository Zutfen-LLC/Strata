// Compile and exercise the production bounded quant dispatch with a CPU Vulkan shim.
#include "ggml.h"
#include "ggml-backend-impl.h"
#include "native_q8_abi.h"
#include <array>
#include <memory>
#include <stdexcept>
#include <cstdio>
struct MockDevice { struct { struct { uint32_t maxComputeWorkGroupCount[3]={65535,65535,65535}; uint32_t maxStorageBufferRange=1u<<30; } limits; } properties; };
struct ggml_backend_vk_context { std::shared_ptr<MockDevice> device=std::make_shared<MockDevice>(); };
struct ggml_backend_vk_buffer_context { std::weak_ptr<MockDevice> device; };
namespace strata::core::portable_detail {
struct StrataVkLogitsGuard { ggml_backend_t backend; ggml_tensor* status; int pipeline=1; };
struct StrataVkNativeStages { std::unique_ptr<StrataVkLogitsGuard> owner; bool quant_rcp=true; uint64_t quant_submitted=0,quant_completed=0,quant_submission_uncertain=0; };
static size_t submissions=0;
static std::array<uint32_t,9> params_seen{};
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
static void ggml_vk_dispatch_pipeline(ggml_backend_vk_context*,const std::shared_ptr<int>&,int,std::initializer_list<int>,const std::array<uint32_t,9>& p,const std::array<uint32_t,3>& g) { ++submissions; params_seen=p; grid_seen=g; }
#include "strata_vk_ple_key_quant.inc"
}
using namespace strata::core::portable_detail;
static void need(bool b) { if(!b) throw std::runtime_error("quant dispatch contract"); }
template<class F> static void rejects(F f) { auto before=submissions; bool caught=false; try { f(); } catch(const std::exception&) { caught=true; } need(caught && before==submissions); }
int main() { try {
 ggml_backend_vk_context vk; ggml_backend backend{}; backend.context=&vk;
 ggml_backend_vk_buffer_context bc; bc.device=vk.device; ggml_backend_buffer buffer{}; buffer.context=&bc; buffer.size=1u<<20;
 std::unique_ptr<ggml_context,decltype(&ggml_free)> c(ggml_init({16384,nullptr,true}),ggml_free);
 auto tensor=[&](ggml_type type,size_t n,uintptr_t addr) { auto* t=ggml_new_tensor_1d(c.get(),type,n); t->buffer=&buffer; t->data=reinterpret_cast<void*>(addr); return t; };
 auto* x=tensor(GGML_TYPE_F32,5120,0x2010); auto* y=tensor(GGML_TYPE_I32,1440,0x10018); auto* status=tensor(GGML_TYPE_I32,128,0x18020);
 StrataVkNativeStages s; s.owner=std::make_unique<StrataVkLogitsGuard>(StrataVkLogitsGuard{&backend,status});
 auto run=[&](size_t t){strata_vk_ple_key_quantize_bounded(&s,x,y,t);};
 run(2); need(grid_seen==std::array<uint32_t,3>{80,2,1}); need(params_seen==std::array<uint32_t,9>{0,2560,2560,4,4,8,89,4,6});
 rejects([&]{run(0);}); rejects([&]{run(3);}); rejects([&]{run(1);});
 auto* old=y->data; y->data=x->data; rejects([&]{run(2);}); y->data=reinterpret_cast<void*>(0x100ffc); rejects([&]{run(2);}); y->data=old;
 vk.device->properties.limits.maxComputeWorkGroupCount[1]=1; rejects([&]{run(2);}); vk.device->properties.limits.maxComputeWorkGroupCount[1]=65535;
 ggml_backend_vk_buffer_context foreign; foreign.device=std::make_shared<MockDevice>(); ggml_backend_buffer fb{}; fb.context=&foreign; fb.size=buffer.size; y->buffer=&fb; rejects([&]{run(2);}); y->buffer=&buffer;
 auto gdn=[&](size_t t){strata_vk_gdn_input_quantize_bounded(&s,x,y,t);};
 gdn(2); need(grid_seen==std::array<uint32_t,3>{80,2,1});
 rejects([&]{gdn(0);}); rejects([&]{gdn(3);});
 old=y->data; y->data=x->data; rejects([&]{gdn(2);}); y->data=old;
 x->ne[0]/=2; y->ne[0]/=2; run(1); need(grid_seen==std::array<uint32_t,3>{80,1,1} && s.quant_submitted==3 && s.quant_completed==3);
 gdn(1); need(grid_seen==std::array<uint32_t,3>{80,1,1} && s.quant_completed==4);
 auto* ox=tensor(GGML_TYPE_F32,2*6144,0x2010); auto* oq=tensor(GGML_TYPE_I32,2*6912/4,0x10018);
 auto out=[&](size_t n){strata_vk_gdn_output_quantize_bounded(&s,ox,oq,n);};out(2);
 need(grid_seen==std::array<uint32_t,3>{192,2,1} && params_seen[1]==6144 && params_seen[2]==6144);
 rejects([&]{out(1);});rejects([&]{out(3);});auto saved=oq->data;oq->data=ox->data;rejects([&]{out(2);});oq->data=saved;
 ox->ne[0]/=2;oq->ne[0]/=2;out(1);need(grid_seen==std::array<uint32_t,3>{192,1,1});
 x->ne[0]*=2;y->ne[0]*=2;strata_vk_qsa_input_quantize_bounded(&s,x,y,2);need(grid_seen==std::array<uint32_t,3>{80,2,1});
 rejects([&]{strata_vk_qsa_input_quantize_bounded(&s,x,y,1);});
 ox->ne[0]*=2;oq->ne[0]*=2;strata_vk_qsa_wo_quantize_bounded(&s,ox,oq,2);need(grid_seen==std::array<uint32_t,3>{192,2,1});
 rejects([&]{strata_vk_qsa_wo_quantize_bounded(&s,ox,oq,3);});
 puts("PLE, GDN and QSA quant wrapper mock dispatch passed"); return 0;
} catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; } }
