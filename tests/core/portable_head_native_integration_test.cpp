#include "strata/core/portable_model.hpp"
#include "strata_vk_native_hc.hpp"
#include "native_hc_layout.hpp"
#include "portable_head_execution.hpp"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <limits>
#include <vector>
using namespace strata::core;
using namespace strata::core::portable_detail;
namespace {ggml_tensor* final_bank=nullptr;ggml_tensor* norm_bank=nullptr;ggml_tensor* mixed_bank=nullptr;bool fail_norm=false,fail_mix=false,fail_graph=false;size_t graph_links=0;decltype(ggml_backend_i::graph_compute) original_compute=nullptr;
ggml_status inspect(ggml_backend_t b,ggml_cgraph* g){for(int i=0;i<ggml_graph_n_nodes(g);++i){auto* n=ggml_graph_node(g,i);if(n->op==GGML_OP_MUL_MAT&&std::string(ggml_get_name(n))=="portable_logits"){native_hc_need(n->src[1]==mixed_bank&&mixed_bank->buffer&&ggml_backend_buffer_get_type(mixed_bank->buffer)==ggml_backend_get_default_buffer_type(b),"head actual mixer/vocabulary bank or device differs");++graph_links;}}return fail_graph?GGML_STATUS_FAILED:original_compute(b,g);}}
namespace strata::core::portable_detail {
struct StrataVkNativeHc{};
StrataVkNativeHc* strata_vk_native_hc_create(ggml_backend_t,const char* p,size_t b){native_hc_need(b>=512,"mock norm budget");std::ofstream(p)<<"mock";return new StrataVkNativeHc;}
size_t strata_vk_native_hc_bytes(const StrataVkNativeHc*){return 512;}void strata_vk_native_hc_destroy(StrataVkNativeHc*p)noexcept{delete p;}
void strata_vk_native_hc_norm_only(StrataVkNativeHc*,const ggml_tensor* r,const ggml_tensor*,ggml_tensor* n,ggml_tensor* rs){native_hc_need(!fail_norm&&r==final_bank&&r->buffer,"head actual final producer differs");norm_bank=n;std::vector<float> x(10240,0.f),v(4,1.f);ggml_backend_tensor_set(n,x.data(),0,x.size()*4);ggml_backend_tensor_set(rs,v.data(),0,v.size()*4);}
struct StrataVkNativeHcMix{};
StrataVkNativeHcMix* strata_vk_native_hc_mix_create(ggml_backend_t,const char* p,size_t b){native_hc_need(b>=512,"mock mix budget");std::ofstream(p)<<"mock";return new StrataVkNativeHcMix;}
size_t strata_vk_native_hc_mix_bytes(const StrataVkNativeHcMix*){return 512;}void strata_vk_native_hc_mix_destroy(StrataVkNativeHcMix*p)noexcept{delete p;}
void strata_vk_native_hc_mix_run(StrataVkNativeHcMix*,const ggml_tensor* n,const ggml_tensor* down,const ggml_tensor* up,ggml_tensor* lp,ggml_tensor* lo,ggml_tensor* le,ggml_tensor* pre,ggml_tensor* gates,ggml_tensor* ue,ggml_tensor* mixed){native_hc_need(!fail_mix&&n==norm_bank,"head actual normalized producer differs");for(auto* w:{down,up}){std::vector<uint16_t> v(3276800);ggml_backend_tensor_get(w,v.data(),0,v.size()*2);for(auto x:v)native_hc_need(x==0x4000,"head native Q8 to BF16 role conversion differs");}for(auto*t:{lp,lo,le,pre,gates,ue}){std::vector<float> v(size_t(ggml_nelements(t)),0.f);ggml_backend_tensor_set(t,v.data(),0,v.size()*4);}std::vector<float> v(2560,1.f);ggml_backend_tensor_set(mixed,v.data(),0,v.size()*4);mixed_bank=mixed;}
}
int main(int argc,char**argv){try{
 native_hc_need(argc==2,"scratch root required");std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> b(ggml_backend_cpu_init(),ggml_backend_free);native_hc_need(bool(b),"mock backend");
 original_compute=b->iface.graph_compute;b->iface.graph_compute=inspect;
 std::unique_ptr<ggml_context,decltype(&ggml_free)> c(ggml_init({8192,nullptr,true}),ggml_free);final_bank=ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,10240);
 std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> input(ggml_backend_alloc_ctx_tensors(c.get(),b.get()),ggml_backend_buffer_free);native_hc_need(bool(input),"final producer allocation");
 std::vector<float> norm(10240,1.f),vocab(5120,1.f);std::vector<uint8_t> q(3276800/32*34);for(size_t i=0;i<q.size();i+=34){const uint16_t scale=0x3c00;std::memcpy(q.data()+i,&scale,2);std::memset(q.data()+i+2,2,32);}
 std::vector<PortableTensorView> views{{"output_hc_norm.weight",{10240},0,norm.data(),norm.size()*4},{"output_hc_down.weight",{10240,320},8,q.data(),q.size()},{"output_hc_up.weight",{320,10240},8,q.data(),q.size()},{"output.weight",{2560,2},0,vocab.data(),vocab.size()*4}};
 PortableFfnOptions o;o.accelerator=b.get();o.request_accelerator=o.resident_dense_weights=o.native_head_hc_t1=true;o.native_head_hc_receipt_prefix=(std::filesystem::path(argv[1])/("head-product-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))).string();
 std::string error;const PortableHeadGeometry geo{2560,4,320,2,1e-6f};const auto weight=PortableHead::weight_allocation_size(b.get(),geo,views,true);auto low=o;low.weight_budget=weight-1;native_hc_need(!PortableHead::create(geo,views,low,error),"head weight budget refusal");
 low=o;low.graph_budget=(3u<<20)-1;native_hc_need(!PortableHead::create(geo,views,low,error),"head device bound budget refusal");
 auto wrong=views;wrong[1].type=0;native_hc_need(!PortableHead::create(geo,wrong,o,error),"head unsupported source encoding accepted");
 native_hc_need(!PortableHead::create({2560,4,319,2,1e-6f},views,o,error),"head unsupported rank accepted");
 const auto saved=q[1];q[1]=0x7c;native_hc_need(!PortableHead::create(geo,views,o,error),"head nonfinite source scale accepted");q[1]=saved;
 norm[0]=std::numeric_limits<float>::infinity();native_hc_need(!PortableHead::create(geo,views,o,error),"head nonfinite norm accepted");norm[0]=1.f;
 const auto admission=PortableHead::admission(geo,views,o,false);auto head=PortableHead::create(geo,views,o,error);native_hc_need(bool(head),error.c_str());native_hc_need(head->weight_bytes()<=weight&&head->graph_bytes()<=o.graph_budget&&head->memory_usage().device_state>0&&admission.graph_bound>=head->memory_usage().device_state,"head converted weights/owner admission");
 std::vector<float> logits;native_hc_need(head->run_device(final_bank,logits,error),error.c_str());native_hc_need(graph_links==1&&logits==std::vector<float>({2560.f,2560.f})&&head->native_hc_calls()==1,"actual head mixer/vocabulary link");
 PortableModelWork work;logits={42.f};const auto calls_before=head->native_hc_calls();native_hc_need(model_optional_head(false,work,logits,[&]{return head->run_device(final_bank,logits,error);})&&logits.empty()&&work.heads_skipped==1&&work.heads_executed==0&&head->native_hc_calls()==calls_before,"elided head did work or returned stale logits");
 fail_norm=true;native_hc_need(!head->run_device(final_bank,logits,error)&&logits.empty()&&head->native_hc_calls()==1,"head norm failure published logits");fail_norm=false;
 fail_mix=true;native_hc_need(!head->run_device(final_bank,logits,error)&&logits.empty()&&head->native_hc_calls()==1,"head mixer failure published logits");fail_mix=false;
 fail_graph=true;native_hc_need(!head->run_device(final_bank,logits,error)&&logits.empty()&&head->native_hc_calls()==2,"head projection failure/counter");fail_graph=false;
 native_hc_need(head->run_device(final_bank,logits,error)&&logits==std::vector<float>({2560.f,2560.f})&&head->native_hc_calls()==3,"head retry");
 native_hc_need(!head->run(std::vector<float>(10240),logits,error)&&logits.empty(),"head host-copy fallback accepted");
 std::puts("actual head owner/vocabulary graph, role conversion, budget, refusal and retry pass with host providers");return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
