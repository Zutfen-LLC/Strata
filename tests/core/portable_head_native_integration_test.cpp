#include "strata/core/portable_model.hpp"
#include "strata_vk_native_hc.hpp"
#include "strata_vk_native_stages.hpp"
#include "strata_vk_q8_wo.hpp"
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
namespace {ggml_backend_t actual_backend=nullptr;ggml_tensor* final_bank=nullptr;ggml_tensor* norm_bank=nullptr;ggml_tensor* mixed_bank=nullptr;const ggml_tensor* packed_bank=nullptr;bool fail_norm=false,fail_mix=false,fail_graph=false,fail_domain=false,fail_vocab=false,nonfinite_vocab=false;size_t graph_links=0,quant_calls=0,dot_calls=0,fail_dot_at=SIZE_MAX;decltype(ggml_backend_i::graph_compute) original_compute=nullptr;
ggml_status inspect(ggml_backend_t b,ggml_cgraph* g){for(int i=0;i<ggml_graph_n_nodes(g);++i){auto* n=ggml_graph_node(g,i);if(n->op==GGML_OP_MUL_MAT&&std::string(ggml_get_name(n))=="portable_logits"){native_hc_need(n->src[1]==mixed_bank&&mixed_bank->buffer&&ggml_backend_buffer_get_type(mixed_bank->buffer)==ggml_backend_get_default_buffer_type(b),"head actual mixer/vocabulary bank or device differs");++graph_links;}}return fail_graph?GGML_STATUS_FAILED:original_compute(b,g);}}
namespace strata::core::portable_detail {
struct StrataVkNativeHc{};
StrataVkNativeHc* strata_vk_native_hc_create(ggml_backend_t,const char* p,size_t b){native_hc_need(b>=512,"mock norm budget");std::ofstream(p)<<"mock";return new StrataVkNativeHc;}
size_t strata_vk_native_hc_bytes(const StrataVkNativeHc*){return 512;}void strata_vk_native_hc_destroy(StrataVkNativeHc*p)noexcept{delete p;}
void strata_vk_native_hc_norm_only(StrataVkNativeHc*,const ggml_tensor* r,const ggml_tensor*,ggml_tensor* n,ggml_tensor* rs){native_hc_need(!fail_norm&&r==final_bank&&r->buffer,"head actual final producer differs");norm_bank=n;std::vector<float> x(10240,0.f),v(4,1.f);ggml_backend_tensor_set(n,x.data(),0,x.size()*4);ggml_backend_tensor_set(rs,v.data(),0,v.size()*4);}
struct StrataVkNativeHcMix{};
StrataVkNativeHcMix* strata_vk_native_head_hc_mix_rn_create(ggml_backend_t,const char* p,size_t b){native_hc_need(b>=512,"mock mix budget");std::ofstream(p)<<"mock";return new StrataVkNativeHcMix;}
size_t strata_vk_native_hc_mix_bytes(const StrataVkNativeHcMix*){return 512;}void strata_vk_native_hc_mix_destroy(StrataVkNativeHcMix*p)noexcept{delete p;}
void strata_vk_native_hc_mix_run(StrataVkNativeHcMix*,const ggml_tensor* n,const ggml_tensor* down,const ggml_tensor* up,ggml_tensor* lp,ggml_tensor* lo,ggml_tensor* le,ggml_tensor* pre,ggml_tensor* gates,ggml_tensor* ue,ggml_tensor* mixed){native_hc_need(!fail_mix&&n==norm_bank,"head actual normalized producer differs");if(fail_domain){for(auto item:{std::pair{lp,101.f},std::pair{le,102.f},std::pair{lo,103.f}}){std::vector<float> v(320,item.second);ggml_backend_tensor_set(item.first,v.data(),0,v.size()*4);}throw NativeHcMixDomainRefusal(117);}for(auto* w:{down,up}){std::vector<uint16_t> v(3276800);ggml_backend_tensor_get(w,v.data(),0,v.size()*2);for(auto x:v)native_hc_need(x==0x4000,"head native Q8 to BF16 role conversion differs");}for(auto*t:{lp,lo,le,pre,gates,ue}){std::vector<float> v(size_t(ggml_nelements(t)),0.f);ggml_backend_tensor_set(t,v.data(),0,v.size()*4);}std::vector<float> v(2560,1.f);ggml_backend_tensor_set(mixed,v.data(),0,v.size()*4);mixed_bank=mixed;}
struct StrataVkNativeStages{};
StrataVkNativeStages* strata_vk_native_stages_create_quant_rcp_observation(ggml_backend_t,const char* p,size_t b){native_hc_need(b>=512,"vocabulary quant budget");std::ofstream(p)<<"mock";return new StrataVkNativeStages;}
size_t strata_vk_native_stages_bytes(const StrataVkNativeStages*){return 512;}
void strata_vk_native_stages_destroy(StrataVkNativeStages*p)noexcept{delete p;}
void strata_vk_native_stages_run(StrataVkNativeStages*,uint32_t mode,const ggml_tensor* input,const ggml_tensor*,ggml_tensor*,ggml_tensor* packed,uint32_t n,const int32_t*,size_t){native_hc_need(mode==0&&n==2560&&input==mixed_bank&&input->buffer&&ggml_backend_buffer_get_type(input->buffer)==ggml_backend_get_default_buffer_type(actual_backend),"vocabulary ignored actual mixer bank or device");std::vector<uint8_t> bytes(2880,0);ggml_backend_tensor_set(packed,bytes.data(),0,bytes.size());packed_bank=packed;++quant_calls;}
struct StrataVkQ8Wo{};
StrataVkQ8Wo* strata_vk_head_vocab_create(ggml_backend_t,const char* p,size_t b){native_hc_need(b>=512,"vocabulary dot budget");std::ofstream(p)<<"mock";return new StrataVkQ8Wo;}
size_t strata_vk_q8_wo_bytes(const StrataVkQ8Wo*){return 512;}
void strata_vk_q8_wo_destroy(StrataVkQ8Wo*p)noexcept{delete p;}
void strata_vk_q8_wo_run(StrataVkQ8Wo*,const ggml_tensor* weight,const ggml_tensor* packed,ggml_tensor* output,size_t rows){native_hc_need(!fail_vocab&&dot_calls!=fail_dot_at&&weight->view_src&&weight->buffer&&packed==packed_bank&&output->buffer&&ggml_backend_buffer_get_type(weight->buffer)==ggml_backend_get_default_buffer_type(actual_backend)&&ggml_backend_buffer_get_type(output->buffer)==ggml_backend_get_default_buffer_type(actual_backend),"vocabulary ignored resident weight, packed producer or device");native_hc_need(weight->view_offs/2720==output->view_offs/4,"vocabulary row slice offset differs");std::vector<float> v(rows);for(size_t i=0;i<rows;++i)v[i]=float(output->view_offs/4+i+1);if(nonfinite_vocab)v[0]=std::numeric_limits<float>::infinity();ggml_backend_tensor_set(output,v.data(),0,v.size()*4);++dot_calls;}
}
int main(int argc,char**argv){try{
 native_hc_need(argc==2,"scratch root required");std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> b(ggml_backend_cpu_init(),ggml_backend_free);native_hc_need(bool(b),"mock backend");
 actual_backend=b.get();original_compute=b->iface.graph_compute;b->iface.graph_compute=inspect;
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
 PortableFinalHcResidual producer{final_bank,24,42,3,47,final_bank,[](const void* owner,const PortableFinalHcResidual& info){return owner==info.tensor;}};
 std::vector<float> logits;native_hc_need(head->run_device(producer,logits,error),error.c_str());native_hc_need(graph_links==1&&logits==std::vector<float>({2560.f,2560.f})&&head->native_hc_calls()==1,"actual head mixer/vocabulary link");
 producer.position=26;producer.write_sequence=4;std::vector<float> current(10240,26.f);ggml_backend_tensor_set(final_bank,current.data(),0,current.size()*4);
 const auto graph_before=graph_links;fail_domain=true;logits={42.f};native_hc_need(!head->run_device(producer,logits,error)&&error.find("status=117")!=std::string::npos&&logits.empty()&&head->native_hc_calls()==1&&graph_links==graph_before,"domain refusal reached vocabulary or committed");fail_domain=false;
 std::ifstream first_metadata(std::filesystem::path(o.native_head_hc_receipt_prefix)/"first.json");std::string first_text((std::istreambuf_iterator<char>(first_metadata)),{});native_hc_need(first_text.find("\"position\":24")!=std::string::npos&&first_text.find("\"producer_write_sequence\":3")!=std::string::npos,"first success was overwritten");
 const auto refusal=std::filesystem::path(o.native_head_hc_receipt_prefix)/"refusal";std::ifstream metadata(refusal/"metadata.json");std::string text((std::istreambuf_iterator<char>(metadata)),{});native_hc_need(text.find("\"position\":26")!=std::string::npos&&text.find("\"producer_write_sequence\":4")!=std::string::npos&&text.find("\"status\":117")!=std::string::npos,"integration refusal binding differs");
 PortableModelWork work;logits={42.f};const auto calls_before=head->native_hc_calls();native_hc_need(model_optional_head(false,work,logits,[&]{return head->run_device(producer,logits,error);})&&logits.empty()&&work.heads_skipped==1&&work.heads_executed==0&&head->native_hc_calls()==calls_before,"elided head did work or returned stale logits");
 fail_norm=true;native_hc_need(!head->run_device(producer,logits,error)&&logits.empty()&&head->native_hc_calls()==1,"head norm failure published logits");fail_norm=false;
 fail_mix=true;native_hc_need(!head->run_device(producer,logits,error)&&logits.empty()&&head->native_hc_calls()==1,"head mixer failure published logits");fail_mix=false;
 fail_graph=true;native_hc_need(!head->run_device(producer,logits,error)&&logits.empty()&&head->native_hc_calls()==2,"head projection failure/counter");fail_graph=false;
 native_hc_need(head->run_device(producer,logits,error)&&logits==std::vector<float>({2560.f,2560.f})&&head->native_hc_calls()==3,"head retry");
 auto invalid=producer;invalid.write_sequence=0;native_hc_need(!head->run_device(invalid,logits,error)&&logits.empty(),"missing producer identity accepted");
 native_hc_need(!head->run(std::vector<float>(10240),logits,error)&&logits.empty(),"head host-copy fallback accepted");
 std::vector<uint8_t> output_q8(2*80*34,0);for(size_t i=0;i<output_q8.size();i+=34){output_q8[i+1]=0x3c;output_q8[i+2]=1;}
 auto vocab_views=views;vocab_views[3]={"output.weight",{2560,2},8,output_q8.data(),output_q8.size()};auto vo=o;vo.native_head_vocab_t1=true;vo.native_head_hc_receipt_prefix=(std::filesystem::path(argv[1])/("native-vocab-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))).string();
 auto bad_role=vo;bad_role.native_head_hc_receipt_prefix+="-bad-role";native_hc_need(!PortableHead::create(geo,views,bad_role,error),"native vocabulary accepted F32 output role");
 auto nonfinite_role=vo;nonfinite_role.native_head_hc_receipt_prefix+="-nonfinite-role";output_q8[1]=0x7c;native_hc_need(!PortableHead::create(geo,vocab_views,nonfinite_role,error),"native vocabulary accepted nonfinite weight scale");output_q8[1]=0x3c;
 const auto vadmission=PortableHead::admission(geo,vocab_views,vo,false);auto vhead=PortableHead::create(geo,vocab_views,vo,error);native_hc_need(bool(vhead),error.c_str());native_hc_need(vhead->graph_bytes()<=vadmission.graph_bound&&vhead->memory_usage().device_state>head->memory_usage().device_state,"native vocabulary admission or ownership accounting");const auto old_graph=graph_links;
 native_hc_need(vhead->run_device(producer,logits,error)&&logits==std::vector<float>({1.f,2.f})&&quant_calls==1&&dot_calls==1&&vhead->native_vocab_calls()==1&&graph_links==old_graph,"native vocabulary producer or output link");
 fail_vocab=true;logits={42.f};native_hc_need(!vhead->run_device(producer,logits,error)&&logits.empty()&&vhead->native_vocab_calls()==1&&graph_links==old_graph,"failed native vocabulary published logits");fail_vocab=false;
 native_hc_need(vhead->run_device(producer,logits,error)&&logits==std::vector<float>({1.f,2.f})&&vhead->native_vocab_calls()==2,"native vocabulary retry");
 auto guard_opts=vo;guard_opts.native_head_hc_receipt_prefix+="-guard";auto guarded=PortableHead::create(geo,vocab_views,guard_opts,error,true);native_hc_need(bool(guarded),error.c_str());nonfinite_vocab=true;native_hc_need(!guarded->run_device(producer,logits,error)&&logits.empty(),"native vocabulary bypassed nonfinite guard");nonfinite_vocab=false;native_hc_need(guarded->run_device(producer,logits,error)&&logits==std::vector<float>({1.f,2.f}),"native vocabulary guard retry failed");
 constexpr size_t large_rows=16385;std::vector<uint8_t> large_output(large_rows*80*34,0);auto large_views=views;large_views[3]={"output.weight",{2560,int64_t(large_rows)},8,large_output.data(),large_output.size()};auto large_opts=vo;large_opts.native_head_hc_receipt_prefix+="-sliced";
 auto sliced=PortableHead::create({2560,4,320,int64_t(large_rows),1e-6f},large_views,large_opts,error);native_hc_need(bool(sliced),error.c_str());const auto old_q=quant_calls,old_dot=dot_calls;
 native_hc_need(sliced->run_device(producer,logits,error)&&logits.size()==large_rows&&logits.front()==1.f&&logits.back()==float(large_rows)&&quant_calls==old_q+1&&dot_calls==old_dot+2,"native vocabulary slicing or shared quantizer differs");
 fail_dot_at=dot_calls+1;logits={42.f};native_hc_need(!sliced->run_device(producer,logits,error)&&logits.empty(),"partial vocabulary slices published logits");fail_dot_at=SIZE_MAX;
 native_hc_need(sliced->run_device(producer,logits,error)&&logits.size()==large_rows&&logits.back()==float(large_rows),"partial vocabulary retry failed");
 std::puts("actual head owner/vocabulary graph, role conversion, budget, refusal and retry pass with host providers");return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
