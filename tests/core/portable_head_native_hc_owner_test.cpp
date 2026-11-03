#include "portable_native_head_hc.hpp"
#include "ggml-cpu.h"
#include <cstdio>
#include <chrono>
#include <fstream>
#include <limits>
#include <memory>
#include <vector>
using namespace strata::core;
using namespace strata::core::portable_detail;
namespace {const ggml_tensor* actual_residual=nullptr;const ggml_tensor* actual_normalized=nullptr;ggml_tensor* actual_mixed=nullptr;bool fail_norm=false,fail_mix=false,fail_domain=false;size_t norm_calls=0,mix_calls=0;}
namespace strata::core::portable_detail {
struct StrataVkNativeHc{};
StrataVkNativeHc* strata_vk_native_hc_create(ggml_backend_t,const char* p,size_t b){native_hc_need(b>=512,"mock norm budget");std::ofstream(p)<<"mock";return new StrataVkNativeHc;}
size_t strata_vk_native_hc_bytes(const StrataVkNativeHc*){return 512;}
void strata_vk_native_hc_destroy(StrataVkNativeHc* p)noexcept{delete p;}
void strata_vk_native_hc_norm_only(StrataVkNativeHc*,const ggml_tensor* r,const ggml_tensor*,ggml_tensor* n,ggml_tensor* rs){native_hc_need(!fail_norm&&r==actual_residual,"mock actual residual identity");actual_normalized=n;std::vector<float> v(10240,26.f),x(4,27.f);ggml_backend_tensor_set(n,v.data(),0,v.size()*4);ggml_backend_tensor_set(rs,x.data(),0,x.size()*4);++norm_calls;}
struct StrataVkNativeHcMix{};
StrataVkNativeHcMix* strata_vk_native_head_hc_mix_rn_create(ggml_backend_t,const char* p,size_t b){native_hc_need(b>=512,"mock mix budget");std::ofstream(p)<<"mock";return new StrataVkNativeHcMix;}
size_t strata_vk_native_hc_mix_bytes(const StrataVkNativeHcMix*){return 512;}
void strata_vk_native_hc_mix_destroy(StrataVkNativeHcMix* p)noexcept{delete p;}
void strata_vk_native_hc_mix_run(StrataVkNativeHcMix*,const ggml_tensor* n,const ggml_tensor*,const ggml_tensor*,ggml_tensor* pre,ggml_tensor* low,ggml_tensor* exp,ggml_tensor*,ggml_tensor*,ggml_tensor*,ggml_tensor* out){native_hc_need(!fail_mix&&n==actual_normalized,"mock actual norm producer identity");if(fail_domain){std::vector<float> v(320);for(auto item:{std::pair{pre,101.f},std::pair{exp,102.f},std::pair{low,103.f}}){std::fill(v.begin(),v.end(),item.second);ggml_backend_tensor_set(item.first,v.data(),0,v.size()*4);}throw NativeHcMixDomainRefusal(117);}actual_mixed=out;std::vector<float> v(2560,1.f);ggml_backend_tensor_set(out,v.data(),0,v.size()*sizeof(float));++mix_calls;}
}
template<class F>void reject(F f){bool caught=false;try{f();}catch(const std::invalid_argument&){caught=true;}native_hc_need(caught,"head owner refusal absent");}
int main(int argc,char**argv){try{
 native_hc_need(argc==2,"fresh receipt path required");
 std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> b(ggml_backend_cpu_init(),ggml_backend_free);
 std::unique_ptr<ggml_context,decltype(&ggml_free)> c(ggml_init({8192,nullptr,true}),ggml_free);
 auto* residual=ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,10240);auto* norm=ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,10240);
 auto* down=ggml_new_tensor_1d(c.get(),GGML_TYPE_I32,1638400);auto* up=ggml_new_tensor_1d(c.get(),GGML_TYPE_I32,1638400);auto* vocab=ggml_new_tensor_2d(c.get(),GGML_TYPE_F32,2560,2);
 std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> bank(ggml_backend_alloc_ctx_tensors(c.get(),b.get()),ggml_backend_buffer_free);
 native_hc_need(bool(b)&&bool(bank),"mock allocation");actual_residual=residual;
 const std::filesystem::path dir=std::filesystem::path(argv[1])/("head-hc-owner-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));reject([&]{PortableNativeHeadHc bad(b.get(),norm,down,up,1,dir);});
 PortableNativeHeadHc owner(b.get(),norm,down,up,1u<<20,dir);reject([&]{owner.mixed();});
 const auto exact=owner.bytes();PortableNativeHeadHc exact_owner(b.get(),norm,down,up,exact,dir.string()+"-exact");
 native_hc_need(exact_owner.bytes()==exact,"head owner exact budget");
 reject([&]{PortableNativeHeadHc bad(b.get(),norm,down,up,exact-1,dir.string()+"-short");});
 PortableFinalHcResidual producer{residual,24,42,3,47,residual,[](const void* o,const PortableFinalHcResidual& p){return o==p.tensor;}};
 auto* result=owner.run(producer);native_hc_need(result==actual_mixed&&owner.mixed()==result&&owner.calls()==1&&norm_calls==1&&mix_calls==1,"head owner producer identity/counters");
 std::vector<float> input_values(10240,24.f);ggml_backend_tensor_set(residual,input_values.data(),0,40960);owner.record_first(producer,true,true);
 std::ifstream before_file(dir/"first-residual.f32le",std::ios::binary);std::string before((std::istreambuf_iterator<char>(before_file)),{});
 std::fill(input_values.begin(),input_values.end(),26.f);ggml_backend_tensor_set(residual,input_values.data(),0,40960);producer.position=26;producer.write_sequence=4;fail_domain=true;
 bool domain_seen=false;try{owner.run(producer);}catch(const NativeHcMixDomainRefusal& e){domain_seen=e.status==117;}fail_domain=false;
 native_hc_need(domain_seen&&owner.calls()==1,"head exact domain status/counter");reject([&]{owner.mixed();});
 const auto refusal=dir/"refusal";std::ifstream mf(refusal/"metadata.json");std::string meta((std::istreambuf_iterator<char>(mf)),{});
 for(const auto* part:{"\"position\":26","\"producer_layer\":47","\"producer_owner_id\":42","\"producer_write_sequence\":4","\"attempted_call\":2","\"successful_calls\":1","\"status\":117","\"stage_status\":[117,0,0]","\"all_dispatches_completed\":true","\"accepted\":false","\"payload_bytes\":85776"})native_hc_need(meta.find(part)!=std::string::npos,"domain metadata differs");
 for(auto item:{std::pair{"residual",26.f},std::pair{"normalized",26.f},std::pair{"rs",27.f},std::pair{"low-pre",101.f},std::pair{"low-exp",102.f},std::pair{"low",103.f}}){auto path=refusal/(std::string(item.first)+".f32le");const size_t bytes=std::string(item.first)=="rs"?16:(std::string(item.first)=="residual"||std::string(item.first)=="normalized"?40960:1280);native_hc_need(std::filesystem::file_size(path)==bytes,"domain payload size differs");std::ifstream f(path,std::ios::binary);float value=0;f.read(reinterpret_cast<char*>(&value),sizeof(value));native_hc_need(value==item.second,"domain payload current value differs");}
 std::ifstream after_file(dir/"first-residual.f32le",std::ios::binary);std::string after((std::istreambuf_iterator<char>(after_file)),{});native_hc_need(before==after,"first successful packet changed");
 const auto generic_dir=dir.string()+"-generic";PortableNativeHeadHc generic(b.get(),norm,down,up,exact,generic_dir);fail_norm=true;reject([&]{generic.run(producer);});fail_norm=false;
 native_hc_need(std::filesystem::is_regular_file(std::filesystem::path(generic_dir)/"refusal/residual.f32le")&&!std::filesystem::exists(std::filesystem::path(generic_dir)/"refusal/normalized.f32le"),"generic refusal exposed scratch");
 std::ifstream gf(std::filesystem::path(generic_dir)/"refusal/metadata.json");std::string gm((std::istreambuf_iterator<char>(gf)),{});native_hc_need(gm.find("\"all_dispatches_completed\":false")!=std::string::npos&&gm.find("\"outputs_captured\":false")!=std::string::npos,"generic failure metadata");
 const auto io_dir=dir.string()+"-io";PortableNativeHeadHc io(b.get(),norm,down,up,exact,io_dir);std::filesystem::create_directory(std::filesystem::path(io_dir)/"refusal");fail_domain=true;domain_seen=false;try{io.run(producer);}catch(const NativeHcMixDomainRefusal& e){domain_seen=e.status==117;}fail_domain=false;native_hc_need(domain_seen&&io.calls()==0,"diagnostic failure hid domain exception");reject([&]{io.mixed();});
 native_hc_need(ggml_nbytes(result)==10240&&ggml_is_contiguous(result)&&result->buffer&&ggml_backend_buffer_get_type(result->buffer)==ggml_backend_get_default_buffer_type(b.get()),"head mixed producer buffer contract");
 std::vector<float> ones(5120,1.f);ggml_backend_tensor_set(vocab,ones.data(),0,ones.size()*sizeof(float));
 std::unique_ptr<ggml_context,decltype(&ggml_free)> graphctx(ggml_init({2u<<20,nullptr,true}),ggml_free);
 auto* logits=ggml_mul_mat(graphctx.get(),vocab,result);native_hc_need(logits->src[1]==result,"head vocabulary graph must consume the actual mixer bank");
 auto* graph=ggml_new_graph(graphctx.get());ggml_build_forward_expand(graph,logits);
 std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> graphbuf(ggml_backend_alloc_ctx_tensors(graphctx.get(),b.get()),ggml_backend_buffer_free);
 native_hc_need(bool(graphbuf)&&ggml_backend_graph_compute(b.get(),graph)==GGML_STATUS_SUCCESS,"head vocabulary mock consumer compute");
 float projected[2]{};ggml_backend_tensor_get(logits,projected,0,sizeof projected);native_hc_need(projected[0]==2560&&projected[1]==2560,"head actual mixed bank consumer value");
 fail_norm=true;reject([&]{owner.run(producer);});reject([&]{owner.mixed();});fail_norm=false;
 fail_mix=true;reject([&]{owner.run(producer);});reject([&]{owner.mixed();});fail_mix=false;
 native_hc_need(owner.calls()==1&&owner.run(producer)==result&&owner.calls()==2,"head retry/current-result contract");
 auto* old=residual->buffer;residual->buffer=nullptr;reject([&]{owner.run(producer);});reject([&]{owner.mixed();});residual->buffer=old;
 std::puts("head HC borrowed residual and mixed bank ownership, budget, failure and retry pass with host mock");return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
