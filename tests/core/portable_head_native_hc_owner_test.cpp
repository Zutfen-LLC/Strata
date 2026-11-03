#include "portable_native_head_hc.hpp"
#include "ggml-cpu.h"
#include <cstdio>
#include <chrono>
#include <fstream>
#include <limits>
#include <memory>
#include <vector>
using namespace strata::core::portable_detail;
namespace {const ggml_tensor* actual_residual=nullptr;const ggml_tensor* actual_normalized=nullptr;ggml_tensor* actual_mixed=nullptr;bool fail_norm=false,fail_mix=false;size_t norm_calls=0,mix_calls=0;}
namespace strata::core::portable_detail {
struct StrataVkNativeHc{};
StrataVkNativeHc* strata_vk_native_hc_create(ggml_backend_t,const char* p,size_t b){native_hc_need(b>=512,"mock norm budget");std::ofstream(p)<<"mock";return new StrataVkNativeHc;}
size_t strata_vk_native_hc_bytes(const StrataVkNativeHc*){return 512;}
void strata_vk_native_hc_destroy(StrataVkNativeHc* p)noexcept{delete p;}
void strata_vk_native_hc_norm_only(StrataVkNativeHc*,const ggml_tensor* r,const ggml_tensor*,ggml_tensor* n,ggml_tensor*){native_hc_need(!fail_norm&&r==actual_residual,"mock actual residual identity");actual_normalized=n;++norm_calls;}
struct StrataVkNativeHcMix{};
StrataVkNativeHcMix* strata_vk_native_hc_mix_create(ggml_backend_t,const char* p,size_t b){native_hc_need(b>=512,"mock mix budget");std::ofstream(p)<<"mock";return new StrataVkNativeHcMix;}
size_t strata_vk_native_hc_mix_bytes(const StrataVkNativeHcMix*){return 512;}
void strata_vk_native_hc_mix_destroy(StrataVkNativeHcMix* p)noexcept{delete p;}
void strata_vk_native_hc_mix_run(StrataVkNativeHcMix*,const ggml_tensor* n,const ggml_tensor*,const ggml_tensor*,ggml_tensor*,ggml_tensor*,ggml_tensor*,ggml_tensor*,ggml_tensor*,ggml_tensor*,ggml_tensor* out){native_hc_need(!fail_mix&&n==actual_normalized,"mock actual norm producer identity");actual_mixed=out;std::vector<float> v(2560,1.f);ggml_backend_tensor_set(out,v.data(),0,v.size()*sizeof(float));++mix_calls;}
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
 auto* result=owner.run(residual);native_hc_need(result==actual_mixed&&owner.mixed()==result&&owner.calls()==1&&norm_calls==1&&mix_calls==1,"head owner producer identity/counters");
 native_hc_need(ggml_nbytes(result)==10240&&ggml_is_contiguous(result)&&result->buffer&&ggml_backend_buffer_get_type(result->buffer)==ggml_backend_get_default_buffer_type(b.get()),"head mixed producer buffer contract");
 std::vector<float> ones(5120,1.f);ggml_backend_tensor_set(vocab,ones.data(),0,ones.size()*sizeof(float));
 std::unique_ptr<ggml_context,decltype(&ggml_free)> graphctx(ggml_init({2u<<20,nullptr,true}),ggml_free);
 auto* logits=ggml_mul_mat(graphctx.get(),vocab,result);native_hc_need(logits->src[1]==result,"head vocabulary graph must consume the actual mixer bank");
 auto* graph=ggml_new_graph(graphctx.get());ggml_build_forward_expand(graph,logits);
 std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> graphbuf(ggml_backend_alloc_ctx_tensors(graphctx.get(),b.get()),ggml_backend_buffer_free);
 native_hc_need(bool(graphbuf)&&ggml_backend_graph_compute(b.get(),graph)==GGML_STATUS_SUCCESS,"head vocabulary mock consumer compute");
 float projected[2]{};ggml_backend_tensor_get(logits,projected,0,sizeof projected);native_hc_need(projected[0]==2560&&projected[1]==2560,"head actual mixed bank consumer value");
 fail_norm=true;reject([&]{owner.run(residual);});reject([&]{owner.mixed();});fail_norm=false;
 fail_mix=true;reject([&]{owner.run(residual);});reject([&]{owner.mixed();});fail_mix=false;
 native_hc_need(owner.calls()==1&&owner.run(residual)==result&&owner.calls()==2,"head retry/current-result contract");
 auto* old=residual->buffer;residual->buffer=nullptr;reject([&]{owner.run(residual);});reject([&]{owner.mixed();});residual->buffer=old;
 std::puts("head HC borrowed residual and mixed bank ownership, budget, failure and retry pass with host mock");return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
