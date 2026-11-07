// Real composed publication and GGML products, transport-only scalar mocks.
#include "portable_native_hc_prefill_selection.hpp"
#include "portable_native_ffn_hc_factory.hpp"
#include "ggml-cpu.h"
#include <cstdio>
using namespace strata::core::portable_detail;
static void need(bool x,const char*m){if(!x)throw std::runtime_error(m);}
static std::vector<float> get(const ggml_tensor*t){std::vector<float>v(ggml_nbytes(t)/4);ggml_backend_tensor_get(t,v.data(),0,v.size()*4);return v;}
static void put(ggml_tensor*t,const std::vector<float>&v){ggml_backend_tensor_set(t,v.data(),0,v.size()*4);}
static int fail_stage=-1;static std::vector<int> order;
namespace strata::core::portable_detail {
struct StrataVkNativeHc{};struct StrataVkNativeHcMix{};struct StrataVkNativeHcPending{};struct StrataVkNativeHcPrefill{};
StrataVkNativeHc* strata_vk_native_hc_create(ggml_backend_t,const char*,size_t){throw std::runtime_error("legacy called");}
size_t strata_vk_native_hc_bytes(const StrataVkNativeHc*){return 0;}void strata_vk_native_hc_destroy(StrataVkNativeHc*) noexcept{}
void strata_vk_native_hc_run(StrataVkNativeHc*,const ggml_tensor*,const ggml_tensor*,const ggml_tensor*,ggml_tensor*,ggml_tensor*,ggml_tensor*,uint32_t){throw std::runtime_error("legacy norm called");}
StrataVkNativeHcMix* strata_vk_native_hc_mix_create(ggml_backend_t,const char*,size_t){throw std::runtime_error("legacy mix called");}
size_t strata_vk_native_hc_mix_bytes(const StrataVkNativeHcMix*){return 0;}void strata_vk_native_hc_mix_destroy(StrataVkNativeHcMix*) noexcept{}
void strata_vk_native_hc_mix_run(StrataVkNativeHcMix*,const ggml_tensor*,const ggml_tensor*,const ggml_tensor*,ggml_tensor*,ggml_tensor*,ggml_tensor*,ggml_tensor*,ggml_tensor*,ggml_tensor*,ggml_tensor*){throw std::runtime_error("legacy mix called");}
StrataVkNativeHcPending* strata_vk_native_hc_pending_create(ggml_backend_t,const char*,size_t cap){need(cap>=512,"pending cap");return new StrataVkNativeHcPending;}
size_t strata_vk_native_hc_pending_bytes(const StrataVkNativeHcPending*){return 512;}void strata_vk_native_hc_pending_destroy(StrataVkNativeHcPending*p) noexcept{delete p;}
void strata_vk_native_hc_pending_run(StrataVkNativeHcPending*,const ggml_tensor*,const ggml_tensor*,const ggml_tensor*,ggml_tensor*,ggml_tensor*,ggml_tensor*){throw std::runtime_error("unexpected pending");}
StrataVkNativeHcPrefill* strata_vk_native_hc_prefill_create(ggml_backend_t,const char*,size_t cap){need(cap>=512,"scalar cap");return new StrataVkNativeHcPrefill;}
size_t strata_vk_native_hc_prefill_bytes(const StrataVkNativeHcPrefill*){return 512;}void strata_vk_native_hc_prefill_destroy(StrataVkNativeHcPrefill*p) noexcept{delete p;}
void strata_vk_native_hc_prefill_run(StrataVkNativeHcPrefill*,uint32_t mode,uint32_t rows,const std::array<ggml_tensor*,6>&t){order.push_back(int(mode));if(fail_stage==int(mode))throw std::runtime_error("scalar fail");
 if(mode==0){auto v=get(t[0]);put(t[2],v);put(t[3],v);}else if(mode==1){auto v=get(t[0]);need(v[0]==2&&((rows==1)||v[320]==4),"down not copied before SiLU");std::vector<float>x(rows*320);x[0]=1;if(rows==2)x[320]=2;put(t[2],x);put(t[3],x);}else {auto v=get(t[1]);need(v[0]==3&&((rows==1)||v[10240]==6),"up not copied before inject/mix");std::vector<float>x(rows*2560,7);put(t[2],x);}}
}
template<class F>static bool fails(F f){try{f();return false;}catch(...){return true;}}
int main(){try{
 using S=PortableNativeHcPrefillSelection;using HC=PortableNativeHcComposed;using Tx=PortableNativePleTransaction;
 // Independent host image control: deriving FP16 from the BF16 image loses bits.
 need(ggml_fp32_to_bf16(1.00390625f).bits==0x3f80&&ggml_fp32_to_fp16(1.00390625f)==0x3c04,"independent image control");
 std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> b(ggml_backend_cpu_init(),ggml_backend_free);
 std::unique_ptr<ggml_context,decltype(&ggml_free)> c(ggml_init({16384,nullptr,true}),ggml_free);
 std::array<ggml_tensor*,4> w{ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,10240),ggml_new_tensor_1d(c.get(),GGML_TYPE_I32,20480),ggml_new_tensor_1d(c.get(),GGML_TYPE_I32,1638400),ggml_new_tensor_1d(c.get(),GGML_TYPE_I32,1638400)};
 std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buf(ggml_backend_alloc_ctx_tensors(c.get(),b.get()),ggml_backend_buffer_free);
 std::vector<uint16_t> big(3276800),inj(40960);big[0]=0x4000;ggml_backend_tensor_set(w[2],big.data(),0,big.size()*2);big[0]=0x4040;ggml_backend_tensor_set(w[3],big.data(),0,big.size()*2);inj[0]=0x4080;ggml_backend_tensor_set(w[1],inj.data(),0,inj.size()*2);
 std::array<const ggml_tensor*,4> roles{w[0],w[1],w[2],w[3]};S::Caps caps{512,512};size_t hb=S::host_bound(4),db=S::device_bound(b.get(),roles,caps);
 need(fails([&]{S::create(b.get(),roles,hb-1,db,"mock",caps);}),"short host");need(fails([&]{S::create(b.get(),roles,hb,db-1,"mock",caps);}),"short device");
 auto hc=HC::create_prefill(b.get(),roles,hb,db,"mock",caps);need(hc->read_method()==PortableHcReadMethod::baseline_prefill_bf16,"selected method accessor");buf.reset();c.reset(); // Immutable copied roles outlive the source.
 auto exec=[](const Tx::Request&r,const Tx::OutputView&o){size_t n=r.tokens.size();for(size_t i=0;i<n;++i){o.normalized[i]=float(r.tokens[i]);std::fill(o.residual.begin()+i*10240,o.residual.begin()+(i+1)*10240,float(r.tokens[i]));}std::copy(r.history.begin()+n,r.history.end(),o.history.begin());std::copy(o.normalized.begin(),o.normalized.end(),o.history.begin()+9-n);};
 Tx tx(1,10240,200000,exec);auto h=tx.stage(0,{1,2},std::vector<float>(20480));std::shared_ptr<HC::View> view;
 auto consume=[&](const HC::View&v,uint32_t n){need(n==2,"rows");auto x=v.read(HC::View::Field::inject);need(x[0]==4&&x[4]==8,"inject graph");need(v.read(HC::View::Field::mixed)[0]==7,"mixed");view=std::make_shared<HC::View>(v);};
 for(int mode:{0,1,2}){fail_stage=mode;need(fails([&]{hc->run(1,0,tx,h,consume,consume);})&&hc->rows()==0&&tx.position()==0,"scalar failure published");}fail_stage=-1;
 size_t aborts=0;ggml_backend_cpu_set_abort_callback(b.get(),[](void*p){++*static_cast<size_t*>(p);return true;},&aborts);need(fails([&]{hc->run(1,0,tx,h,consume,consume);})&&aborts&&!hc->rows()&&tx.position()==0,"graph abort published");ggml_backend_cpu_set_abort_callback(b.get(),nullptr,nullptr);
 order.clear();hc->run(1,0,tx,h,consume,consume);need(order==std::vector<int>({0,1,2})&&tx.position()==2&&view->rows()==2,"order/commit");auto old=view;
 auto h2=tx.stage(2,{1,2},std::vector<float>(20480));fail_stage=2;need(fails([&]{hc->run(1,2,tx,h2,consume,consume);})&&old->rows()==2&&tx.position()==2,"failure changed publication");fail_stage=-1;
 std::shared_ptr<HC::View> candidate;auto first=[&](const HC::View&v,uint32_t){candidate=std::make_shared<HC::View>(v);};auto bad=[](const HC::View&,uint32_t){throw std::runtime_error("consumer");};need(fails([&]{hc->run(1,2,tx,h2,first,bad);})&&fails([&]{candidate->rows();})&&old->rows()==2&&tx.position()==2,"consumer publication");need(fails([&]{hc->run(1,2,tx,h2,consume,consume);}),"consumer poison");
 hc->reset();need(fails([&]{old->rows();}),"reset stale");hc->run(1,2,tx,h2,consume,consume);need(tx.position()==4,"reset recovery");hc->reset();need(fails([&]{view->rows();}),"second reset stale");
 auto tail=tx.stage(4,{1},std::vector<float>(10240));auto one=[](const HC::View&v,uint32_t n){need(n==1&&v.read(HC::View::Field::inject)[0]==4,"T1 products");};hc->run(1,4,tx,tail,one,one);need(hc->rows()==1&&tx.position()==5,"T1 transition");
 auto again=tx.stage(5,{1,2},std::vector<float>(20480));hc->run(1,5,tx,again,consume,consume);need(hc->rows()==2&&tx.position()==7,"T2 reuse");
 // Exact named Q8 roles go through the real factory conversion for both halves.
 hc.reset();std::vector<float> norm(10240,1);std::vector<uint8_t> qi(40960/32*34),qd(3276800/32*34),qu(qd.size());
 for(auto* q:{&qi,&qd,&qu}){(*q)[0]=0;(*q)[1]=0x3c;}qi[2]=4;qd[2]=2;qu[2]=3;
 for(bool attention:{false,true}){std::string pre=std::string("blk.3.hc_")+(attention?"attn_":"ffn_");using strata::core::PortableTensorView;
  std::vector<PortableTensorView> views{{pre+"norm.weight",{10240},0,norm.data(),norm.size()*4},{pre+"inject.weight",{10240,4},8,qi.data(),qi.size()},{pre+"down.weight",{10240,320},8,qd.data(),qd.size()},{pre+"up.weight",{320,10240},8,qu.data(),qu.size()}};
  auto hbound=NativeFfnHcFactory::prefill_host_bound(4);need(fails([&]{NativeFfnHcFactory::create_prefill(3,b.get(),views,100u<<20,hbound-1,"mock",caps,attention);}),"factory host refusal");
  PortableHcReadSelection sel{PortableHcReadMethod::baseline_prefill_bf16,512,512};auto exact=NativeFfnHcFactory::selected_device_bound(b.get(),{},sel);
  need(fails([&]{NativeFfnHcFactory::create_selected(3,b.get(),views,exact-1,hbound,"mock",{},sel,attention);}),"factory device refusal");
  need(fails([&]{NativeFfnHcFactory::selected_device_bound(b.get(),{},PortableHcReadSelection{PortableHcReadMethod::baseline_prefill_bf16,1,512});}),"short component cap");
  need(fails([&]{NativeFfnHcFactory::selected_host_bound(4,PortableHcReadSelection{static_cast<PortableHcReadMethod>(99),0,0});}),"invalid method");
  auto selected=NativeFfnHcFactory::create_selected(3,b.get(),views,exact,hbound,"mock",{},sel,attention);need(selected->read_method()==PortableHcReadMethod::baseline_prefill_bf16,"factory selected method");uint16_t word=0;ggml_backend_tensor_get(selected->role_bank()->role(2),&word,0,2);need(word==0x4000,"factory down conversion");ggml_backend_tensor_get(selected->role_bank()->role(3),&word,0,2);need(word==0x4040,"factory up conversion");ggml_backend_tensor_get(selected->role_bank()->role(1),&word,0,2);need(word==0x4080,"factory injection conversion");}
 puts("HC selected composition: real CPU product graphs, mocked scalar stages; no device arithmetic");return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
