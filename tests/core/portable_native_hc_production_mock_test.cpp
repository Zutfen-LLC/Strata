// CPU-only test of the real empty-Dispatch wiring; not shader or GPU evidence.
#include "portable_native_hc_composed.hpp"
#include "ggml-cpu.h"
#include <algorithm>
#include <array>
#include <cstdio>

using namespace strata::core::portable_detail;
namespace {
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
constexpr size_t pb=113, nb=227, mb=331, budget=20*1024*1024;
std::array<size_t,3> budgets{};
std::array<std::string,3> paths;
std::array<int,3> creates{}, destroys{}, runs{};
int fail_factory=-1;
void factory_call(int i) { ++creates[i]; if (i==fail_factory) throw std::runtime_error("injected factory failure"); }
std::array<std::vector<unsigned char>,4> expected;
const ggml_tensor* norm_output=nullptr;
void verify(const ggml_tensor* t, size_t role) {
 std::vector<unsigned char> got(ggml_nbytes(t)); ggml_backend_tensor_get(t,got.data(),0,got.size());
 check(got==expected[role],"captured role differs from copied source");
}
void fill(ggml_tensor* t, float value) {
 std::vector<float> v(size_t(ggml_nelements(t)),value); ggml_backend_tensor_set(t,v.data(),0,v.size()*sizeof(float));
}
}
namespace strata::core::portable_detail {
struct StrataVkNativeHcPending {};
struct StrataVkNativeHc {};
struct StrataVkNativeHcMix {};
StrataVkNativeHcPending* strata_vk_native_hc_pending_create(ggml_backend_t, const char* p,size_t b){factory_call(0); budgets[0]=b; paths[0]=p; return new StrataVkNativeHcPending;}
size_t strata_vk_native_hc_pending_bytes(const StrataVkNativeHcPending*){return pb;}
void strata_vk_native_hc_pending_destroy(StrataVkNativeHcPending* p) noexcept {if(p)++destroys[0]; delete p;}
void strata_vk_native_hc_pending_run(StrataVkNativeHcPending*,const ggml_tensor*,const ggml_tensor*,const ggml_tensor*,ggml_tensor*,ggml_tensor*,ggml_tensor*){++runs[0]; throw std::runtime_error("unexpected pending");}
StrataVkNativeHc* strata_vk_native_hc_create(ggml_backend_t,const char* p,size_t b){factory_call(1); budgets[1]=b; paths[1]=p; return new StrataVkNativeHc;}
size_t strata_vk_native_hc_bytes(const StrataVkNativeHc*){return nb;}
void strata_vk_native_hc_destroy(StrataVkNativeHc* p) noexcept {if(p)++destroys[1]; delete p;}
void strata_vk_native_hc_run(StrataVkNativeHc*,const ggml_tensor* x,const ggml_tensor* w,const ggml_tensor* iw,ggml_tensor* out,ggml_tensor* rs,ggml_tensor* inject,uint32_t n){
 ++runs[1]; verify(w,0); verify(iw,1); check(ggml_nbytes(x)==n*10240*sizeof(float),"norm full batch");
 std::vector<float> v(n*10240); ggml_backend_tensor_get(x,v.data(),0,v.size()*sizeof(float));
 for (uint32_t r=0;r<n;++r) for(size_t i=0;i<10240;++i) check(v[r*10240+i]==float(n==2?r+1:3),"materialized residual rows");
 for(auto& a:v) a+=10; ggml_backend_tensor_set(out,v.data(),0,v.size()*sizeof(float)); fill(rs,2); fill(inject,3); norm_output=out;
}
StrataVkNativeHcMix* strata_vk_native_hc_mix_create(ggml_backend_t,const char* p,size_t b){factory_call(2); budgets[2]=b; paths[2]=p; return new StrataVkNativeHcMix;}
size_t strata_vk_native_hc_mix_bytes(const StrataVkNativeHcMix*){return mb;}
void strata_vk_native_hc_mix_destroy(StrataVkNativeHcMix* p) noexcept {if(p)++destroys[2]; delete p;}
void strata_vk_native_hc_mix_run(StrataVkNativeHcMix*,const ggml_tensor* x,const ggml_tensor* down,const ggml_tensor* up,ggml_tensor* lp,ggml_tensor* low,ggml_tensor* le,ggml_tensor* u,ggml_tensor* gates,ggml_tensor* ue,ggml_tensor* mixed){
 ++runs[2]; verify(down,2); verify(up,3); check(x==norm_output,"mix did not use norm output");
 const size_t n=ggml_nelements(mixed)/2560; check((n==1||n==2)&&ggml_nelements(x)==n*10240,"mix full batch");
 std::vector<float> v(n*10240), out(n*2560); ggml_backend_tensor_get(x,v.data(),0,v.size()*sizeof(float));
 for(size_t r=0;r<n;++r) std::fill(out.begin()+r*2560,out.begin()+(r+1)*2560,v[r*10240]+20);
 for(auto* t:{lp,low,le,u,gates,ue}) fill(t,4); ggml_backend_tensor_set(mixed,out.data(),0,out.size()*sizeof(float));
}
}
int main(){try {
 using Owner=PortableNativeHcComposed; using Tx=PortableNativePleTransaction;
 std::shared_ptr<ggml_backend> cpu(ggml_backend_cpu_init(),ggml_backend_free); check(bool(cpu),"CPU backend");
 std::shared_ptr<ggml_context> ctx(ggml_init({8192,nullptr,true}),ggml_free);
 std::array<ggml_tensor*,4> source={ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,10240),ggml_new_tensor_1d(ctx.get(),GGML_TYPE_I32,20480),ggml_new_tensor_1d(ctx.get(),GGML_TYPE_I32,1638400),ggml_new_tensor_1d(ctx.get(),GGML_TYPE_I32,1638400)};
 std::shared_ptr<ggml_backend_buffer> storage(ggml_backend_alloc_ctx_tensors(ctx.get(),cpu.get()),ggml_backend_buffer_free); check(bool(storage),"source allocation");
 for(size_t i=0;i<4;++i){expected[i].assign(ggml_nbytes(source[i]),static_cast<unsigned char>(0x31+i)); ggml_backend_tensor_set(source[i],expected[i].data(),0,expected[i].size());}
 {
  const std::array<const ggml_tensor*,4> descriptors={source[0],source[1],source[2],source[3]};
  const size_t base=Owner::tensor_bound(cpu.get(),descriptors);
  const Owner::ComponentCaps caps{pb+7,nb+11,mb+13};
  const size_t exact=base+caps.pending+caps.norm+caps.mix;
  auto construct=[&](size_t b, Owner::ComponentCaps c) {
   Owner owner(cpu.get(),source[0],source[1],source[2],source[3],b,"mock-receipt",{},SIZE_MAX,c);
  };
  // Throw at each actual empty-Dispatch factory, before it returns an owner.
  for (int i=0;i<3;++i) {
   creates={}; destroys={}; fail_factory=i;
   bool injected=false;
   try { construct(exact,caps); } catch(const std::runtime_error& e) { injected=std::string(e.what())=="injected factory failure"; }
   fail_factory=-1;
   std::array<int,3> calls{}, released{};
   for(int j=0;j<=i;++j) calls[j]=1;
   for(int j=0;j<i;++j) released[j]=1;
   check(injected && creates==calls && destroys==released,"factory failure did not unwind exactly or ran a later factory");
  }
  creates={}; destroys={};
  bool rejected=false; try { construct(exact-1,caps); } catch(const std::runtime_error&) { rejected=true; }
  check(rejected && creates==std::array<int,3>{0,0,0},"short caps budget called factory");
  construct(exact,caps);
  check(creates==std::array<int,3>{1,1,1} && budgets==std::array<size_t,3>{caps.pending,caps.norm,caps.mix},"exact caps or factory arguments");
  for (int i=0;i<3;++i) {
   creates={}; destroys={}; auto small=caps;
   if(i==0) small.pending=pb-1; if(i==1) small.norm=nb-1; if(i==2) small.mix=mb-1;
   rejected=false; try { construct(exact,small); } catch(const std::runtime_error&) { rejected=true; }
   std::array<int,3> expected_calls{}; for(int j=0;j<=i;++j) expected_calls[j]=1;
   check(rejected && creates==expected_calls && destroys==expected_calls,"over-cap component accepted or leaked");
  }
  creates={}; destroys={}; rejected=false;
  try { construct(SIZE_MAX,Owner::ComponentCaps{SIZE_MAX,1,1}); } catch(const std::overflow_error&) { rejected=true; }
  check(rejected && creates==std::array<int,3>{0,0,0},"caps overflow called factory");
  creates={}; destroys={};
 }
 {
 Owner owner(cpu.get(),source[0],source[1],source[2],source[3],budget,"mock-receipt");
 check(creates==std::array<int,3>{1,1,1},"factory calls");
 check(paths==std::array<std::string,3>{"mock-receipt.pending.spv","mock-receipt.norm.spv","mock-receipt.mix.spv"},"factory paths");
 // CPU scratch allocation is exactly its allocator bound. Infer it independently
 // from actual accounting, and cross-check against the factory budget.
 const size_t roles=owner.role_bank()->bytes(), scratch=owner.bytes()-roles-pb-nb-mb;
 check(budgets[0]==budget-roles-scratch && budgets[1]==budgets[0]-pb && budgets[2]==budgets[1]-nb,"factory budget subtraction");
 storage.reset(); ctx.reset();
 for(size_t i=0;i<4;++i) verify(owner.role_bank()->role(i),i);
 auto executor=[](const Tx::Request& r, const Tx::OutputView& o){ size_t n=r.tokens.size();
   for(size_t i=0;i<n;++i) { o.normalized[i]=float(r.tokens[i]); std::fill(o.residual.begin()+i*10240,o.residual.begin()+(i+1)*10240,float(r.tokens[i])); }
   std::copy(r.history.begin()+n,r.history.end(),o.history.begin());
   std::copy(o.normalized.begin(),o.normalized.end(),o.history.begin()+9-n); };
 Tx tx(1,10240,200000,executor); std::shared_ptr<Owner::View> prior, failed;
 auto capture=[&](const Owner::View& v,uint32_t n){check(v.rows()==n,"view rows"); auto values=v.read(Owner::View::Field::mixed); check(values.size()==n*2560,"mixed size"); for(size_t r=0;r<n;++r) for(size_t i=0;i<2560;++i) check(values[r*2560+i]==float((n==2?r+1:3)+30),"mixed values"); prior=std::make_shared<Owner::View>(v);};
 auto noop=[](const Owner::View&,uint32_t){};
 auto h=tx.stage(0,{1,2},std::vector<float>(20480)); owner.run(1,0,tx,h,capture,noop);
 check(tx.position()==2 && owner.rows()==2 && runs==std::array<int,3>{0,1,1},"T2 dispatch");
 auto tail=tx.stage(2,{3},std::vector<float>(10240));
 auto capture_failed=[&](const Owner::View& v,uint32_t){failed=std::make_shared<Owner::View>(v);};
 auto throws=[](const Owner::View&,uint32_t){throw std::runtime_error("consumer failure");};
 bool threw=false; try {owner.run(1,2,tx,tail,capture_failed,throws);} catch(const std::runtime_error&){threw=true;}
 check(threw && tx.position()==2 && owner.rows()==2 && prior->read(Owner::View::Field::mixed).size()==5120,"failed consumer publication");
 bool stale=false; try{failed->rows();}catch(const std::runtime_error&){stale=true;} check(stale,"failed view valid");
 threw=false; try{owner.run(1,2,tx,tail,capture,noop);}catch(const std::runtime_error&){threw=true;} check(threw && runs==std::array<int,3>{0,2,2},"poison did not block dispatch");
 owner.reset(); stale=false; try{prior->rows();}catch(const std::runtime_error&){stale=true;} check(stale && owner.rows()==0,"reset publication");
 owner.run(1,2,tx,tail,capture,noop); check(tx.position()==3 && owner.rows()==1 && runs==std::array<int,3>{0,3,3},"T1 dispatch");
 owner.reset(); check(owner.rows()==0,"final reset");
 }
 check(destroys==std::array<int,3>{1,1,1},"component destruction");
 std::puts("composed HC production wiring CPU mock passed"); return 0;
 }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what()); return 1;}}
