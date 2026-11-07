// Actual PLE/HC composition and GDN owners, with low-level component mocks.
#define STRATA_CHAIN_QSA_MOCKS 1
#include "../kernels/native_qsa_norm_rope_fixture.hpp"
#include "../kernels/native_qsa_indexer_fixture.hpp"
#include "../kernels/native_qsa_select_fixture.hpp"
#include "../kernels/native_qsa_attention_fixture.hpp"
#define main prior_ple_integration_main
#include "portable_native_ple_integration_test.cpp"
#undef main
#include "portable_native_ple_gdn_composition.hpp"
#include "portable_native_ffn_test_access.hpp"
#include <map>
namespace {
void need(bool b,const char*m){if(!b)throw std::runtime_error(m);}
int fail=0;size_t conv_calls=0,ab_calls=0,recur_calls=0;
std::vector<float> get(const ggml_tensor*t){std::vector<float>v(ggml_nbytes(t)/4);ggml_backend_tensor_get(t,v.data(),0,v.size()*4);return v;}
void put(ggml_tensor*t,const std::vector<float>&v){need(ggml_nbytes(t)==v.size()*4,"mock span");ggml_backend_tensor_set(t,v.data(),0,v.size()*4);}
}
 #include "portable_native_chain_qsa_mocks.hpp"
namespace {std::vector<float> chain_qsa_input,chain_qsa_gated; size_t chain_qsa_qkv=0;}
namespace strata::core::portable_detail {
StrataVkQ8Wo* strata_vk_qsa_qkv_rows_create(ggml_backend_t,const char*p,size_t){std::ofstream(p)<<"mock";return new StrataVkQ8Wo;}
StrataVkQ8Wo* strata_vk_qsa_wo_rows_create(ggml_backend_t,const char*p,size_t){std::ofstream(p)<<"mock";return new StrataVkQ8Wo;}
StrataVkNativeShared* strata_vk_qsa_indexer_bf16_rows_create(ggml_backend_t,const char*p,size_t){std::ofstream(p)<<"mock";return new StrataVkNativeShared;}
void strata_vk_qsa_input_quantize_bounded(StrataVkNativeStages*,const ggml_tensor*x,ggml_tensor*q,size_t rows){chain_qsa_input=get(x);std::vector<uint8_t> packed(rows*2880);ggml_backend_tensor_set(q,packed.data(),0,packed.size());}
void strata_vk_qsa_qkv_run_bounded(StrataVkQ8Wo*,const ggml_tensor*,const ggml_tensor*,ggml_tensor*y,size_t n,size_t rows){size_t role=chain_qsa_qkv++%3;std::vector<float> v(rows*n);for(size_t r=0;r<rows;++r)for(size_t i=0;i<n;++i)v[r*n+i]=chain_qsa_input[r*2560]+float(int((i+role*3)%17)-8)/32.f;put(y,v);}
void strata_vk_qsa_wo_quantize_bounded(StrataVkNativeStages*,const ggml_tensor*x,ggml_tensor*q,size_t rows){chain_qsa_gated=get(x);std::vector<uint8_t> packed(rows*6912);ggml_backend_tensor_set(q,packed.data(),0,packed.size());}
void strata_vk_qsa_wo_run_bounded(StrataVkQ8Wo*,const ggml_tensor*,const ggml_tensor*,ggml_tensor*y,size_t rows){std::vector<float> v(rows*2560);for(size_t r=0;r<rows;++r)std::copy_n(chain_qsa_gated.begin()+r*6144,2560,v.begin()+r*2560);put(y,v);}
struct StrataVkNativeGdnConvRows{};struct StrataVkNativeGdnAbRows{};struct StrataVkNativeGdnRows{};
StrataVkNativeGdnConvRows* strata_vk_native_gdn_conv_rows_create(ggml_backend_t,const char*p,size_t){std::ofstream(p)<<"mock";if(fail==1)throw std::runtime_error("conv factory");return new StrataVkNativeGdnConvRows;}
void strata_vk_native_gdn_conv_rows_destroy(StrataVkNativeGdnConvRows*p)noexcept{delete p;}size_t strata_vk_native_gdn_conv_rows_bytes(const StrataVkNativeGdnConvRows*){return 512;}
StrataVkNativeGdnAbRows* strata_vk_native_gdn_ab_rows_create(ggml_backend_t,const char*p,size_t){std::ofstream(p)<<"mock";if(fail==2)throw std::runtime_error("AB factory");return new StrataVkNativeGdnAbRows;}
void strata_vk_native_gdn_ab_rows_destroy(StrataVkNativeGdnAbRows*p)noexcept{delete p;}size_t strata_vk_native_gdn_ab_rows_bytes(const StrataVkNativeGdnAbRows*){return 512;}
StrataVkNativeGdnRows* strata_vk_native_gdn_rows_create(ggml_backend_t,const char*p,size_t){std::ofstream(p)<<"mock";if(fail==3)throw std::runtime_error("recurrence factory");return new StrataVkNativeGdnRows;}
void strata_vk_native_gdn_rows_destroy(StrataVkNativeGdnRows*p)noexcept{delete p;}size_t strata_vk_native_gdn_rows_bytes(const StrataVkNativeGdnRows*){return 512;}
void strata_vk_native_gdn_conv_rows_run(StrataVkNativeGdnConvRows*,ggml_tensor*const t[10],uint32_t n){++conv_calls;auto h=get(t[0]),q=get(t[1]);std::vector<float>out(n*10240);for(size_t r=0;r<n;++r)for(size_t c=0;c<10240;++c){out[r*10240+c]=q[r*10240+c]+h[c*3];h[c*3]=h[c*3+1];h[c*3+1]=h[c*3+2];h[c*3+2]=q[r*10240+c];}put(t[3],h);put(t[4],out);if(fail==4)throw std::runtime_error("conv wrote then threw");}
uint32_t strata_vk_native_gdn_ab_rows_run(StrataVkNativeGdnAbRows*,ggml_tensor*const t[8],uint32_t n){++ab_calls;auto mixed=get(t[0]);std::vector<float>gate(n*48),beta(n*48);for(size_t r=0;r<n;++r)for(size_t i=0;i<48;++i){gate[r*48+i]=mixed[r*2560];beta[r*48+i]=1;}put(t[5],gate);put(t[6],beta);return fail==5?1:0;}
void strata_vk_native_gdn_rows_run(StrataVkNativeGdnRows*,ggml_tensor*const t[13],ggml_tensor*,uint32_t n){++recur_calls;auto state=get(t[0]),h=get(t[1]),gate=get(t[2]),z=get(t[4]);std::vector<float>y(n*6144);for(size_t r=0;r<n;++r)for(size_t i=0;i<6144;++i){state[i]+=h[r*10240+i]+gate[r*48]+z[r*6144+i];y[r*6144+i]=state[i];}put(t[6],state);put(t[7],y);if(fail==6)throw std::runtime_error("recurrence wrote then threw");}
}

namespace { std::vector<float> gdn_quant_input; bool gdn_bad_second_scale=false; size_t in_quant=0,in_qkv=0,in_z=0,out_quant=0,out_dot=0; }
namespace strata::core::portable_detail {
StrataVkQ8Wo* strata_vk_gdn_input_create(ggml_backend_t,const char*p,size_t){std::ofstream(p)<<"mock";inject("gdn input factory");return new StrataVkQ8Wo;}
StrataVkQ8Wo* strata_vk_gdn_output_rows_create(ggml_backend_t,const char*p,size_t){std::ofstream(p)<<"mock";inject("gdn output factory");return new StrataVkQ8Wo;}
void strata_vk_gdn_input_quantize_bounded(StrataVkNativeStages*,const ggml_tensor*x,ggml_tensor*q,size_t rows){++in_quant;gdn_quant_input=get(x);std::vector<uint8_t>p(rows*2880);if(gdn_bad_second_scale&&rows==2)p[2880]=1;ggml_backend_tensor_set(q,p.data(),0,p.size());write_then_throw("gdn input quant",{q},9);}
void strata_vk_gdn_input_run_bounded(StrataVkQ8Wo*,const ggml_tensor*,const ggml_tensor*,ggml_tensor*y,size_t columns,size_t rows){if(columns==10240)++in_qkv;else ++in_z;std::vector<float>v(rows*columns);for(size_t r=0;r<rows;++r)for(size_t i=0;i<columns;++i)v[r*columns+i]=gdn_quant_input[r*2560]+float(i%7)/16;put(y,v);write_then_throw(columns==10240?"gdn qkv":"gdn z",{y},9);}
void strata_vk_gdn_output_quantize_bounded(StrataVkNativeStages*,const ggml_tensor*x,ggml_tensor*q,size_t rows){++out_quant;gdn_quant_input=get(x);std::vector<uint8_t>p(rows*6912);ggml_backend_tensor_set(q,p.data(),0,p.size());write_then_throw("gdn output quant",{q},9);}
void strata_vk_gdn_output_run_bounded(StrataVkQ8Wo*,const ggml_tensor*,const ggml_tensor*,ggml_tensor*y,size_t rows){++out_dot;std::vector<float>v(rows*2560);for(size_t r=0;r<rows;++r)for(size_t i=0;i<2560;++i)v[r*2560+i]=gdn_quant_input[r*6144+i];put(y,v);write_then_throw("gdn output dot",{y},9);}
}

int main(){try{
 std::string error;auto table=ple_fixture::small_table(error);check(bool(table),"fixture table");auto b=std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)>(ggml_backend_cpu_init(),ggml_backend_free);
 auto c=std::unique_ptr<ggml_context,decltype(&ggml_free)>(ggml_init({32768,nullptr,true}),ggml_free);auto*k=ggml_new_tensor_2d(c.get(),GGML_TYPE_Q8_0,2560,10240);auto*v=ggml_new_tensor_1d(c.get(),GGML_TYPE_I32,2560*2560/2);
 auto*qi=ggml_new_tensor_2d(c.get(),GGML_TYPE_Q8_0,2560,10240);auto*zi=ggml_new_tensor_2d(c.get(),GGML_TYPE_Q8_0,2560,6144);auto*wo=ggml_new_tensor_2d(c.get(),GGML_TYPE_Q8_0,6144,2560);
 std::array<ggml_tensor*,6> gr{};const size_t counts[]={40960,61440,61440,48,48,128};for(size_t i=0;i<6;++i)gr[i]=ggml_new_tensor_1d(c.get(),(i==1||i==2)?GGML_TYPE_I32:GGML_TYPE_F32,counts[i]);
 auto buf=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>(ggml_backend_alloc_ctx_tensors(c.get(),b.get()),ggml_backend_buffer_free);
 std::vector<uint8_t>kr(ggml_nbytes(k)),qr(ggml_nbytes(qi)),zr(ggml_nbytes(zi)),wr(ggml_nbytes(wo));std::vector<uint16_t>vr(2560*2560),cw(40960,0x3c00);vr[0]=0x3f80;vr[2560]=0x4000;for(auto pair:{std::pair{k,&kr},std::pair{qi,&qr},std::pair{zi,&zr},std::pair{wo,&wr}})ggml_backend_tensor_set(pair.first,pair.second->data(),0,pair.second->size());ggml_backend_tensor_set(v,vr.data(),0,vr.size()*2);
 std::array<std::vector<float>,3>gam{std::vector<float>(10240,1),std::vector<float>(10240,1),std::vector<float>(10240,1)};auto root=std::filesystem::temp_directory_path()/("ple-gdn-composition-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
 PortableNativePleExecutor::Roles roles{b.get(),table.get(),k,kr.data(),kr.size(),v,vr.data(),vr.size(),{gam[0].data(),gam[1].data(),gam[2].data()},cw.data(),cw.size()*2,1,root.string()};
 auto hcctx=std::unique_ptr<ggml_context,decltype(&ggml_free)>(ggml_init({8192,nullptr,true}),ggml_free);std::array<const ggml_tensor*,4>hc{ggml_new_tensor_1d(hcctx.get(),GGML_TYPE_F32,10240),ggml_new_tensor_1d(hcctx.get(),GGML_TYPE_I32,20480),ggml_new_tensor_1d(hcctx.get(),GGML_TYPE_I32,1638400),ggml_new_tensor_1d(hcctx.get(),GGML_TYPE_I32,1638400)};auto hcbuf=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>(ggml_backend_alloc_ctx_tensors(hcctx.get(),b.get()),ggml_backend_buffer_free);
 for(size_t i=0;i<4;++i){hc_roles[i].assign(ggml_nbytes(hc[i]),0x31+i);ggml_backend_tensor_set(const_cast<ggml_tensor*>(hc[i]),hc_roles[i].data(),0,hc_roles[i].size());}
 PortableNativeHcComposed::ComponentCaps hc_caps{113,227,331};auto prefix=(root/"hc").string();auto bounds=PortableNativePleAdmission::calculate(b.get(),hc,roles.directory,prefix.size(),hc_caps);
 PortableNativePleComposition ple(roles,hc,b.get(),hc_caps,bounds.total,prefix);
 for(int which=0;which<4;++which){bool caught=false;fail_at=which==1?"gdn input factory":which==3?"gdn output factory":"";factory_probe=which==0||which==2;factory_fail=0;factory_calls.clear();factory_destroyed.clear();
  try{if(which<2){PortableNativeGdnInputRows bad(b.get(),{qi,zi},{qr.data(),zr.data()},{qr.size(),zr.size()},(root/("bad-input-"+std::to_string(which))).string(),PortableNativeGdnInputRows::estimate(b.get()));}else{PortableNativeGdnOutputRows bad(b.get(),wo,wr.data(),wr.size(),(root/("bad-output-"+std::to_string(which))).string(),PortableNativeGdnOutputRows::estimate(b.get()));}}catch(const std::exception&){caught=true;}
  factory_probe=false;fail_at.clear();check(caught&&factory_owners.empty(),"projection factory failure/leak");}
 PortableNativeGdnInputRows in(b.get(),{qi,zi},{qr.data(),zr.data()},{qr.size(),zr.size()},(root/"gdn-in").string(),PortableNativeGdnInputRows::estimate(b.get()));
 PortableNativeGdnCoreRows::Caps caps{512,512,512};auto core=std::make_unique<PortableNativeGdnCoreRows>(b.get(),gr,caps,PortableNativeGdnCoreRows::estimate(b.get(),caps),(root/"gdn-core").string());
 PortableNativeGdnOutputRows out(b.get(),wo,wr.data(),wr.size(),(root/"gdn-out").string(),PortableNativeGdnOutputRows::estimate(b.get()));
 using Owner=PortableNativePleGdnComposition;Owner owner(ple,in,std::move(core),out,Owner::host_bound());owner.reset({3,5});
 bool distinct_stage=false;auto stage=[&](std::vector<int32_t>tokens){std::vector<std::array<uint32_t,16>>indices;auto prev=ple.previous();std::vector<std::array<int32_t,2>>pred;for(auto t:tokens){pred.push_back(prev);prev={prev[1],t};}check(table->gather_batch(tokens,pred,indices,expected_embedding,error),"gather");norm_call=0;std::vector<float> input(tokens.size()*10240,0.25f);if(distinct_stage)for(size_t r=0;r<tokens.size();++r)std::fill_n(input.begin()+r*10240,10240,float(tokens[r])/32);return ple.stage(ple.position(),tokens,input);};
 // Drive the actual PLE/GDN transaction through the standalone FFN chain.
 {
  using HC=PortableNativeHcComposed;
  auto metadata=std::unique_ptr<ggml_context,decltype(&ggml_free)>(ggml_init({8192,nullptr,true}),ggml_free);
  auto* rw=ggml_new_tensor_2d(metadata.get(),GGML_TYPE_F32,2560,512);auto rb=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>(ggml_backend_alloc_ctx_tensors(metadata.get(),b.get()),ggml_backend_buffer_free);
  std::vector<float> zero(2560*512);ggml_backend_tensor_set(rw,zero.data(),0,zero.size()*4);PortableNativeFfnProjection projection(b.get(),rw,16u<<20);
  bool connected_combine=false;PortableNativeFfnRows::Dispatch d;d.router=[&](size_t,const ggml_tensor*,std::vector<PortableRoute>&r){for(auto&v:r){v.native_router=true;int base=connected_combine?int(std::fabs(v.activation[0])*64)%400:0;for(int k=0;k<10;++k){v.ids.push_back(base+k);v.weights.push_back(connected_combine?float((k+1)*(1+base%3))/128.f:.1f);}}};
  d.shared=[](size_t n,const ggml_tensor*){return std::vector<float>(n*2560,1.f);};d.combine=[&](size_t n,const auto& parts,const auto& weights,const auto&s){auto result=s;if(connected_combine)for(size_t r=0;r<n;++r)for(size_t k=0;k<10;++k)for(size_t h=0;h<2560;++h)result[r*2560+h]+=parts[(r*10+k)*2560+h]*weights[r*10+k];return result;};PortableNativeFfnRows row_owner(1,projection,d);
  HC::Dispatch dispatch;dispatch.pending=[](const HC::Bank&bank){auto r=get(bank.input),o=get(bank.block);for(size_t t=0;t<o.size()/2560;++t)for(size_t s=0;s<4;++s)for(size_t h=0;h<2560;++h)r[t*10240+s*2560+h]+=o[t*2560+h];put(bank.residual,r);};
  dispatch.norm=[](const HC::Bank&bank,uint32_t n){put(bank.inject,std::vector<float>(n*4,.25f));};dispatch.mix=[](const HC::Bank&bank){auto r=get(bank.residual);std::vector<float>m(r.size()/4);for(size_t t=0;t<m.size()/2560;++t)std::copy_n(r.begin()+t*10240,2560,m.begin()+t*2560);put(bank.mixed,m);};
  auto ff=std::make_unique<HC>(b.get(),hc[0],hc[1],hc[2],hc[3],20*1024*1024,nullptr,dispatch);auto chain=PortableNativeFfnCompositionTestAccess::make(owner,std::move(ff),row_owner,PortableNativeFfnComposition::host_bound());HC next(b.get(),hc[0],hc[1],hc[2],hc[3],20*1024*1024,nullptr,dispatch);
  auto ordinary_hc=std::make_unique<HC>(b.get(),hc[0],hc[1],hc[2],hc[3],20*1024*1024,nullptr,dispatch);
  auto ordinary_core=std::make_unique<PortableNativeGdnCoreRows>(b.get(),gr,caps,PortableNativeGdnCoreRows::estimate(b.get(),caps),(root/"ordinary-core").string());
  ordinary_core->reset(std::vector<float>(786432,0.5f),std::vector<float>(30720,0.25f));
  PortableNativeOrdinaryGdnComposition ordinary(2,std::move(ordinary_hc),in,std::move(ordinary_core),out,PortableNativeOrdinaryGdnComposition::host_bound());
  check(ordinary.committed_state()==std::vector<float>(786432,0.f)&&ordinary.committed_history()==std::vector<float>(30720,0.f),"ordinary constructor must clear seeded core");
  PortableNativeFfnRows rows2(2,projection,d);auto hc2=std::make_unique<HC>(b.get(),hc[0],hc[1],hc[2],hc[3],20*1024*1024,nullptr,dispatch);
  auto ff2=PortableNativeFfnCompositionTestAccess::make(ordinary,std::move(hc2),rows2,PortableNativeFfnComposition::host_bound());
  // One connected layer 1 -> 2 -> 3 -> 4 replay, with issued FFN snapshots.
  {
   auto qc=std::unique_ptr<ggml_context,decltype(&ggml_free)>(ggml_init({16384,nullptr,true}),ggml_free);
   std::array<ggml_tensor*,3> q{ggml_new_tensor_2d(qc.get(),GGML_TYPE_Q8_0,2560,12288),ggml_new_tensor_2d(qc.get(),GGML_TYPE_Q8_0,2560,512),ggml_new_tensor_2d(qc.get(),GGML_TYPE_Q8_0,2560,512)};
   std::array<ggml_tensor*,2> ix{ggml_new_tensor_1d(qc.get(),GGML_TYPE_I32,128*2560/2),ggml_new_tensor_1d(qc.get(),GGML_TYPE_I32,512*2560/2)};
   auto* qw=ggml_new_tensor_2d(qc.get(),GGML_TYPE_Q8_0,6144,2560);
   std::array<ggml_tensor*,4> gamma{ggml_new_tensor_1d(qc.get(),GGML_TYPE_F32,256),ggml_new_tensor_1d(qc.get(),GGML_TYPE_F32,256),ggml_new_tensor_1d(qc.get(),GGML_TYPE_F32,128),ggml_new_tensor_1d(qc.get(),GGML_TYPE_F32,128)};
   auto qb=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>(ggml_backend_alloc_ctx_tensors(qc.get(),b.get()),ggml_backend_buffer_free);
   std::array<std::vector<uint8_t>,3> qbytes;for(size_t i=0;i<3;++i){qbytes[i].resize(ggml_nbytes(q[i]));ggml_backend_tensor_set(q[i],qbytes[i].data(),0,qbytes[i].size());}
   std::array<std::vector<uint8_t>,2> ibytes;for(size_t i=0;i<2;++i){ibytes[i].resize(ggml_nbytes(ix[i]));for(size_t r=0;r<(i?512u:128u);++r){ibytes[i][r*5120]=0x80;ibytes[i][r*5120+1]=0x3b;}ggml_backend_tensor_set(ix[i],ibytes[i].data(),0,ibytes[i].size());}
   std::vector<uint8_t> wbytes(ggml_nbytes(qw));ggml_backend_tensor_set(qw,wbytes.data(),0,wbytes.size());for(auto*t:gamma)put(t,std::vector<float>(ggml_nelements(t),1.f));
   constexpr uint32_t cap=8;auto qhc=std::make_unique<HC>(b.get(),hc[0],hc[1],hc[2],hc[3],20*1024*1024,nullptr,dispatch);
   auto qkv=std::make_unique<PortableNativeQsaQkvRows>(b.get(),q,std::array<const void*,3>{qbytes[0].data(),qbytes[1].data(),qbytes[2].data()},std::array<size_t,3>{qbytes[0].size(),qbytes[1].size(),qbytes[2].size()},(root/"chain-qkv").string(),PortableNativeQsaQkvRows::estimate(b.get()));
   auto ip=std::make_unique<PortableNativeQsaIndexerRows>(b.get(),ix,std::array<const void*,2>{ibytes[0].data(),ibytes[1].data()},std::array<size_t,2>{ibytes[0].size(),ibytes[1].size()},(root/"chain-ip").string(),PortableNativeQsaIndexerRows::estimate(b.get()));
   auto nr=std::make_unique<PortableNativeQsaNormRope>(b.get(),std::array<ggml_tensor*,3>{gamma[0],gamma[1],gamma[2]},(root/"chain-nr").string(),PortableNativeQsaNormRope::estimate(b.get()),false);
   auto idx=std::make_unique<PortableNativeQsaIndexer>(b.get(),gamma[3],cap,(root/"chain-idx").string(),PortableNativeQsaIndexer::estimate(b.get(),cap),false);
   auto sel=std::make_unique<PortableNativeQsaSelect>(b.get(),cap,(root/"chain-sel").string(),PortableNativeQsaSelect::estimate(b.get(),cap),false);
   auto att=std::make_unique<PortableNativeQsaAttention>(b.get(),cap,(root/"chain-att").string(),PortableNativeQsaAttention::estimate(b.get(),cap),false);
   auto ow=std::make_unique<PortableNativeQsaWoRows>(b.get(),qw,wbytes.data(),wbytes.size(),(root/"chain-wo").string(),PortableNativeQsaWoRows::estimate(b.get()));
   PortableNativeQsaComposition qowner(3,cap,PortableNativeQsaComposition::host_bound(cap),std::move(qhc),std::move(qkv),std::move(ip),std::move(nr),std::move(idx),std::move(sel),std::move(att),std::move(ow));
   PortableNativeFfnRows rows3(3,projection,d);auto h3=std::make_unique<HC>(b.get(),hc[0],hc[1],hc[2],hc[3],20*1024*1024,nullptr,dispatch);auto ff3=PortableNativeFfnCompositionTestAccess::make(qowner,std::move(h3),rows3,PortableNativeFfnComposition::host_bound());
   auto h4=std::make_unique<HC>(b.get(),hc[0],hc[1],hc[2],hc[3],20*1024*1024,nullptr,dispatch);auto c4=std::make_unique<PortableNativeGdnCoreRows>(b.get(),gr,caps,PortableNativeGdnCoreRows::estimate(b.get(),caps),(root/"chain-core4").string());PortableNativeOrdinaryGdnComposition g4(4,std::move(h4),in,std::move(c4),out,PortableNativeOrdinaryGdnComposition::host_bound());
   struct Result {std::vector<float> outputs;std::map<std::pair<int64_t,size_t>,std::pair<std::vector<int32_t>,std::vector<float>>> routes;std::vector<float> s1,h1,s2,h2,s4,h4;PortableNativeQsaComposition::Seed q;};
   connected_combine=true;distinct_stage=true;auto replay=[&](bool batched){Result result;std::array<size_t,4> positions{};PortableExpertRowsBatch provider=[&](int64_t layer,const auto& routes,auto& values,auto&){values.resize(routes.size()*10*2560);for(size_t r=0;r<routes.size();++r){auto pos=positions.at(layer)++;result.routes.emplace(std::make_pair(layer,pos),std::make_pair(routes[r].ids,routes[r].weights));std::fill_n(values.begin()+r*10*2560,10*2560,float(layer)+float(pos+1)/8);}return true;};
    auto step=[&](std::vector<int32_t> tokens){auto first=chain.position();auto input0=stage(tokens);PortableNativeFfnComposition::Batch a,batch,c;chain.run(input0,provider,[&](const auto&v){a=v;});ff2.run_ordinary(first,chain,a,provider,[&](const auto&v){batch=v;});ff3.run(first,ff2,batch,provider,[&](const auto&v){c=v;});g4.run(first,ff3,c,[&](const auto&v){auto x=v.output();result.outputs.insert(result.outputs.end(),x.begin(),x.end());});};
    if(batched){step({7,9});step({11});}else{step({7});step({9});step({11});}result.s1=owner.committed_state();result.h1=owner.committed_history();result.s2=ordinary.committed_state();result.h2=ordinary.committed_history();result.s4=g4.committed_state();result.h4=g4.committed_history();result.q=qowner.state();return result;};
   auto batch_result=replay(true);owner.reset({3,5});chain.reset();ordinary.reset();ff2.reset();qowner.reset();ff3.reset();g4.reset();auto serial_result=replay(false);
   check(batch_result.outputs==serial_result.outputs&&batch_result.routes==serial_result.routes&&batch_result.s1==serial_result.s1&&batch_result.h1==serial_result.h1&&batch_result.s2==serial_result.s2&&batch_result.h2==serial_result.h2&&batch_result.s4==serial_result.s4&&batch_result.h4==serial_result.h4&&batch_result.q.keys==serial_result.q.keys&&batch_result.q.values==serial_result.q.values&&batch_result.q.indexer==serial_result.q.indexer,"connected T2/T1 chain differs");
   check(batch_result.routes.at({1,0})!=batch_result.routes.at({1,1}),"connected distinct routes");check(batch_result.q.indexer[0]!=0&&batch_result.q.indexer[0]!=batch_result.q.indexer[128],"connected nonzero distinct indexer history");check(batch_result.outputs.size()==3*2560&&!std::equal(batch_result.outputs.begin(),batch_result.outputs.begin()+2560,batch_result.outputs.begin()+2560),"connected distinct row outputs");
   PortableExpertRowsBatch provider=[](int64_t l,const auto&r,auto&v,auto&){v.assign(r.size()*10*2560,float(l));return true;};
   auto reset_all=[&]{owner.reset({3,5});chain.reset();ordinary.reset();ff2.reset();qowner.reset();ff3.reset();g4.reset();};
   auto produce=[&](int token,PortableNativeFfnComposition& target){auto first=chain.position();auto h=stage({token});PortableNativeFfnComposition::Batch a,batch;chain.run(h,provider,[&](const auto&v){a=v;});target.run_ordinary(first,chain,a,provider,[&](const auto&v){batch=v;});return batch;};
   // Restore a genuine nonzero FFN-anchored seed and repeat its continuation.
   auto seed=qowner.state();auto next_snap=produce(13,ff2);std::vector<float> continuation,repeated;qowner.run(3,ff2,next_snap,[&](const auto&v){continuation=v.output();});auto continued=qowner.state();qowner.reset(seed);qowner.run(3,ff2,next_snap,[&](const auto&v){repeated=v.output();});auto restored=qowner.state();check(continuation==repeated&&continued.keys==restored.keys&&continued.values==restored.values&&continued.indexer==restored.indexer,"FFN anchored seed continuation");
   // After a committed prefix, downstream preparation fails after both ordinary
   // GDN and QSA have produced candidates. Earlier layers remain committed.
   reset_all();auto prefix_snap=produce(7,ff2);std::optional<PortableNativeQsaComposition::View> prior;qowner.run(0,ff2,prefix_snap,[&](const auto&v){prior=v;});auto prior_output=prior->output();ff3.reset();auto next=produce(9,ff2);auto os=ordinary.committed_state(),oh=ordinary.committed_history();auto qs=qowner.state();PortableNativeFfnComposition::Batch failed;bool caught=false;
   try{ff3.run(1,ff2,next,provider,[&](const auto&v){failed=v;},[](const auto&){throw std::runtime_error("late downstream prepare");});}catch(...){caught=true;}auto qa=qowner.state();check(caught&&ordinary.position()==2&&ordinary.committed_state()==os&&ordinary.committed_history()==oh&&qa.position==qs.position&&qa.keys==qs.keys&&qa.values==qs.values&&qa.indexer==qs.indexer&&prior->output()==prior_output,"downstream failure changed committed prefix");caught=false;try{(void)failed.at(0)->tuple();}catch(...){caught=true;}check(caught,"failed FFN candidate survived");reset_all();auto replayed=replay(false);check(replayed.outputs==batch_result.outputs,"coordinated reset replay");
   // Independently advanced ordinary owners keep both FFN histories live.
   reset_all();PortableNativeFfnRows foreign_rows(2,projection,d);auto foreign_hc=std::make_unique<HC>(b.get(),hc[0],hc[1],hc[2],hc[3],20*1024*1024,nullptr,dispatch);auto foreign_core=std::make_unique<PortableNativeGdnCoreRows>(b.get(),gr,caps,PortableNativeGdnCoreRows::estimate(b.get(),caps),(root/"foreign-chain-core").string());PortableNativeOrdinaryGdnComposition foreign_gdn(2,std::move(foreign_hc),in,std::move(foreign_core),out,PortableNativeOrdinaryGdnComposition::host_bound());auto foreign_ffhc=std::make_unique<HC>(b.get(),hc[0],hc[1],hc[2],hc[3],20*1024*1024,nullptr,dispatch);auto foreign=PortableNativeFfnCompositionTestAccess::make_unique(foreign_gdn,std::move(foreign_ffhc),foreign_rows,PortableNativeFfnComposition::host_bound());
   auto both=[&](int token){auto pos=chain.position();auto h=stage({token});PortableNativeFfnComposition::Batch a,local,other;chain.run(h,provider,[&](const auto&v){a=v;});ff2.run_ordinary(pos,chain,a,provider,[&](const auto&v){local=v;});foreign->run_ordinary(pos,chain,a,provider,[&](const auto&v){other=v;});return std::make_pair(local,other);};
   auto pair0=both(7);qowner.run(0,ff2,pair0.first,[](const auto&){});auto pair1=both(9);check(qowner.provenance_live()&&pair1.second[0]->valid_for(*foreign,3,1),"foreign test needs two live producers");auto before_calls=chain_qsa_qkv;auto before_foreign=qowner.state();caught=false;try{qowner.run(1,*foreign,pair1.second,[](const auto&){});}catch(...){caught=true;}auto after_foreign=qowner.state();check(caught&&!qowner.poisoned()&&chain_qsa_qkv==before_calls&&before_foreign.keys==after_foreign.keys&&before_foreign.indexer==after_foreign.indexer,"foreign FFN predecessor dispatched");qowner.run(1,ff2,pair1.first,[](const auto&){});
   // Build a new FFN-anchored prefix, then destroy that predecessor.
   reset_all();foreign.reset();auto fh=std::make_unique<HC>(b.get(),hc[0],hc[1],hc[2],hc[3],20*1024*1024,nullptr,dispatch);foreign=PortableNativeFfnCompositionTestAccess::make_unique(ordinary,std::move(fh),foreign_rows,PortableNativeFfnComposition::host_bound());auto fs=produce(7,*foreign);qowner.run(0,*foreign,fs,[](const auto&){});auto dead_seed=qowner.state();foreign.reset();caught=false;try{qowner.reset(dead_seed);}catch(...){caught=true;}check(caught&&!qowner.provenance_live(),"destroyed FFN seed accepted");ff2.reset();auto after_destroy=produce(9,ff2);before_calls=chain_qsa_qkv;caught=false;try{qowner.run(1,ff2,after_destroy,[](const auto&){});}catch(...){caught=true;}check(caught&&chain_qsa_qkv==before_calls&&!qowner.poisoned(),"destroyed FFN continuation dispatched");
   reset_all();auto live=produce(7,ff2);qowner.run(0,ff2,live,[](const auto&){});auto stale_seed=qowner.state();ff2.reset();caught=false;try{qowner.reset(stale_seed);}catch(...){caught=true;}check(caught,"reset FFN seed accepted");auto later=produce(9,ff2);before_calls=chain_qsa_qkv;caught=false;try{qowner.run(1,ff2,later,[](const auto&){});}catch(...){caught=true;}check(caught&&chain_qsa_qkv==before_calls&&!qowner.poisoned(),"reset FFN continuation dispatched");
   connected_combine=false;distinct_stage=false;
   owner.reset({3,5});chain.reset();ordinary.reset();ff2.reset();
  }
  size_t calls=0;PortableExpertRowsBatch experts=[&](int64_t l,const auto&r,auto&out,auto&){++calls;check((l==1||l==2)&&(r.size()==1||r.size()==2),"PLE FFN route");out.assign(r.size()*10*2560,0);return true;};
  PortableNativeFfnComposition::Batch snapshots;auto input=stage({7,9});chain.run(input,experts,[&](const auto&v){snapshots=v;});
  check(calls==1&&chain.position()==2&&snapshots.size()==2&&snapshots[0]->valid_for(chain,2,0),"PLE FFN T2 commit");auto residual=snapshots[0]->tuple().residual;
  next.run(2,0,chain,snapshots,[&](const HC::View&v,uint32_t n){check(n==2&&v.read(HC::View::Field::residual)[0]==residual[0]+1.f,"PLE FFN next HC");},[](const HC::View&,uint32_t){});
  bool reversed=false;try{ordinary.run(0,chain,{snapshots[1],snapshots[0]},[](const auto&){});}catch(...){reversed=true;}check(reversed&&ordinary.position()==0&&!ordinary.poisoned(),"ordinary GDN reversed rows");
  PortableNativeFfnComposition::Batch second;ff2.run_ordinary(0,chain,snapshots,experts,[&](const auto&v){second=v;});check(second.size()==2&&second[0]->valid_for(ff2,3,0)&&ordinary.position()==2,"ordinary GDN FFN T2");
  auto tail_input=stage({11});PortableNativeFfnComposition::Batch tail;chain.run(tail_input,experts,[&](const auto&v){tail=v;});check(calls==3&&chain.position()==3&&tail[0]->valid_for(chain,2,2)&&snapshots[0]->tuple().residual==residual,"PLE FFN tail and retained tuple");
  PortableNativeFfnComposition::Batch second_tail;ff2.run_ordinary(2,chain,tail,experts,[&](const auto&v){second_tail=v;});check(calls==4&&ordinary.position()==3&&second_tail[0]->valid_for(ff2,3,2),"ordinary GDN FFN tail");
  auto ordinary_seed=ordinary.state();check(ordinary_seed.position==3,"ordinary GDN seed position");
  std::vector<std::vector<float>> batch_blocks{second[0]->tuple().block,second[1]->tuple().block,second_tail[0]->tuple().block};
  auto state_before=owner.committed_state(),history_before=owner.committed_history(),ple_before=ple.history();auto bad_input=stage({13});PortableNativeFfnComposition::Batch failed;bool rejected=false;
  try{chain.run(bad_input,experts,[&](const auto&v){failed=v;},[](const auto&){throw std::runtime_error("late prepare");});}catch(...){rejected=true;}
  check(rejected&&chain.poisoned()&&chain.position()==3&&owner.position()==3&&owner.committed_state()==state_before&&owner.committed_history()==history_before&&ple.history()==ple_before&&snapshots[0]->tuple().residual==residual,"PLE FFN preparation rollback");
  rejected=false;try{failed[0]->tuple();}catch(...){rejected=true;}check(rejected,"PLE FFN failed candidate retained");
  owner.reset({3,5});check(!second[0]->valid_for(ff2,3,0),"ordinary transitive invalidation");bool seed_stale=false;try{ordinary.reset(ordinary_seed);}catch(...){seed_stale=true;}check(seed_stale,"ordinary stale seed accepted");chain.reset();bool stale=false;try{snapshots[0]->tuple();}catch(...){stale=true;}check(stale,"PLE FFN reset provenance");
  ordinary.reset();ff2.reset();
  for(size_t i=0;i<3;++i){auto h=stage({int32_t(7+2*i)});PortableNativeFfnComposition::Batch a,batch;chain.run(h,experts,[&](const auto& v){a=v;});ff2.run_ordinary(i,chain,a,experts,[&](const auto& v){batch=v;});check(batch[0]->tuple().block==batch_blocks[i],"ordinary T2 versus serial FFN output");}
  check(ordinary.committed_state()==ordinary_seed.state&&ordinary.committed_history()==ordinary_seed.history,"ordinary T2 versus serial state/history");
  owner.reset({3,5});chain.reset();ordinary.reset();ff2.reset();
 }
 in_quant=in_qkv=in_z=conv_calls=ab_calls=recur_calls=out_quant=out_dot=0;
 std::optional<Owner::View> old;auto h=stage({7,9});auto expected_residual=h.residual();owner.run(h,[&](const Owner::View&view){check(view.position()==0&&view.tokens()==std::vector<int32_t>{7,9}&&view.residual()==expected_residual&&view.injection()==std::vector<float>(8,3),"row provenance");check(view.output().size()==5120,"output rows");bool reset_rejected=false,abort_rejected=false,owner_reset_rejected=false;try{ple.reset();}catch(...){reset_rejected=true;}try{ple.abort(h);}catch(...){abort_rejected=true;}try{owner.reset();}catch(...){owner_reset_rejected=true;}check(reset_rejected&&abort_rejected&&owner_reset_rejected,"reentrant mutation accepted");old=view;});
 check(owner.position()==2&&ple.position()==2&&old&&old->tokens()==std::vector<int32_t>{7,9}&&in_quant==1&&in_qkv==1&&in_z==1&&conv_calls==1&&ab_calls==1&&recur_calls==1&&out_quant==1&&out_dot==1,"coordinated T2 commit and dispatch counts");auto before_state=owner.committed_state(),before_history=owner.committed_history(),before_ple=ple.history();
 auto tail=stage({11});std::optional<Owner::View> failed;bool rejected=false;try{owner.run(tail,[&](const Owner::View&view){failed=view;throw std::runtime_error("consumer failure");});}catch(const std::exception&){rejected=true;}
 check(rejected&&owner.poisoned()&&owner.position()==2&&ple.position()==2&&ple.history()==before_ple&&owner.committed_state()==before_state&&owner.committed_history()==before_history&&old->tokens()==std::vector<int32_t>{7,9},"failed consumer changed commit");
 rejected=false;try{failed->output();}catch(...){rejected=true;}check(rejected,"failed view valid");owner.reset({3,5});rejected=false;try{old->output();}catch(...){rejected=true;}check(rejected&&owner.position()==0&&ple.position()==0,"reset invalidation");
 std::vector<float> seeded_state(786432,0.25f),seeded_history(30720,0.5f);
 auto run=[&](bool batch,float seed){std::fill(seeded_state.begin(),seeded_state.end(),seed);owner.reset({3,5},0,seeded_state,seeded_history);std::vector<float>all;auto step=[&](std::vector<int32_t>ts){auto h0=stage(ts);owner.run(h0,[&](const Owner::View&view){auto v0=view.output();all.insert(all.end(),v0.begin(),v0.end());});};if(batch){step({7,9});step({11});}else{step({7});step({9});step({11});}return all;};
 auto batched=run(true,0.25f),serial=run(false,0.25f);check(batched==serial&&owner.position()==3&&ple.position()==3,"seeded T2 tail vs serial");
 auto other=run(true,-0.5f),again=run(true,0.25f);check(other!=batched&&again==batched,"seeded reset A/B/A");
 for(const std::string&point:{"hc norm written","hc mix written","gdn input quant","gdn qkv","gdn z","core conv","core ab","core recurrence","gdn output quant","gdn output dot","second scale","consumer","publication preparation"}){
  owner.reset({3,5});auto first=stage({7});std::optional<Owner::View> prior;owner.run(first,[&](const Owner::View&view){prior=view;});
  auto state=owner.committed_state(),history=owner.committed_history(),ph=ple.history();auto candidate=stage({9,11});std::optional<Owner::View>attempt;
  fail_at=point;fail=point=="core conv"?4:point=="core ab"?5:point=="core recurrence"?6:0;gdn_bad_second_scale=point=="second scale";bool caught=false;
  try{owner.run(candidate,[&](const Owner::View&view){attempt=view;if(point=="consumer")throw std::runtime_error("consumer");},[&](const Owner::View&){if(point=="publication preparation")throw std::runtime_error("prepare");});}catch(const std::exception&){caught=true;}
  fail_at.clear();fail=0;gdn_bad_second_scale=false;
  check(caught&&owner.poisoned()&&owner.position()==1&&ple.position()==1&&ple.history()==ph&&owner.committed_state()==state&&owner.committed_history()==history&&prior->tokens()==std::vector<int32_t>{7},"failed component changed publication/state");
  if(attempt){bool dead=false;try{attempt->output();}catch(...){dead=true;}check(dead,"failed candidate view survived");}
 }
 owner.reset({3,5});auto stale=stage({7,9});owner.reset({3,5});bool stale_rejected=false;try{owner.run(stale,[](const Owner::View&){});}catch(...){stale_rejected=true;}check(stale_rejected&&owner.position()==0,"reset handle accepted");
 auto foreign_roles=roles;foreign_roles.directory=(root/"foreign").string();auto foreign_prefix=(root/"foreign-hc").string();auto foreign_bounds=PortableNativePleAdmission::calculate(b.get(),hc,foreign_roles.directory,foreign_prefix.size(),hc_caps);PortableNativePleComposition foreign(foreign_roles,hc,b.get(),hc_caps,foreign_bounds.total,foreign_prefix);foreign.reset({3,5});
 std::vector<std::array<uint32_t,16>>fi;check(table->gather_batch({7},{{3,5}},fi,expected_embedding,error),"foreign gather");norm_call=0;auto foreign_handle=foreign.stage(0,{7},std::vector<float>(10240,0.25f));bool foreign_rejected=false;try{owner.run(foreign_handle,[](const Owner::View&){});}catch(...){foreign_rejected=true;}check(foreign_rejected&&owner.position()==0&&ple.position()==0&&foreign.position()==0,"foreign handle accepted");
 owner.reset({3,5});foreign.reset({3,5},1);norm_call=0;auto reordered=foreign.stage(1,{7},std::vector<float>(10240,0.25f));bool reordered_rejected=false;try{owner.run(reordered,[](const Owner::View&){});}catch(...){reordered_rejected=true;}check(reordered_rejected&&owner.position()==0,"wrong position accepted");
 std::cout<<"PLE HC GDN concrete owners and consumer mock passed\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
