// Actual PLE/HC composition and GDN owners, with low-level component mocks.
#define STRATA_CHAIN_QSA_MOCKS 1
#include "../kernels/native_qsa_norm_rope_fixture.hpp"
#include "../kernels/native_qsa_indexer_fixture.hpp"
#include "../kernels/native_qsa_select_fixture.hpp"
#include "../kernels/native_qsa_attention_fixture.hpp"
#define STRATA_CONNECTED_LAYER0_TEST 1
#define main prior_ple_integration_main
#include "portable_native_ple_integration_test.cpp"
#undef main
#include "portable_native_ple_gdn_composition.hpp"
#include "portable_native_ffn_test_access.hpp"
#include "portable_native_ffn_hc_factory.hpp"
#include <map>
namespace {
void need(bool b,const char*m){if(!b)throw std::runtime_error(m);}
int fail=0;size_t conv_calls=0,ab_calls=0,recur_calls=0;
std::vector<float> get(const ggml_tensor*t){std::vector<float>v(ggml_nbytes(t)/4);ggml_backend_tensor_get(t,v.data(),0,v.size()*4);return v;}
void put(ggml_tensor*t,const std::vector<float>&v){need(ggml_nbytes(t)==v.size()*4,"mock span");ggml_backend_tensor_set(t,v.data(),0,v.size()*4);}
}
 #include "portable_native_chain_qsa_mocks.hpp"
namespace {std::vector<float> chain_qsa_input,chain_qsa_gated; size_t chain_qsa_qkv=0;
#ifdef STRATA_MODEL_ROWS_FIXTURE
size_t model_qsa_t2=0,model_gdn_t2=0;
// Synthetic perturbation for model plumbing sensitivity, not a numerical oracle.
float model_gdn_signal=1.f; bool model_gdn_saturated=false,model_qsa_saturated=false;
size_t family_alpha_seen=0,family_input_seen=0,family_output_seen=0,family_conv_live=0;
bool family_f16_capture=false;std::vector<float> family_f16_mixed,family_f16_qkv,family_f16_z,family_f16_y;
bool family_ab_capture=false;size_t family_ab_calls=0;std::vector<float> family_ab_mixed,family_ab_alpha,family_ab_beta,family_ab_recurrence_beta;
#endif
}
namespace strata::core::portable_detail {
StrataVkQ8Wo* strata_vk_qsa_qkv_rows_create(ggml_backend_t,const char*p,size_t){std::ofstream(p)<<"mock";return new StrataVkQ8Wo;}
StrataVkQ8Wo* strata_vk_qsa_wo_rows_create(ggml_backend_t,const char*p,size_t){std::ofstream(p)<<"mock";return new StrataVkQ8Wo;}
StrataVkNativeShared* strata_vk_qsa_indexer_bf16_rows_create(ggml_backend_t,const char*p,size_t){std::ofstream(p)<<"mock";return new StrataVkNativeShared;}
void strata_vk_qsa_input_quantize_bounded(StrataVkNativeStages*,const ggml_tensor*x,ggml_tensor*q,size_t rows){chain_qsa_input=get(x);
#ifdef STRATA_MODEL_ROWS_FIXTURE
 for(size_t r=0;r<rows;++r)model_qsa_saturated|=std::fabs(std::tanh(chain_qsa_input[r*2560]/1024.f))==1.f;
#endif
 std::vector<uint8_t> packed(rows*2880);ggml_backend_tensor_set(q,packed.data(),0,packed.size());}
void strata_vk_qsa_qkv_run_bounded(StrataVkQ8Wo*,const ggml_tensor*weight,const ggml_tensor*,ggml_tensor*y,size_t n,size_t rows){
#ifdef STRATA_MODEL_ROWS_FIXTURE
 if(rows==2)++model_qsa_t2;
#endif
 size_t role=chain_qsa_qkv++%3;
#ifdef STRATA_MODEL_ROWS_FIXTURE
 uint8_t marker[3]{};ggml_backend_tensor_get(weight,marker,0,3);if(marker[1]){check(marker[1]==0x3c&&marker[2]==41+role,"owned QSA QKV role");++owned_qkv_seen;}
#endif
 std::vector<float> v(rows*n);for(size_t r=0;r<rows;++r)for(size_t i=0;i<n;++i)v[r*n+i]=
#ifdef STRATA_MODEL_ROWS_FIXTURE
 std::tanh(chain_qsa_input[r*2560]/1024.f)*.125f
#else
 chain_qsa_input[r*2560]
#endif
 +float(int((i+role*3)%17)-8)/32.f;put(y,v);}
void strata_vk_qsa_wo_quantize_bounded(StrataVkNativeStages*,const ggml_tensor*x,ggml_tensor*q,size_t rows){chain_qsa_gated=get(x);std::vector<uint8_t> packed(rows*6912);ggml_backend_tensor_set(q,packed.data(),0,packed.size());}
void strata_vk_qsa_wo_run_bounded(StrataVkQ8Wo*,const ggml_tensor*weight,const ggml_tensor*,ggml_tensor*y,size_t rows){
#ifdef STRATA_MODEL_ROWS_FIXTURE
 uint8_t marker[3]{};ggml_backend_tensor_get(weight,marker,0,3);if(marker[1]){check(marker[1]==0x3c&&marker[2]==44,"owned QSA output role");++owned_wo_seen;}
#endif
 std::vector<float> v(rows*2560);for(size_t r=0;r<rows;++r)std::copy_n(chain_qsa_gated.begin()+r*6144,2560,v.begin()+r*2560);put(y,v);}
struct StrataVkNativeGdnConvRows{};struct StrataVkNativeGdnAbRows{};struct StrataVkNativeGdnRows{};
StrataVkNativeGdnConvRows* strata_vk_native_gdn_conv_rows_create(ggml_backend_t,const char*p,size_t){std::ofstream(p)<<"mock";if(fail==1)throw std::runtime_error("conv factory");
#ifdef STRATA_MODEL_ROWS_FIXTURE
 ++family_conv_live;
#endif
 return new StrataVkNativeGdnConvRows;}
void strata_vk_native_gdn_conv_rows_destroy(StrataVkNativeGdnConvRows*p)noexcept{
#ifdef STRATA_MODEL_ROWS_FIXTURE
 if(p)--family_conv_live;
#endif
 delete p;}size_t strata_vk_native_gdn_conv_rows_bytes(const StrataVkNativeGdnConvRows*){return 512;}
StrataVkNativeGdnAbRows* strata_vk_native_gdn_ab_rows_create(ggml_backend_t,const char*p,size_t){std::ofstream(p)<<"mock";if(fail==2)throw std::runtime_error("AB factory");return new StrataVkNativeGdnAbRows;}
StrataVkNativeGdnAbRows* strata_vk_native_gdn_gate_products_create(ggml_backend_t b,const char*p,size_t n){return strata_vk_native_gdn_ab_rows_create(b,p,n);}
uint32_t strata_vk_native_gdn_gate_products_run(StrataVkNativeGdnAbRows*,ggml_tensor*const t[8],uint32_t n){
#ifdef STRATA_MODEL_ROWS_FIXTURE
 check(family_ab_capture,"unexpected product gate");family_ab_mixed=get(t[0]);family_ab_alpha=get(t[1]);family_ab_beta=get(t[2]);++family_ab_calls;check(family_ab_alpha.size()==n*48&&family_ab_beta.size()==n*48,"product spans");put(t[5],family_ab_alpha);put(t[6],family_ab_beta);return fail==5?1:0;
#else
 (void)t;(void)n;throw std::runtime_error("unexpected product gate");
#endif
}
void strata_vk_native_gdn_ab_rows_destroy(StrataVkNativeGdnAbRows*p)noexcept{delete p;}size_t strata_vk_native_gdn_ab_rows_bytes(const StrataVkNativeGdnAbRows*){return 512;}
StrataVkNativeGdnRows* strata_vk_native_gdn_rows_create(ggml_backend_t,const char*p,size_t){std::ofstream(p)<<"mock";if(fail==3)throw std::runtime_error("recurrence factory");return new StrataVkNativeGdnRows;}
void strata_vk_native_gdn_rows_destroy(StrataVkNativeGdnRows*p)noexcept{delete p;}size_t strata_vk_native_gdn_rows_bytes(const StrataVkNativeGdnRows*){return 512;}
void strata_vk_native_gdn_conv_rows_run(StrataVkNativeGdnConvRows*,ggml_tensor*const t[10],uint32_t n){++conv_calls;
#ifdef STRATA_MODEL_ROWS_FIXTURE
 float role=0;ggml_backend_tensor_get(t[2],&role,0,4);if(role)check(role==3.f,"family convolution role");
#endif
 auto h=get(t[0]),q=get(t[1]);
#ifdef STRATA_MODEL_ROWS_FIXTURE
 if(family_f16_capture)family_f16_qkv=q;
#endif
 std::vector<float>out(n*10240);for(size_t r=0;r<n;++r)for(size_t c=0;c<10240;++c){out[r*10240+c]=q[r*10240+c]+h[c*3];h[c*3]=h[c*3+1];h[c*3+1]=h[c*3+2];h[c*3+2]=q[r*10240+c];}put(t[3],h);put(t[4],out);if(fail==4)throw std::runtime_error("conv wrote then threw");}
uint32_t strata_vk_native_gdn_ab_rows_run(StrataVkNativeGdnAbRows*,ggml_tensor*const t[8],uint32_t n){++ab_calls;
#ifdef STRATA_MODEL_ROWS_FIXTURE
 uint16_t alpha_bits=0,beta_bits=0;ggml_backend_tensor_get(t[1],&alpha_bits,0,2);ggml_backend_tensor_get(t[2],&beta_bits,0,2);if(alpha_bits){float dt=0,a=0;ggml_backend_tensor_get(t[3],&dt,0,4);ggml_backend_tensor_get(t[4],&a,0,4);check(alpha_bits==0x3f80&&beta_bits==0x4000&&dt==4.f&&a==5.f,"family alpha/beta/dt/A roles");++family_alpha_seen;}
#endif
 auto mixed=get(t[0]);
#ifdef STRATA_MODEL_ROWS_FIXTURE
 if(family_f16_capture)family_f16_mixed=mixed;
#endif
 std::vector<float>gate(n*48),beta(n*48);for(size_t r=0;r<n;++r)for(size_t i=0;i<48;++i){gate[r*48+i]=mixed[r*2560];beta[r*48+i]=1;}put(t[5],gate);put(t[6],beta);return fail==5?1:0;}
void strata_vk_native_gdn_rows_run(StrataVkNativeGdnRows*,ggml_tensor*const t[13],ggml_tensor*,uint32_t n){++recur_calls;
#ifdef STRATA_MODEL_ROWS_FIXTURE
 float gamma=0;ggml_backend_tensor_get(t[5],&gamma,0,4);if(gamma)check(gamma==6.f,"family gamma role");
#endif
 auto state=get(t[0]),h=get(t[1]),gate=get(t[2]),z=get(t[4]);std::vector<float>y(n*6144);for(size_t r=0;r<n;++r)for(size_t i=0;i<6144;++i){state[i]+=h[r*10240+i]+gate[r*48]+z[r*6144+i];y[r*6144+i]=state[i];}
#ifdef STRATA_MODEL_ROWS_FIXTURE
 if(family_f16_capture){family_f16_z=z;family_f16_y=y;}
 if(family_ab_capture)family_ab_recurrence_beta=get(t[3]);
#endif
 put(t[6],state);put(t[7],y);if(fail==6)throw std::runtime_error("recurrence wrote then threw");}
}

namespace { std::vector<float> gdn_quant_input; bool gdn_bad_second_scale=false; size_t in_quant=0,in_qkv=0,in_z=0,out_quant=0,out_dot=0; }
namespace strata::core::portable_detail {
StrataVkQ8Wo* strata_vk_gdn_input_create(ggml_backend_t,const char*p,size_t){std::ofstream(p)<<"mock";inject("gdn input factory");return new StrataVkQ8Wo;}
StrataVkQ8Wo* strata_vk_gdn_output_rows_create(ggml_backend_t,const char*p,size_t){std::ofstream(p)<<"mock";inject("gdn output factory");return new StrataVkQ8Wo;}
void strata_vk_gdn_input_quantize_bounded(StrataVkNativeStages*,const ggml_tensor*x,ggml_tensor*q,size_t rows){
#ifdef STRATA_MODEL_ROWS_FIXTURE
 if(rows==2)++model_gdn_t2;
#endif
 ++in_quant;gdn_quant_input=get(x);std::vector<uint8_t>p(rows*2880);if(gdn_bad_second_scale&&rows==2)p[2880]=1;ggml_backend_tensor_set(q,p.data(),0,p.size());write_then_throw("gdn input quant",{q},9);}
void strata_vk_gdn_input_run_bounded(StrataVkQ8Wo*,const ggml_tensor* weight,const ggml_tensor*,ggml_tensor*y,size_t columns,size_t rows){
#ifdef STRATA_MODEL_ROWS_FIXTURE
 uint8_t marker[3];ggml_backend_tensor_get(weight,marker,0,3);if(marker[1]){check(marker[1]==0x3c&&marker[2]==(columns==10240?11:12),"family input role identity");++family_input_seen;}
#endif
 if(columns==10240)++in_qkv;else ++in_z;std::vector<float>v(rows*columns);for(size_t r=0;r<rows;++r)for(size_t i=0;i<columns;++i)v[r*columns+i]=gdn_quant_input[r*2560]+float(i%7)/16;put(y,v);write_then_throw(columns==10240?"gdn qkv":"gdn z",{y},9);}
void strata_vk_gdn_output_quantize_bounded(StrataVkNativeStages*,const ggml_tensor*x,ggml_tensor*q,size_t rows){++out_quant;gdn_quant_input=get(x);
#ifdef STRATA_MODEL_ROWS_FIXTURE
 for(size_t r=0;r<rows;++r)model_gdn_saturated|=std::fabs(std::tanh(gdn_quant_input[r*6144]/1024.f))==1.f;
#endif
 std::vector<uint8_t>p(rows*6912);ggml_backend_tensor_set(q,p.data(),0,p.size());write_then_throw("gdn output quant",{q},9);}
void strata_vk_gdn_output_run_bounded(StrataVkQ8Wo*,const ggml_tensor* weight,const ggml_tensor*,ggml_tensor*y,size_t rows){
#ifdef STRATA_MODEL_ROWS_FIXTURE
 uint8_t marker[3];ggml_backend_tensor_get(weight,marker,0,3);if(marker[1]){check(marker[1]==0x3c&&marker[2]==13,"family output role identity");++family_output_seen;}
#endif
 ++out_dot;std::vector<float>v(rows*2560);for(size_t r=0;r<rows;++r)for(size_t i=0;i<2560;++i)v[r*2560+i]=
#ifdef STRATA_MODEL_ROWS_FIXTURE
 std::tanh(gdn_quant_input[r*6144+i]/1024.f)*(.125f*model_gdn_signal);
#else
 gdn_quant_input[r*6144+i];
#endif
 put(y,v);write_then_throw("gdn output dot",{y},9);}
}

#ifdef STRATA_TERMINAL_FIXTURE
int prior_chain_main(){try{
#else
int main(){try{
#endif
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
   // Checked layer-zero roles and a real Q8_0 token table, under CPU component mocks.
   std::vector<float> hc_norm(10240,1.f);std::vector<uint8_t> hc_inject(40960/32*34),hc_down(3276800/32*34),hc_up(3276800/32*34);
   auto named_hc=[&](bool attention){std::string p=attention?"blk.0.hc_attn_":"blk.0.hc_ffn_";return std::vector<PortableTensorView>{{p+"norm.weight",{10240},0,hc_norm.data(),hc_norm.size()*4},{p+"inject.weight",{10240,4},8,hc_inject.data(),hc_inject.size()},{p+"down.weight",{10240,320},8,hc_down.data(),hc_down.size()},{p+"up.weight",{320,10240},8,hc_up.data(),hc_up.size()}};};
   auto a0=NativeFfnHcFactory::create(0,b.get(),named_hc(true),64u<<20,SIZE_MAX,nullptr,dispatch,std::nullopt,true);
   auto f0=NativeFfnHcFactory::create(0,b.get(),named_hc(false),64u<<20,SIZE_MAX,nullptr,dispatch);
   std::vector<uint8_t> embeddings(16*80*34);for(size_t t=0;t<16;++t)for(size_t block=0;block<80;++block){auto* p=embeddings.data()+(t*80+block)*34;p[0]=0;p[1]=0x3c;std::fill_n(p+2,32,uint8_t(t+1));}
   PortableTensorView embedding{"token_embd.weight",{2560,16},8,embeddings.data(),embeddings.size()};
   PortableNativeGdnInputRows in0(b.get(),{qi,zi},{qr.data(),zr.data()},{qr.size(),zr.size()},(root/"layer0-in").string(),PortableNativeGdnInputRows::estimate(b.get()));
   auto core0=std::make_unique<PortableNativeGdnCoreRows>(b.get(),gr,caps,PortableNativeGdnCoreRows::estimate(b.get(),caps),(root/"layer0-core").string());
   PortableNativeGdnOutputRows out0(b.get(),wo,wr.data(),wr.size(),(root/"layer0-out").string(),PortableNativeGdnOutputRows::estimate(b.get()));
   PortableNativeFfnRows rows0(0,projection,d);auto writes_before=layer0_write_calls;
   PortableNativeLayer0Composition layer0(embedding,std::move(a0),in0,std::move(core0),out0,std::move(f0),rows0,b.get(),8u<<20,PortableNativeLayer0Composition::host_bound(),(root/"layer0-write").string().c_str());
   check(layer0_write_owners==1,"layer0 distinct final write owner");
   struct Result {std::vector<float> outputs,layer0_outputs;std::map<std::pair<int64_t,size_t>,std::pair<std::vector<int32_t>,std::vector<float>>> routes;std::vector<float> s0,h0,ple_history,s1,h1,s2,h2,s4,h4;std::array<int32_t,2> previous{};PortableNativeQsaComposition::Seed q;};
   connected_combine=true;distinct_stage=true;auto replay=[&](bool batched,bool typed=false){Result result;std::array<size_t,4> positions{};PortableExpertRowsBatch provider=[&](int64_t layer,const auto& routes,auto& values,auto&){values.resize(routes.size()*10*2560);for(size_t r=0;r<routes.size();++r){auto pos=positions.at(layer)++;result.routes.emplace(std::make_pair(layer,pos),std::make_pair(routes[r].ids,routes[r].weights));std::fill_n(values.begin()+r*10*2560,10*2560,float(layer)+float(pos+1)/8);}return true;};
    auto step=[&](std::vector<int32_t> tokens){auto first=chain.position();auto input0=[&]{if(!typed)return stage(tokens);auto source=layer0.run(first,tokens,provider);auto residual=source->residual();result.layer0_outputs.insert(result.layer0_outputs.end(),residual.begin(),residual.end());std::vector<std::array<uint32_t,16>> indices;auto prev=ple.previous();std::vector<std::array<int32_t,2>> pred;for(auto t:tokens){pred.push_back(prev);prev={prev[1],t};}check(table->gather_batch(tokens,pred,indices,expected_embedding,error),"typed gather");norm_call=0;auto h=ple.stage(source);check(layer0_ple_residual_seen==residual,"typed PLE residual identity");return h;}();PortableNativeFfnComposition::Batch a,batch,c;chain.run(input0,provider,[&](const auto&v){a=v;});ff2.run_ordinary(first,chain,a,provider,[&](const auto&v){batch=v;});ff3.run(first,ff2,batch,provider,[&](const auto&v){c=v;});g4.run(first,ff3,c,[&](const auto&v){auto x=v.output();result.outputs.insert(result.outputs.end(),x.begin(),x.end());});};
    if(batched){step({7,9});step({11});}else{step({7});step({9});step({11});}if(typed){result.s0=layer0.committed_state();result.h0=layer0.committed_history();result.previous=ple.previous();result.ple_history=ple.history();}result.s1=owner.committed_state();result.h1=owner.committed_history();result.s2=ordinary.committed_state();result.h2=ordinary.committed_history();result.s4=g4.committed_state();result.h4=g4.committed_history();result.q=qowner.state();return result;};
   owner.reset();chain.reset();ordinary.reset();ff2.reset();qowner.reset();ff3.reset();g4.reset();auto typed_batch=replay(true,true);check(layer0_write_calls-writes_before==2,"T2+T1 final write calls");
   layer0.reset();owner.reset();chain.reset();ordinary.reset();ff2.reset();qowner.reset();ff3.reset();g4.reset();auto typed_serial=replay(false,true);check(layer0_write_calls-writes_before==5,"3T1 final write calls");
   check(typed_batch.layer0_outputs==typed_serial.layer0_outputs&&typed_batch.outputs==typed_serial.outputs&&typed_batch.routes==typed_serial.routes&&typed_batch.s0==typed_serial.s0&&typed_batch.h0==typed_serial.h0&&typed_batch.previous==typed_serial.previous&&typed_batch.ple_history==typed_serial.ple_history&&typed_batch.s1==typed_serial.s1&&typed_batch.h1==typed_serial.h1&&typed_batch.s2==typed_serial.s2&&typed_batch.h2==typed_serial.h2&&typed_batch.s4==typed_serial.s4&&typed_batch.h4==typed_serial.h4&&typed_batch.q.keys==typed_serial.q.keys&&typed_batch.q.values==typed_serial.q.values&&typed_batch.q.indexer==typed_serial.q.indexer,"typed connected T2/T1 chain differs");
   check(typed_batch.previous==std::array<int32_t,2>{9,11}&&typed_batch.routes.count({0,0})&&typed_batch.routes.at({0,0})!=typed_batch.routes.at({0,1}),"typed distinct token routes/history");
   check(typed_batch.layer0_outputs.size()==3*10240&&typed_batch.layer0_outputs[0]!=0.f&&!std::equal(typed_batch.layer0_outputs.begin(),typed_batch.layer0_outputs.begin()+10240,typed_batch.layer0_outputs.begin()+10240)&&typed_batch.outputs.size()==3*2560,"typed nonzero distinct materialized rows");
   // Typed provenance and failure cases use real issued endpoints, never fabricated rows.
   auto reset_typed=[&]{layer0.reset();owner.reset();chain.reset();ordinary.reset();ff2.reset();qowner.reset();ff3.reset();g4.reset();};
   PortableExpertRowsBatch typed_provider=[](int64_t l,const auto&r,auto&v,auto&){v.assign(r.size()*10*2560,float(l+1)/8);return true;};
   auto typed_stage=[&](const std::shared_ptr<const PortableNativeLayer0Rows>& source){auto ts=source->tokens();auto prev=ple.previous();std::vector<std::array<int32_t,2>> pred;for(auto t:ts){pred.push_back(prev);prev={prev[1],t};}std::vector<std::array<uint32_t,16>> indices;check(table->gather_batch(ts,pred,indices,expected_embedding,error),"typed regression gather");norm_call=0;return ple.stage(source);};
   reset_typed();auto source0=layer0.run(0,{7},typed_provider);auto h0=typed_stage(source0);PortableNativeFfnComposition::Batch snap1,snap2;chain.run(h0,typed_provider,[&](const auto& v){snap1=v;});ff2.run_ordinary(0,chain,snap1,typed_provider,[&](const auto&v){snap2=v;});qowner.run(0,ff2,snap2,[](const auto&){});auto saved_seed=qowner.state();auto dispatch_before=in_quant;layer0.reset();
   bool rejected=false;try{(void)source0->residual();}catch(...){rejected=true;}check(rejected&&!snap1[0]->valid_for(chain,2,0)&&!snap2[0]->valid_for(ff2,3,0),"layer0 reset did not invalidate typed descendants");rejected=false;try{qowner.reset(saved_seed);}catch(...){rejected=true;}check(rejected&&!qowner.provenance_live(),"layer0 stale seed accepted");rejected=false;try{ff2.run_ordinary(1,chain,snap1,typed_provider,[](const auto&){});}catch(...){rejected=true;}check(rejected&&in_quant==dispatch_before,"stale downstream dispatched");
   reset_typed();auto after_reset=replay(false,true);check(after_reset.outputs==typed_serial.outputs,"typed coordinated reset replay");
   // A second real producer supplies foreign and wrong-position endpoints, then dies.
   reset_typed();auto first=layer0.run(0,{7},typed_provider);auto first_handle=typed_stage(first);chain.run(first_handle,typed_provider,[](const auto&){});
   PortableNativeGdnInputRows foreign0in(b.get(),{qi,zi},{qr.data(),zr.data()},{qr.size(),zr.size()},(root/"foreign0-in").string(),PortableNativeGdnInputRows::estimate(b.get()));PortableNativeGdnOutputRows foreign0out(b.get(),wo,wr.data(),wr.size(),(root/"foreign0-out").string(),PortableNativeGdnOutputRows::estimate(b.get()));
   PortableNativeFfnRows foreign0rows(0,projection,d);auto foreign0=std::make_unique<PortableNativeLayer0Composition>(embedding,NativeFfnHcFactory::create(0,b.get(),named_hc(true),64u<<20,SIZE_MAX,nullptr,dispatch,std::nullopt,true),foreign0in,std::make_unique<PortableNativeGdnCoreRows>(b.get(),gr,caps,PortableNativeGdnCoreRows::estimate(b.get(),caps),(root/"foreign0-core").string()),foreign0out,NativeFfnHcFactory::create(0,b.get(),named_hc(false),64u<<20,SIZE_MAX,nullptr,dispatch),foreign0rows,b.get(),8u<<20,PortableNativeLayer0Composition::host_bound(),(root/"foreign0-write").string().c_str());
   auto wrong=foreign0->run(0,{7},typed_provider);auto foreign_next=foreign0->run(1,{9},typed_provider);auto ple_work=quant;rejected=false;try{(void)ple.stage(wrong);}catch(...){rejected=true;}check(rejected&&quant==ple_work,"wrong position dispatched PLE");rejected=false;try{(void)ple.stage(foreign_next);}catch(...){rejected=true;}check(rejected&&quant==ple_work,"foreign endpoint dispatched PLE");foreign0.reset();check(!wrong->live()&&!foreign_next->live(),"destroyed layer0 endpoints live");rejected=false;try{(void)ple.stage(foreign_next);}catch(...){rejected=true;}check(rejected&&quant==ple_work,"destroyed endpoint dispatched PLE");
   // Layer0 failures cannot publish a candidate or disturb the committed prefix.
   for(bool fail_write:{false,true}){reset_typed();auto prior0=layer0.run(0,{7},typed_provider);auto saved=prior0->residual();auto state=layer0.committed_state(),history=layer0.committed_history();std::shared_ptr<const PortableNativeLayer0Rows> failed0;PortableExpertRowsBatch bad=[](int64_t,const auto&,auto&,auto&){return false;};if(fail_write)fail_at="layer0 final write";rejected=false;try{failed0=layer0.run(1,{9},fail_write?typed_provider:bad);}catch(...){rejected=true;}fail_at.clear();check(rejected&&!failed0&&layer0.poisoned()&&layer0.position()==1&&layer0.committed_state()==state&&layer0.committed_history()==history&&prior0->residual()==saved,"layer0 failed publication changed prefix");auto before=in_quant;rejected=false;try{(void)layer0.run(1,{9},typed_provider);}catch(...){rejected=true;}check(rejected&&in_quant==before,"poisoned layer0 continued");}
   // Layer0 can commit before PLE fails: invalidate during FFN preparation and replay all owners.
   reset_typed();auto prefix0=layer0.run(0,{7},typed_provider);auto prefix_handle=typed_stage(prefix0);PortableNativeFfnComposition::Batch old1;chain.run(prefix_handle,typed_provider,[&](const auto&v){old1=v;});auto old_tuple=old1[0]->tuple().residual;auto old_state=owner.committed_state(),old_history=owner.committed_history(),old_ple=ple.history();auto old_prev=ple.previous();auto next0=layer0.run(1,{9},typed_provider);auto next_handle=typed_stage(next0);PortableNativeFfnComposition::Batch failed1;rejected=false;try{chain.run(next_handle,typed_provider,[&](const auto&v){failed1=v;},[&](const auto&){layer0.reset();});}catch(...){rejected=true;}check(rejected&&layer0.position()==0&&owner.position()==1&&ple.position()==1&&ple.history()==old_ple&&ple.previous()==old_prev&&owner.committed_state()==old_state&&owner.committed_history()==old_history&&!old_tuple.empty(),"typed preparation invalidation changed committed state");rejected=false;try{(void)failed1.at(0)->tuple();}catch(...){rejected=true;}check(rejected&&!old1[0]->valid_for(chain,2,0),"typed failed or stale snapshot survived");reset_typed();auto after_failure=replay(false,true);check(after_failure.outputs==typed_serial.outputs,"typed failure coordinated replay");
   owner.reset({3,5});chain.reset();ordinary.reset();ff2.reset();qowner.reset();ff3.reset();g4.reset();auto batch_result=replay(true);owner.reset({3,5});chain.reset();ordinary.reset();ff2.reset();qowner.reset();ff3.reset();g4.reset();auto serial_result=replay(false);
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
