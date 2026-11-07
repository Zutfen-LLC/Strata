// Actual PLE/HC composition and GDN owners, with low-level component mocks.
#define main prior_ple_integration_main
#include "portable_native_ple_integration_test.cpp"
#undef main
#include "portable_native_ple_gdn_composition.hpp"
namespace {
void need(bool b,const char*m){if(!b)throw std::runtime_error(m);}
int fail=0;size_t conv_calls=0,ab_calls=0,recur_calls=0;
std::vector<float> get(const ggml_tensor*t){std::vector<float>v(ggml_nbytes(t)/4);ggml_backend_tensor_get(t,v.data(),0,v.size()*4);return v;}
void put(ggml_tensor*t,const std::vector<float>&v){need(ggml_nbytes(t)==v.size()*4,"mock span");ggml_backend_tensor_set(t,v.data(),0,v.size()*4);}
}
namespace strata::core::portable_detail {
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
 auto stage=[&](std::vector<int32_t>tokens){std::vector<std::array<uint32_t,16>>indices;auto prev=ple.previous();std::vector<std::array<int32_t,2>>pred;for(auto t:tokens){pred.push_back(prev);prev={prev[1],t};}check(table->gather_batch(tokens,pred,indices,expected_embedding,error),"gather");norm_call=0;return ple.stage(ple.position(),tokens,std::vector<float>(tokens.size()*10240,0.25f));};
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
