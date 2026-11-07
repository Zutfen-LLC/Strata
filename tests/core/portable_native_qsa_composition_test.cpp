// Concrete composition/component owners with CPU low-level mocks; no GPU evidence.
#include "portable_native_hc_test_backend.hpp"
#include "portable_qsa_fixture.hpp"
#include "portable_native_qsa_composition.hpp"
#include "portable_native_ffn_test_access.hpp"
#ifdef STRATA_FFN_CPU_PROVIDER_TEST
#include "portable_cpu_experts.hpp"
#include "portable_model_fixture.hpp"
#endif
#include <optional>
namespace {size_t input_quant=0,qkv_calls=0,wo_quant=0,wo_calls=0;std::vector<float>row_input,row_gated;}
namespace strata::core::portable_detail {
#ifdef STRATA_FFN_CPU_PROVIDER_TEST
struct StrataVkNativeRouter{};struct StrataVkNativeMoeCombine{};
StrataVkNativeRouter* strata_vk_native_router_create(ggml_backend_t,const char*p){std::ofstream(p)<<"mock";return new StrataVkNativeRouter;}
size_t strata_vk_native_router_bytes(const StrataVkNativeRouter*){return 512;}
void strata_vk_native_router_destroy(StrataVkNativeRouter*p)noexcept{delete p;}
void strata_vk_native_router_run(StrataVkNativeRouter*,const ggml_tensor*logits,ggml_tensor*ids,ggml_tensor*weights,uint32_t n){check(logits->ne[1]==n,"factory router columns");std::vector<int32_t>v(n*10);for(size_t t=0;t<n;++t)for(size_t k=0;k<10;++k)v[t*10+k]=9-k;ggml_backend_tensor_set(ids,v.data(),0,v.size()*4);put(weights,std::vector<float>(n*10,.1f));}
StrataVkNativeMoeCombine* strata_vk_native_moe_combine_create(ggml_backend_t,const char*p){std::ofstream(p)<<"mock";return new StrataVkNativeMoeCombine;}
size_t strata_vk_native_moe_combine_bytes(const StrataVkNativeMoeCombine*){return 512;}
void strata_vk_native_moe_combine_destroy(StrataVkNativeMoeCombine*p)noexcept{delete p;}
void strata_vk_native_moe_combine_run(StrataVkNativeMoeCombine*,const ggml_tensor*parts,const ggml_tensor*weights,const ggml_tensor*shared,ggml_tensor*out,uint32_t n){auto a=get(parts),w=get(weights),v=get(shared);for(size_t t=0;t<n;++t)for(size_t k=0;k<10;++k)for(size_t h=0;h<2560;++h)v[t*2560+h]+=a[(t*10+k)*2560+h]*w[t*10+k];put(out,v);}
StrataVkNativeStages* strata_vk_native_stages_create_quant_rn_observation(ggml_backend_t,const char*p,size_t){std::ofstream(p)<<"mock";return new StrataVkNativeStages;}
StrataVkNativeShared* strata_vk_native_shared_create(ggml_backend_t,const char*p,size_t budget){std::ofstream(p)<<"mock";auto*r=new StrataVkNativeShared;r->budget=budget;return r;}
#endif
StrataVkQ8Wo* strata_vk_qsa_qkv_rows_create(ggml_backend_t,const char*p,size_t){std::ofstream(p)<<"mock";return new StrataVkQ8Wo;}
StrataVkQ8Wo* strata_vk_qsa_wo_rows_create(ggml_backend_t,const char*p,size_t){std::ofstream(p)<<"mock";return new StrataVkQ8Wo;}
StrataVkNativeShared* strata_vk_qsa_indexer_bf16_rows_create(ggml_backend_t,const char*p,size_t budget){check(budget>=512,"BF16 budget");std::ofstream(p)<<"mock";auto*x=new StrataVkNativeShared;x->budget=budget;return x;}
void strata_vk_qsa_input_quantize_bounded(StrataVkNativeStages*,const ggml_tensor*x,ggml_tensor*q,size_t rows){++input_quant;row_input=get(x);check(row_input.size()==rows*2560,"QSA input quant rows");auto bytes=native_q8_fixture::quantize(row_input);ggml_backend_tensor_set(q,bytes.data(),0,bytes.size());if(qsa_rows_fail=="input quant")throw std::runtime_error("input quant write then fail");}
void strata_vk_qsa_qkv_run_bounded(StrataVkQ8Wo*,const ggml_tensor*,const ggml_tensor*,ggml_tensor*y,size_t n,size_t rows){const size_t role=qkv_calls++%3;check((role==0&&n==12288)||(role&&n==512),"QKV role order");std::vector<float>v(rows*n);for(size_t r=0;r<rows;++r)for(size_t i=0;i<n;++i)v[r*n+i]=row_input[r*2560]+float(int((i+role*3)%17)-8)/32.f;put(y,v);if(qsa_rows_fail=="qkv")throw std::runtime_error("QKV write then fail");}
void strata_vk_qsa_wo_quantize_bounded(StrataVkNativeStages*,const ggml_tensor*x,ggml_tensor*q,size_t rows){++wo_quant;row_gated=get(x);check(row_gated.size()==rows*6144,"QSA WO quant rows");auto bytes=native_q8_fixture::quantize(row_gated);ggml_backend_tensor_set(q,bytes.data(),0,bytes.size());if(qsa_rows_fail=="wo quant")throw std::runtime_error("WO quant write then fail");}
void strata_vk_qsa_wo_run_bounded(StrataVkQ8Wo*,const ggml_tensor*,const ggml_tensor*,ggml_tensor*y,size_t rows){++wo_calls;std::vector<float>v(rows*2560);for(size_t r=0;r<rows;++r)std::copy_n(row_gated.begin()+r*6144,2560,v.begin()+r*2560);put(y,v);if(qsa_rows_fail=="wo")throw std::runtime_error("WO write then fail");}
}
int main(int argc,char**argv){try{
 check(argc==2,"scratch path");std::string error;auto b=std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)>(ggml_backend_cpu_init(),ggml_backend_free);
 gdn_fixture::Fixture f;auto views=f.views;for(auto&v:views)v.name.replace(0,6,"blk.2.");PortableGdnOptions options;options.execution.threads=1;options.execution.accelerator=b.get();options.execution.request_accelerator=true;options.defer_ffn_hc_write=true;
 auto producer=PortableGdnLayer::create(2,views,options,error);check(bool(producer)&&producer->reset(error),error);PortableRoute route;auto experts=[](int64_t,const PortableRoute&r,std::vector<float>&v,std::string&e){return gdn_fixture::experts()(0,r,v,e);};
 std::vector<std::shared_ptr<const PortableGdnSnapshot>> snapshots;for(int p=0;p<5;++p){check(producer->step(f.input(p),experts,route,error),error);snapshots.push_back(producer->pending_snapshot(error));check(bool(snapshots.back()),error);}
 auto c=std::unique_ptr<ggml_context,decltype(&ggml_free)>(ggml_init({32768,nullptr,true}),ggml_free);
 std::array<ggml_tensor*,4>hcroles{ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,10240),ggml_new_tensor_1d(c.get(),GGML_TYPE_I32,20480),ggml_new_tensor_1d(c.get(),GGML_TYPE_I32,1638400),ggml_new_tensor_1d(c.get(),GGML_TYPE_I32,1638400)};
 std::array<ggml_tensor*,3>q{ggml_new_tensor_2d(c.get(),GGML_TYPE_Q8_0,2560,12288),ggml_new_tensor_2d(c.get(),GGML_TYPE_Q8_0,2560,512),ggml_new_tensor_2d(c.get(),GGML_TYPE_Q8_0,2560,512)};
 std::array<ggml_tensor*,2>ix{ggml_new_tensor_1d(c.get(),GGML_TYPE_I32,128*2560/2),ggml_new_tensor_1d(c.get(),GGML_TYPE_I32,512*2560/2)};auto*wo=ggml_new_tensor_2d(c.get(),GGML_TYPE_Q8_0,6144,2560);
 std::array<ggml_tensor*,4>gamma{ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,256),ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,256),ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,128),ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,128)};
 auto buf=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>(ggml_backend_alloc_ctx_tensors(c.get(),b.get()),ggml_backend_buffer_free);check(bool(buf),"roles allocation");
 for(auto*t:hcroles){std::vector<uint8_t>z(ggml_nbytes(t));ggml_backend_tensor_set(t,z.data(),0,z.size());}for(auto*t:gamma){std::vector<float>x(ggml_nelements(t),1.f);put(t,x);}
 std::array<std::vector<uint8_t>,3>qr;for(size_t i=0;i<3;++i){qr[i].resize(ggml_nbytes(q[i]));ggml_backend_tensor_set(q[i],qr[i].data(),0,qr[i].size());}std::vector<uint8_t>wr(ggml_nbytes(wo));ggml_backend_tensor_set(wo,wr.data(),0,wr.size());
 std::array<std::vector<uint8_t>,2>ir;for(size_t i=0;i<2;++i){ir[i].resize(ggml_nbytes(ix[i]));for(size_t r=0;r<(i?512u:128u);++r){ir[i][r*5120]=0x80;ir[i][r*5120+1]=0x3f;}ggml_backend_tensor_set(ix[i],ir[i].data(),0,ir[i].size());}
 auto root=std::filesystem::path(argv[1])/("qsa-composition-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));check(std::filesystem::create_directory(root),"fresh root");
 using HC=PortableNativeHcComposed;HC::Dispatch dispatch;float hc_bias=0.f;size_t hc_pending_calls=0;
 dispatch.pending=[&](const HC::Bank&bank){++hc_pending_calls;auto v=get(bank.input);put(bank.residual,v);};
 dispatch.norm=[](const HC::Bank&bank,uint32_t n){std::vector<float>in(n*4,0.25f);put(bank.inject,in);};
 dispatch.mix=[&](const HC::Bank&bank){auto block=get(bank.block);size_t n=block.size()/2560;std::vector<float>m(n*2560);for(size_t r=0;r<n;++r){float x=0.5f+hc_bias+block[r*2560]*0.0625f;std::fill(m.begin()+r*2560,m.begin()+(r+1)*2560,x);}put(bank.mixed,m);};
 auto hc=std::make_unique<HC>(b.get(),hcroles[0],hcroles[1],hcroles[2],hcroles[3],20*1024*1024,nullptr,dispatch);
 auto qkv=std::make_unique<PortableNativeQsaQkvRows>(b.get(),q,std::array<const void*,3>{qr[0].data(),qr[1].data(),qr[2].data()},std::array<size_t,3>{qr[0].size(),qr[1].size(),qr[2].size()},(root/"qkv").string(),PortableNativeQsaQkvRows::estimate(b.get()));
 auto proj=std::make_unique<PortableNativeQsaIndexerRows>(b.get(),ix,std::array<const void*,2>{ir[0].data(),ir[1].data()},std::array<size_t,2>{ir[0].size(),ir[1].size()},(root/"proj").string(),PortableNativeQsaIndexerRows::estimate(b.get()));
 constexpr uint32_t cap=8;auto norm=std::make_unique<PortableNativeQsaNormRope>(b.get(),std::array<ggml_tensor*,3>{gamma[0],gamma[1],gamma[2]},(root/"norm").string(),PortableNativeQsaNormRope::estimate(b.get()),false);
 auto indexer=std::make_unique<PortableNativeQsaIndexer>(b.get(),gamma[3],cap,(root/"indexer").string(),PortableNativeQsaIndexer::estimate(b.get(),cap),false);
 auto select=std::make_unique<PortableNativeQsaSelect>(b.get(),cap,(root/"select").string(),PortableNativeQsaSelect::estimate(b.get(),cap),false);
 auto attention=std::make_unique<PortableNativeQsaAttention>(b.get(),cap,(root/"attention").string(),PortableNativeQsaAttention::estimate(b.get(),cap),false);
 auto output=std::make_unique<PortableNativeQsaWoRows>(b.get(),wo,wr.data(),wr.size(),(root/"wo").string(),PortableNativeQsaWoRows::estimate(b.get()));
 using Owner=PortableNativeQsaComposition;Owner owner(3,cap,Owner::host_bound(cap),std::move(hc),std::move(qkv),std::move(proj),std::move(norm),std::move(indexer),std::move(select),std::move(attention),std::move(output));observe_qsa_rows=true;
 // Connected QSA candidate -> FFN HC -> real row owner -> next HC.
 {
  auto fc=std::unique_ptr<ggml_context,decltype(&ggml_free)>(ggml_init({8192,nullptr,true}),ggml_free);
  auto*rw=ggml_new_tensor_2d(fc.get(),GGML_TYPE_F32,2560,512);auto fb=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>(ggml_backend_alloc_ctx_tensors(fc.get(),b.get()),ggml_backend_buffer_free);
  std::vector<float>zero(2560*512);ggml_backend_tensor_set(rw,zero.data(),0,zero.size()*4);PortableNativeFfnProjection router(b.get(),rw,16u<<20);
  PortableNativeFfnRows::Dispatch fd;
  fd.router=[](size_t n,const ggml_tensor* logits,std::vector<PortableRoute>&r){check(logits->ne[1]==int64_t(n),"FFN router columns");for(auto&v:r){v.native_router=true;for(int k=0;k<10;++k){v.ids.push_back(9-k);v.weights.push_back(.1f);}}};
  fd.shared=[](size_t n,const ggml_tensor*){return std::vector<float>(n*2560,1.f);};
  fd.combine=[](size_t n,const auto&parts,const auto&w,const auto&shared){auto out=shared;for(size_t t=0;t<n;++t)for(size_t k=0;k<10;++k)for(size_t h=0;h<2560;++h)out[t*2560+h]+=parts[(t*10+k)*2560+h]*w[t*10+k];return out;};
  PortableNativeFfnRows rows(3,router,fd);HC::Dispatch ff;
  ff.pending=[](const HC::Bank&bank){auto r=get(bank.input),block=get(bank.block);size_t n=block.size()/2560;for(size_t t=0;t<n;++t)for(size_t s=0;s<4;++s)for(size_t h=0;h<2560;++h)r[t*10240+s*2560+h]+=block[t*2560+h];put(bank.residual,r);};
  ff.norm=[](const HC::Bank&bank,uint32_t n){put(bank.inject,std::vector<float>(n*4,.5f));};
  ff.mix=[](const HC::Bank&bank){auto r=get(bank.residual);size_t n=r.size()/10240;std::vector<float>m(n*2560);for(size_t t=0;t<n;++t)std::copy_n(r.begin()+t*10240,2560,m.begin()+t*2560);put(bank.mixed,m);};
  auto ffhc=std::make_unique<HC>(b.get(),hcroles[0],hcroles[1],hcroles[2],hcroles[3],20*1024*1024,nullptr,ff);
  auto chain=PortableNativeFfnCompositionTestAccess::make(owner,std::move(ffhc),rows,PortableNativeFfnComposition::host_bound());HC next(b.get(),hcroles[0],hcroles[1],hcroles[2],hcroles[3],20*1024*1024,nullptr,ff);
  size_t calls=0;PortableExpertRowsBatch expert=[&](int64_t layer,const auto&r,auto&v,auto&){++calls;check(layer==3&&r.size()>=1&&r.size()<=2,"FFN expert rows");v.assign(r.size()*10*2560,2.f);return true;};
  PortableNativeFfnComposition::Batch first,tail,failed_ff;
  bool wrong=false;try{chain.run(1,*producer,{snapshots[1]},expert,[](const auto&){});}catch(...){wrong=true;}check(wrong&&!chain.poisoned()&&calls==0,"FFN wrong source position dispatched");
  chain.run(0,*producer,{snapshots[0],snapshots[1]},expert,[&](const auto&v){first=v;check(v.size()==2,"FFN T2 candidate");});
  check(calls==1&&chain.position()==2&&first[0]->valid_for(chain,4,0)&&first[1]->valid_for(chain,4,1),"FFN T2 commit");
  auto saved=first[0]->tuple().residual;check(first[0]->tuple().block[0]>2.9f&&first[0]->tuple().injection[0]==.5f,"post FFN tuple values");
  bool bad=false;try{next.run(4,0,chain,{first[1],first[0]},[](const HC::View&,uint32_t){},[](const HC::View&,uint32_t){});}catch(...){bad=true;}check(bad&&next.rows()==0,"swapped FFN snapshots admitted");
  bad=false;try{next.run(4,0,chain,{first[0],first[0]},[](const HC::View&,uint32_t){},[](const HC::View&,uint32_t){});}catch(...){bad=true;}check(bad&&next.rows()==0,"duplicate FFN snapshots admitted");
  next.run(4,0,chain,first,[&](const HC::View&v,uint32_t n){check(n==2&&v.read(HC::View::Field::residual)[0]==saved[0]+first[0]->tuple().block[0],"next HC consumes FFN tuple");},[](const HC::View&,uint32_t){});
  chain.run(2,*producer,{snapshots[2]},expert,[&](const auto&v){tail=v;});check(calls==2&&chain.position()==3&&first[0]->tuple().residual==saved&&first[0]->valid_for(chain,4,0),"FFN tail/retention");
  auto before_late=owner.state();bool rejected=false;try{chain.run(3,*producer,{snapshots[3]},expert,[&](const auto&v){failed_ff=v;},[](const auto&){throw std::runtime_error("late FFN prepare");});}catch(...){rejected=true;}
  auto after_late=owner.state();check(rejected&&chain.poisoned()&&owner.position()==3&&chain.position()==3&&first[0]->tuple().residual==saved&&after_late.keys==before_late.keys&&after_late.values==before_late.values&&after_late.indexer==before_late.indexer,"FFN late rollback");rejected=false;try{failed_ff[0]->tuple();}catch(...){rejected=true;}check(rejected,"failed FFN snapshot survived");
  owner.reset();check(!first[0]->valid_for(chain,4,0),"source reset left FFN provenance live");chain.reset();rejected=false;try{first[0]->tuple();}catch(...){rejected=true;}check(rejected,"FFN reset did not invalidate retained tuple");
  auto replay=[&](float bias){owner.reset();chain.reset();hc_bias=bias;PortableNativeFfnComposition::Batch v;bool reentrant=false;chain.run(0,*producer,{snapshots[0],snapshots[1]},expert,[&](const auto&s){v=s;try{chain.reset();}catch(...){reentrant=true;}});check(reentrant,"reentrant FFN reset");return v[0]->tuple().residual;};
  auto a=replay(0.f),bvalue=replay(.125f),again=replay(0.f);check(a!=bvalue&&a==again,"FFN reset A/B/A");
  PortableNativeFfnComposition::Batch prior;chain.run(2,*producer,{snapshots[2]},expert,[&](const auto&v){prior=v;});auto prior_value=prior[0]->tuple().block;
  PortableExpertRowsBatch refusal=[](int64_t,const auto&,auto&,auto&error){error="rejected";return false;};rejected=false;
  try{chain.run(3,*producer,{snapshots[3]},refusal,[](const auto&){});}catch(...){rejected=true;}
  check(rejected&&chain.poisoned()&&chain.position()==3&&owner.position()==3&&prior[0]->tuple().block==prior_value,"FFN provider failure rollback");
  rejected=false;try{chain.run(3,*producer,{snapshots[3]},expert,[](const auto&){});}catch(...){rejected=true;}check(rejected,"FFN poisoned continuation");
  owner.reset();chain.reset();hc_bias=0;
  PortableNativeFfnComposition::Batch before_source,attempt_source;chain.run(0,*producer,{snapshots[0]},expert,[&](const auto&v){before_source=v;});auto held=before_source[0]->tuple().residual;
  rejected=false;try{chain.run(1,*producer,{snapshots[1]},expert,[&](const auto&v){attempt_source=v;},[&](const auto&){check(producer->reset(error),error);});}catch(...){rejected=true;}
  check(rejected&&chain.poisoned()&&chain.position()==1&&owner.position()==1&&before_source[0]->tuple().residual==held&&!before_source[0]->valid_for(chain,4,0),"FFN source invalidation rollback/provenance");
  rejected=false;try{attempt_source[0]->tuple();}catch(...){rejected=true;}check(rejected,"FFN invalidated source candidate survived");
  snapshots.clear();for(int p=0;p<5;++p){check(producer->step(f.input(p),experts,route,error),error);snapshots.push_back(producer->pending_snapshot(error));check(bool(snapshots.back()),error);}
  owner.reset();chain.reset();
  // Actual named-role factory and production HC component wiring, T2/T1. The
  // component implementation here remains a CPU low-level mock.
  {
   std::vector<float> norm(10240,2.f);std::array<std::vector<uint8_t>,3> packed;const size_t lengths[]={40960,3276800,3276800};
   for(size_t i=0;i<3;++i){packed[i].resize(lengths[i]/32*34);for(size_t j=0;j<packed[i].size();j+=34){packed[i][j+1]=0x3c;packed[i][j+2]=uint8_t(i+1);}factory_ffn_roles[i]=native_hc_q8_bf16(packed[i].data(),packed[i].size(),lengths[i]);}
   std::vector<PortableTensorView> ffviews{{"blk.3.hc_ffn_norm.weight",{10240},0,norm.data(),norm.size()*4},{"blk.3.hc_ffn_inject.weight",{10240,4},8,packed[0].data(),packed[0].size()},{"blk.3.hc_ffn_down.weight",{10240,320},8,packed[1].data(),packed[1].size()},{"blk.3.hc_ffn_up.weight",{320,10240},8,packed[2].data(),packed[2].size()}};
#ifdef STRATA_FFN_CPU_PROVIDER_TEST
   model_fixture::SparseModel fixture;strata::GgufModel model(fixture.paths);PortableModelOptions opts;opts.expert_workers=1;CpuExperts cpu_experts(model,(fixture.root/"manifest").string(),opts);
   auto gguf_roles=NativeFfnHcFactory::from_gguf(model,3);check(gguf_roles.size()==4&&gguf_roles[0].name=="blk.3.hc_ffn_norm.weight"&&gguf_roles[3].name=="blk.3.hc_ffn_up.weight","GGUF role selection");
   bool missing_gguf=false;try{(void)NativeFfnHcFactory::from_gguf(model,48);}catch(...){missing_gguf=true;}check(missing_gguf,"GGUF factory missing roles admitted");
   auto wc=std::unique_ptr<ggml_context,decltype(&ggml_free)>(ggml_init({8192,nullptr,true}),ggml_free);std::array<ggml_tensor*,4> weights;
   for(size_t i=0;i<3;++i)weights[i]=ggml_new_tensor_1d(wc.get(),GGML_TYPE_I32,87040/4);weights[3]=ggml_new_tensor_1d(wc.get(),GGML_TYPE_I32,5120/4);
   auto wb=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>(ggml_backend_alloc_ctx_tensors(wc.get(),b.get()),ggml_backend_buffer_free);
   std::array<std::unique_ptr<PortableNativeRouter>,2> nr;std::array<std::unique_ptr<PortableNativeShared>,2> ns;std::array<std::unique_ptr<PortableNativeMoeCombine>,2> nc;
   for(size_t i=0;i<2;++i){nr[i]=std::make_unique<PortableNativeRouter>(b.get(),i+1,16u<<20,(root/"ffn-router").string());ns[i]=std::make_unique<PortableNativeShared>(b.get(),2560,32,i+1,weights,16u<<20,(root/"ffn-shared").string());nc[i]=std::make_unique<PortableNativeMoeCombine>(b.get(),i+1,16u<<20,(root/"ffn-combine").string());}
   PortableNativeFfnRows native_rows(3,router,PortableNativeFfnRows::native_dispatch({nr[0].get(),nr[1].get()},{ns[0].get(),ns[1].get()},{nc[0].get(),nc[1].get()}));
   size_t actual_calls=0;PortableExpertRowsBatch actual_provider=[&](int64_t l,const auto&r,auto&out,auto&e){++actual_calls;return cpu_experts.run_batch(l,r,out,e);};
   auto& selected_rows=native_rows;auto& selected_provider=actual_provider;
#else
   auto& selected_rows=rows;auto& selected_provider=expert;
#endif
   auto prefix=(root/"ffn-factory").string();PortableNativeFfnComposition real(owner,b.get(),ffviews,selected_rows,40u<<20,32u<<20,prefix.c_str(),HC::ComponentCaps{512,512,512});
   std::fill(norm.begin(),norm.end(),9.f);for(auto&v:packed)std::fill(v.begin(),v.end(),0);factory_ffn_mode=true;factory_shared_mode=true;
   PortableNativeFfnComposition::Batch actual,actual_tail;real.run(0,*producer,{snapshots[0],snapshots[1]},selected_provider,[&](const auto&v){actual=v;});real.run(2,*producer,{snapshots[2]},selected_provider,[&](const auto&v){actual_tail=v;});factory_ffn_mode=false;factory_shared_mode=false;
#ifdef STRATA_FFN_CPU_PROVIDER_TEST
   check(actual_calls==2&&cpu_experts.work().native_batches==2&&cpu_experts.work().batch_max_nt==2,"factory/native owners/CPU grouped provider chain");
#endif
   check(actual.size()==2&&actual_tail.size()==1&&actual[0]->valid_for(real,4,0)&&actual_tail[0]->valid_for(real,4,2)&&actual[0]->tuple().injection[0]==.5f,"production FFN factory chain");
   for(const char* suffix:{".pending.spv",".norm.spv",".mix.spv"})check(std::filesystem::file_size(prefix+suffix)>0,"factory HC receipt");
  }
  owner.reset();
 }
 std::vector<float>batched,serial;std::vector<int32_t>batch_ids,serial_ids;std::optional<Owner::View>old,failed;
 auto run=[&](size_t first,size_t n,std::vector<float>&out,std::vector<int32_t>&ids){std::vector<std::shared_ptr<const PortableGdnSnapshot>>s(snapshots.begin()+first,snapshots.begin()+first+n);owner.run(first,*producer,s,[&](const Owner::View&v){check(v.first()==first&&v.rows()==n&&v.residual().size()==n*10240&&v.injection().size()==n*4,"view provenance");auto x=v.output();out.insert(out.end(),x.begin(),x.end());for(size_t r=0;r<n;++r){auto z=v.selected(r);ids.insert(ids.end(),z.begin(),z.end());}old=v;});};
 input_quant=qkv_calls=qsa_row_bf_calls=wo_quant=wo_calls=qsa_row_idx_calls=qsa_row_score_calls=qsa_row_select_calls=qsa_row_attn_calls=0;auto before_norm=qsa_norm_calls;run(0,2,batched,batch_ids);check(input_quant==1&&qkv_calls==3&&qsa_row_bf_calls==2&&wo_quant==1&&wo_calls==1&&qsa_norm_calls-before_norm==6&&qsa_row_idx_calls==2&&qsa_row_score_calls==2&&qsa_row_select_calls==2&&qsa_row_attn_calls==2,"T2 real stage counts");run(2,1,batched,batch_ids);auto batch_state=owner.state();check(owner.position()==3&&batched.size()==3*2560,"T2 tail commit");
 {auto prior=old;auto bad=batch_state;auto gen=owner.generation();bad.indexer.back()=4;bool rejected=false;try{owner.reset(bad);}catch(...){rejected=true;}auto unchanged=owner.state();check(rejected&&owner.generation()==gen&&unchanged.position==3&&unchanged.keys==batch_state.keys&&unchanged.values==batch_state.values&&unchanged.indexer==batch_state.indexer&&prior->rows()==1,"invalid seed changed publication/state/generation");}
 std::vector<float>cross,cross_replay;std::vector<int32_t>cross_ids,replay_ids;qsa_row_candidate_3.clear();qsa_row_before_4.clear();run(3,2,cross,cross_ids);auto cross_state=owner.state();check(!qsa_row_candidate_3.empty()&&qsa_row_candidate_3==qsa_row_before_4,"row1 did not consume row0 private indexer history");
 owner.reset(batch_state);auto restored=owner.state();check(restored.position==3&&restored.keys==batch_state.keys&&restored.values==batch_state.values&&restored.indexer==batch_state.indexer,"valid seed restore");run(3,2,cross_replay,replay_ids);auto replay_state=owner.state();check(cross==cross_replay&&cross_ids==replay_ids&&cross_state.keys==replay_state.keys&&cross_state.values==replay_state.values&&cross_state.indexer==replay_state.indexer,"saved-state continuation differs");
 owner.reset();run(0,1,serial,serial_ids);run(1,1,serial,serial_ids);run(2,1,serial,serial_ids);auto serial_state=owner.state();check(batched==serial&&batch_ids==serial_ids&&batch_state.keys==serial_state.keys&&batch_state.values==serial_state.values&&batch_state.indexer==serial_state.indexer,"T2 tail differs from three T1");std::vector<float>cross_serial;std::vector<int32_t>cross_serial_ids;run(3,1,cross_serial,cross_serial_ids);run(4,1,cross_serial,cross_serial_ids);auto cross_serial_state=owner.state();check(cross==cross_serial&&cross_ids==cross_serial_ids&&cross_state.keys==cross_serial_state.keys&&cross_state.values==cross_serial_state.values&&cross_state.indexer==cross_serial_state.indexer,"pool completion T2 differs from T1");
 check(!std::equal(batched.begin(),batched.begin()+2560,batched.begin()+2560),"distinct row canaries");
 for(const std::string point:{"input quant","qkv","bf16","indexer","score","select","attention","wo quant","wo","consumer","prepare"})for(int row=0;row<(point=="indexer"||point=="score"||point=="select"||point=="attention"?2:1);++row){
  owner.reset();qkv_calls=0;std::vector<float>x;std::vector<int32_t>z;run(0,1,x,z);auto prior=old;auto state=owner.state();failed.reset();qsa_rows_fail=point;qsa_rows_fail_position=1+row;bool rejected=false;
  try{owner.run(1,*producer,{snapshots[1],snapshots[2]},[&](const Owner::View&v){failed=v;if(point=="consumer")throw std::runtime_error("consumer");},[&](const Owner::View&){if(point=="prepare")throw std::runtime_error("prepare");});}catch(...){rejected=true;}
  qsa_rows_fail.clear();qsa_rows_fail_position=-1;auto after=owner.state();check(rejected&&owner.poisoned()&&after.position==state.position&&after.keys==state.keys&&after.values==state.values&&after.indexer==state.indexer&&prior->output()==x,"failure changed committed state/publication");if(failed){bool dead=false;try{(void)failed->output();}catch(...){dead=true;}check(dead,"failed view survived");}
 }
 owner.reset();hc_bias=0.125f;std::vector<float>other;std::vector<int32_t>other_ids;run(0,2,other,other_ids);run(2,1,other,other_ids);check(other!=batched,"reset B did not alter output");
 owner.reset();hc_bias=0;std::vector<float>again;std::vector<int32_t>again_ids;run(0,2,again,again_ids);run(2,1,again,again_ids);check(again==batched&&again_ids==batch_ids,"reset A/B/A differs");
 owner.reset();bool refused=false;try{owner.run(0,*producer,{snapshots[1],snapshots[0]},[](const Owner::View&){});}catch(...){refused=true;}check(refused&&owner.position()==0,"reordered snapshots admitted");
 owner.reset();refused=false;try{owner.run(1,*producer,{snapshots[1]},[](const Owner::View&){});}catch(...){refused=true;}check(refused&&!owner.poisoned()&&owner.position()==0,"wrong position admitted");
 auto foreign=PortableGdnLayer::create(2,views,options,error);check(bool(foreign)&&foreign->reset(error),error);owner.reset();refused=false;try{owner.run(0,*foreign,{snapshots[0]},[](const Owner::View&){});}catch(...){refused=true;}check(refused&&owner.position()==0,"wrong owner admitted");
 std::vector<std::shared_ptr<const PortableGdnSnapshot>>fs;for(int p=0;p<2;++p){check(foreign->step(f.input(p),experts,route,error),error);fs.push_back(foreign->pending_snapshot(error));check(bool(fs.back()),error);}
 owner.reset();std::vector<float>bound_out;std::vector<int32_t>bound_ids;run(0,1,bound_out,bound_ids);auto bound_view=old;auto bound_state=owner.state();auto bound_generation=owner.generation();auto before_dispatch=input_quant;auto before_hc=hc_pending_calls;refused=false;try{owner.run(1,*foreign,{fs[1]},[](const Owner::View&){throw std::runtime_error("should not run");});}catch(...){refused=true;}auto after_foreign=owner.state();check(refused&&!owner.poisoned()&&input_quant==before_dispatch&&hc_pending_calls==before_hc&&owner.generation()==bound_generation&&after_foreign.position==bound_state.position&&after_foreign.keys==bound_state.keys&&after_foreign.values==bound_state.values&&after_foreign.indexer==bound_state.indexer&&bound_view->output()==bound_out,"foreign valid next-position snapshot mixed histories or dispatched");
 owner.reset();owner.run(0,*foreign,{fs[0],fs[1]},[](const Owner::View&v){check(v.rows()==2,"foreign reset rows");});check(owner.position()==2,"reset did not permit fresh foreign prefix");
 owner.reset();bool reset_refused=false,run_refused=false;std::vector<std::shared_ptr<const PortableGdnSnapshot>>mutable_handles{snapshots[0]};owner.run(0,*producer,mutable_handles,[&](const Owner::View&v){check(v.injection()==std::vector<float>(4,0.25f)&&v.residual()==snapshots[0]->tuple().residual,"HC provenance differs");mutable_handles.clear();try{owner.reset();}catch(...){reset_refused=true;}try{owner.run(0,*producer,{snapshots[0]},[](const Owner::View&){});}catch(...){run_refused=true;}});check(reset_refused&&run_refused&&owner.position()==1,"reentrant mutation accepted");
 // Seeded native component tests use all-zero valid prior histories (and the
 // corresponding block word), avoiding thousands of whole-layer steps.
 {
  constexpr uint32_t big=4096;auto meta=std::unique_ptr<ggml_context,decltype(&ggml_free)>(ggml_init({8192,nullptr,true}),ggml_free);
  auto*raw=ggml_new_tensor_1d(meta.get(),GGML_TYPE_F32,128);auto*iq=ggml_new_tensor_1d(meta.get(),GGML_TYPE_F32,512);auto*query=ggml_new_tensor_1d(meta.get(),GGML_TYPE_F32,6144);auto*full=ggml_new_tensor_1d(meta.get(),GGML_TYPE_F32,12288);
  for(auto*t:{raw,iq,query,full})ggml_set_output(t);auto scratch=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>(ggml_backend_alloc_ctx_tensors(meta.get(),b.get()),ggml_backend_buffer_free);check(bool(scratch),"seeded scratch");
  put(iq,std::vector<float>(512,0.125f));put(query,std::vector<float>(6144,0.125f));put(full,std::vector<float>(12288,0.25f));
  PortableNativeQsaIndexer ia(b.get(),gamma[3],big,(root/"seed-index-a").string(),PortableNativeQsaIndexer::estimate(b.get(),big),false),ib(b.get(),gamma[3],big,(root/"seed-index-b").string(),PortableNativeQsaIndexer::estimate(b.get(),big),false);
  PortableNativeQsaSelect ss(b.get(),big,(root/"seed-select").string(),PortableNativeQsaSelect::estimate(b.get(),big),false);PortableNativeQsaAttention aa(b.get(),big,(root/"seed-attention").string(),PortableNativeQsaAttention::estimate(b.get(),big),false);
  for(uint32_t start:{0u,3u,4u,7u,8u,255u,256u,2050u,2051u,2052u,4094u,4095u}){
   const size_t count=std::min<uint32_t>(2,big-start);std::vector<uint32_t>before(qsa_idx_words(big));before.back()=start>=4?4*((start-1)/4):0;
   auto serial_before=before;ib.commit(serial_before,start);std::vector<uint32_t>private_state=before;std::vector<uint16_t>k(size_t(big)*512),v(k.size());std::vector<std::vector<int32_t>>ids;std::vector<std::vector<float>>outputs;
   for(size_t r=0;r<count;++r){auto p=start+r;put(raw,std::vector<float>(128,r?0.25f:0.125f));std::fill_n(k.begin()+p*512,512,ggml_fp32_to_fp16(0.125f));std::fill_n(v.begin()+p*512,512,ggml_fp32_to_fp16(0.25f));
    private_state=ia.run_before(raw,p,private_state);auto chosen=ss.run(iq,private_state,p);ids.push_back(chosen.ids);std::vector<uint16_t>ak(k.begin(),k.begin()+(p+1)*512),av(v.begin(),v.begin()+(p+1)*512);outputs.push_back(aa.run(query,full,ak,av,chosen.ids,p).gated);
    auto next=ib.run(raw,p);check(next==private_state,"seeded private indexer differs from serial candidate");ib.commit(next,p);auto serial_chosen=ss.run(iq,ib.state(),p);check(serial_chosen.ids==ids.back(),"seeded selected IDs differ");check(aa.run(query,full,ak,av,serial_chosen.ids,p).gated==outputs.back(),"seeded attention differs");
   }
   check(private_state==ib.state(),"seeded final history differs");
  }
 }
 // Recording failure after the real indexer candidate was written, with a
 // separate owner and a deliberately pre-existing receipt directory. Keep it.
 {
  auto d=root/"recording";check(std::filesystem::create_directory(d),"recording root");
  auto h=std::make_unique<HC>(b.get(),hcroles[0],hcroles[1],hcroles[2],hcroles[3],20*1024*1024,nullptr,dispatch);
  auto a=std::make_unique<PortableNativeQsaQkvRows>(b.get(),q,std::array<const void*,3>{qr[0].data(),qr[1].data(),qr[2].data()},std::array<size_t,3>{qr[0].size(),qr[1].size(),qr[2].size()},(d/"qkv").string(),PortableNativeQsaQkvRows::estimate(b.get()));
  auto p=std::make_unique<PortableNativeQsaIndexerRows>(b.get(),ix,std::array<const void*,2>{ir[0].data(),ir[1].data()},std::array<size_t,2>{ir[0].size(),ir[1].size()},(d/"proj").string(),PortableNativeQsaIndexerRows::estimate(b.get()));
  auto n=std::make_unique<PortableNativeQsaNormRope>(b.get(),std::array<ggml_tensor*,3>{gamma[0],gamma[1],gamma[2]},(d/"norm").string(),PortableNativeQsaNormRope::estimate(b.get()),false);
  auto i=std::make_unique<PortableNativeQsaIndexer>(b.get(),gamma[3],cap,(d/"indexer").string(),PortableNativeQsaIndexer::estimate(b.get(),cap),true);
  auto s=std::make_unique<PortableNativeQsaSelect>(b.get(),cap,(d/"select").string(),PortableNativeQsaSelect::estimate(b.get(),cap),false);
  auto t=std::make_unique<PortableNativeQsaAttention>(b.get(),cap,(d/"attention").string(),PortableNativeQsaAttention::estimate(b.get(),cap),false);
  auto w=std::make_unique<PortableNativeQsaWoRows>(b.get(),wo,wr.data(),wr.size(),(d/"wo").string(),PortableNativeQsaWoRows::estimate(b.get()));
  Owner recorded(3,cap,Owner::host_bound(cap),std::move(h),std::move(a),std::move(p),std::move(n),std::move(i),std::move(s),std::move(t),std::move(w));std::optional<Owner::View>previous;
  qkv_calls=0;recorded.run(0,*producer,{snapshots[0]},[&](const Owner::View&v){previous=v;});auto state=recorded.state();auto output_before=previous->output();check(std::filesystem::create_directory(d/"indexer/first-2"),"recording collision setup");auto count_before=qsa_row_idx_calls;bool caught=false;
  try{recorded.run(1,*producer,{snapshots[1],snapshots[2]},[](const Owner::View&){});}catch(const std::exception&e){caught=std::string(e.what()).find("fresh receipt")!=std::string::npos;}
  auto state_after=recorded.state();check(caught&&qsa_row_idx_calls==count_before+2&&recorded.poisoned()&&state_after.position==state.position&&state_after.keys==state.keys&&state_after.values==state.values&&state_after.indexer==state.indexer&&previous->output()==output_before,"recording failure committed or skipped candidate");
  check(std::filesystem::file_size(d/"indexer/first-1/state.u32le")==qsa_idx_words(cap)*4&&std::filesystem::file_size(d/"indexer/first-1/diag.f32le")==386*4&&std::filesystem::exists(d/"indexer/first-1/fixture.bin"),"recording output names/sizes");
 }
 owner.reset();qkv_calls=0;std::vector<float>prior_output;std::vector<int32_t>prior_ids;run(0,1,prior_output,prior_ids);auto prior=old;failed.reset();refused=false;try{owner.run(1,*producer,{snapshots[1]},[&](const Owner::View&v){failed=v;check(producer->reset(error),error);});}catch(...){refused=true;}check(refused&&owner.poisoned()&&owner.position()==1&&prior->output()==prior_output,"producer invalidation published candidate");bool dead=false;try{(void)failed->output();}catch(...){dead=true;}check(dead,"producer invalidation retained candidate");
 auto stale_seed=owner.state();auto stale_generation=owner.generation();refused=false;try{owner.reset(stale_seed);}catch(...){refused=true;}check(refused&&owner.generation()==stale_generation&&owner.position()==1&&owner.poisoned()&&prior->output()==prior_output,"stale seed reset changed committed publication");
 owner.reset();refused=false;try{owner.run(0,*producer,{snapshots[0]},[](const Owner::View&){});}catch(...){refused=true;}check(refused&&owner.position()==0,"stale snapshots admitted");
 std::vector<std::shared_ptr<const PortableGdnSnapshot>>first_generation;for(int p=0;p<2;++p){check(producer->step(f.input(p),experts,route,error),error);first_generation.push_back(producer->pending_snapshot(error));check(bool(first_generation.back()),error);}
 owner.reset();std::optional<Owner::View>generation_view;owner.run(0,*producer,{first_generation[0]},[&](const Owner::View&v){generation_view=v;});auto generation_state=owner.state();auto generation_output=generation_view->output();auto generation_before=owner.generation();check(producer->reset(error),error);
 std::vector<std::shared_ptr<const PortableGdnSnapshot>>replayed;for(int p=0;p<2;++p){check(producer->step(f.input(p),experts,route,error),error);replayed.push_back(producer->pending_snapshot(error));check(bool(replayed.back()),error);}check(replayed[1]->valid_for(*producer,3,1),"replayed next snapshot not genuine/live");
 auto dispatch_before=input_quant;auto hc_before=hc_pending_calls;refused=false;try{owner.run(1,*producer,{replayed[1]},[](const Owner::View&){});}catch(...){refused=true;}auto after_generation=owner.state();check(refused&&!owner.poisoned()&&input_quant==dispatch_before&&hc_pending_calls==hc_before&&owner.generation()==generation_before&&after_generation.position==generation_state.position&&after_generation.keys==generation_state.keys&&after_generation.values==generation_state.values&&after_generation.indexer==generation_state.indexer&&generation_view->output()==generation_output,"replayed generation mixed history or dispatched");
 refused=false;try{owner.reset(generation_state);}catch(...){refused=true;}check(refused&&owner.generation()==generation_before&&generation_view->output()==generation_output,"stale provenance seed admitted");
 owner.reset();owner.run(0,*producer,{replayed[0],replayed[1]},[](const Owner::View&v){check(v.rows()==2,"replay reset rows");});check(owner.position()==2,"coordinated reset did not allow replayed prefix");
 puts("QSA concrete pending composition T2/T1 and rollback passed (CPU component mocks)");return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
