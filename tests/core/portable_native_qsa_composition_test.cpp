// Concrete composition/component owners with CPU low-level mocks; no GPU evidence.
#include "portable_native_hc_test_backend.hpp"
#include "portable_qsa_fixture.hpp"
#include "portable_native_qsa_composition.hpp"
#include <optional>
namespace {size_t input_quant=0,qkv_calls=0,wo_quant=0,wo_calls=0;std::vector<float>row_input,row_gated;}
namespace strata::core::portable_detail {
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
 std::vector<std::shared_ptr<const PortableGdnSnapshot>> snapshots;for(int p=0;p<3;++p){check(producer->step(f.input(p),experts,route,error),error);snapshots.push_back(producer->pending_snapshot(error));check(bool(snapshots.back()),error);}
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
 using HC=PortableNativeHcComposed;HC::Dispatch dispatch;float hc_bias=0.f;
 dispatch.pending=[](const HC::Bank&bank){auto v=get(bank.input);put(bank.residual,v);};
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
 std::vector<float>batched,serial;std::vector<int32_t>batch_ids,serial_ids;std::optional<Owner::View>old,failed;
 auto run=[&](size_t first,size_t n,std::vector<float>&out,std::vector<int32_t>&ids){std::vector<std::shared_ptr<const PortableGdnSnapshot>>s(snapshots.begin()+first,snapshots.begin()+first+n);owner.run(first,*producer,s,[&](const Owner::View&v){check(v.first()==first&&v.rows()==n&&v.residual().size()==n*10240&&v.injection().size()==n*4,"view provenance");auto x=v.output();out.insert(out.end(),x.begin(),x.end());for(size_t r=0;r<n;++r){auto z=v.selected(r);ids.insert(ids.end(),z.begin(),z.end());}old=v;});};
 input_quant=qkv_calls=qsa_row_bf_calls=wo_quant=wo_calls=qsa_row_idx_calls=qsa_row_score_calls=qsa_row_select_calls=qsa_row_attn_calls=0;auto before_norm=qsa_norm_calls;run(0,2,batched,batch_ids);check(input_quant==1&&qkv_calls==3&&qsa_row_bf_calls==2&&wo_quant==1&&wo_calls==1&&qsa_norm_calls-before_norm==6&&qsa_row_idx_calls==2&&qsa_row_score_calls==2&&qsa_row_select_calls==2&&qsa_row_attn_calls==2,"T2 real stage counts");run(2,1,batched,batch_ids);auto batch_state=owner.state();check(owner.position()==3&&batched.size()==3*2560,"T2 tail commit");
 {auto prior=old;auto bad=batch_state;bad.indexer.back()=4;bool rejected=false;try{owner.reset(bad);}catch(...){rejected=true;}auto unchanged=owner.state();check(rejected&&unchanged.position==3&&unchanged.indexer==batch_state.indexer&&prior->rows()==1,"invalid seed changed publication/state");owner.reset(batch_state);auto restored=owner.state();check(restored.position==3&&restored.keys==batch_state.keys&&restored.values==batch_state.values&&restored.indexer==batch_state.indexer,"valid seed restore");bool dead=false;try{(void)prior->output();}catch(...){dead=true;}check(dead,"seed reset kept old view");}
 owner.reset();run(0,1,serial,serial_ids);run(1,1,serial,serial_ids);run(2,1,serial,serial_ids);auto serial_state=owner.state();check(batched==serial&&batch_ids==serial_ids&&batch_state.keys==serial_state.keys&&batch_state.values==serial_state.values&&batch_state.indexer==serial_state.indexer,"T2 tail differs from three T1");
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
 owner.reset();std::vector<float>prior_output;std::vector<int32_t>prior_ids;run(0,1,prior_output,prior_ids);auto prior=old;failed.reset();refused=false;try{owner.run(1,*producer,{snapshots[1]},[&](const Owner::View&v){failed=v;check(producer->reset(error),error);});}catch(...){refused=true;}check(refused&&owner.poisoned()&&owner.position()==1&&prior->output()==prior_output,"producer invalidation published candidate");bool dead=false;try{(void)failed->output();}catch(...){dead=true;}check(dead,"producer invalidation retained candidate");
 owner.reset();refused=false;try{owner.run(0,*producer,{snapshots[0]},[](const Owner::View&){});}catch(...){refused=true;}check(refused&&owner.position()==0,"stale snapshots admitted");
 puts("QSA concrete pending composition T2/T1 and rollback passed (CPU component mocks)");return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
