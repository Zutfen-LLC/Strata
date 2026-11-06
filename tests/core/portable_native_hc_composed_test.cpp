#include "portable_native_hc_test_backend.hpp"
#include "portable_qsa_fixture.hpp"
#include "portable_native_hc_composed.hpp"
int main(){try {
 std::string error; std::shared_ptr<ggml_backend> cpu(ggml_backend_cpu_init(),ggml_backend_free);
 qsa_fixture::Fixture f; PortableQsaOptions options; options.capacity=4; options.execution.threads=1; options.execution.accelerator=cpu.get(); options.execution.request_accelerator=true; options.defer_ffn_hc_write=true;
 auto producer=PortableQsaLayer::create(3,f.views,options,error); check(bool(producer),error); check(producer->reset(error),error); PortableRoute route;
 check(producer->step(0,f.input(0),qsa_fixture::experts(),route,error),error); auto s0=producer->pending_snapshot(error);
 check(producer->step(1,f.input(1),qsa_fixture::experts(),route,error),error); auto s1=producer->pending_snapshot(error); check(bool(s0)&&bool(s1),error);
 std::shared_ptr<ggml_context> ctx(ggml_init({8192,nullptr,true}),ggml_free);
 auto nw=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,10240); auto iw=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_I32,20480);
 auto down=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_I32,1638400); auto up=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_I32,1638400);
 std::shared_ptr<ggml_backend_buffer> weights(ggml_backend_alloc_ctx_tensors(ctx.get(),cpu.get()),ggml_backend_buffer_free); check(bool(weights),"weights");
 std::vector<unsigned char> payload(6553600,0x5a); ggml_backend_tensor_set(down,payload.data(),0,payload.size());
 int pending=0,norm=0,mix=0,consumers=0,fail=0; uint32_t expected=2;
 using Owner=PortableNativeHcComposed;
 Owner::Dispatch dispatch;
 dispatch.pending=[&](const Owner::Bank& b){++pending;check(ggml_nbytes(b.input)==expected*40960 && b.input!=b.residual,"pending rows/banks"); std::vector<float> staged(expected*10240); ggml_backend_tensor_get(b.input,staged.data(),0,staged.size()*4); const auto& first=(expected==2?s0:s1)->tuple().residual; check(std::equal(first.begin(),first.end(),staged.begin()),"first staged snapshot"); if(expected==2) check(std::equal(s1->tuple().residual.begin(),s1->tuple().residual.end(),staged.begin()+10240),"second staged snapshot"); if(fail==1)throw std::runtime_error("pending");};
 dispatch.norm=[&](const Owner::Bank& b,uint32_t n){++norm;check(n==expected&&ggml_nbytes(b.inject)==n*16,"norm rows");if(fail==2)throw std::runtime_error("norm");};
 dispatch.mix=[&](const Owner::Bank& b){++mix;check(ggml_nbytes(b.mixed)==expected*10240,"mix rows");if(fail==3)throw std::runtime_error("mix");};
 bool short_budget=false; try { Owner insufficient(cpu.get(),nw,iw,down,up,1,nullptr,dispatch); } catch(...) {short_budget=true;} check(short_budget,"budget accepted");
 bool overlap=false; try { Owner bad(cpu.get(),nw,iw,down,down,20*1024*1024,nullptr,dispatch); } catch(...) {overlap=true;} check(overlap,"overlapping weights accepted");
 Owner owner(cpu.get(),nw,iw,down,up,20*1024*1024,nullptr,dispatch);
 check(owner.bytes()>=13230080 && owner.role_bank()->bytes()>=13230080,"role accounting");
 auto consumer=[&](const Owner::View&,uint32_t n){++consumers;check(n==expected,"consumer rows");if(fail==3+consumers)throw std::runtime_error("consumer");};
 std::shared_ptr<Owner::View> retained, failed, destroyed;
 auto capture=[&](const Owner::View& v,uint32_t n){check(v.rows()==n && v.read(Owner::View::Field::mixed).size()==n*2560,"view read"); retained=std::make_shared<Owner::View>(v);};
 owner.run(4,0,*producer,{s0,s1},capture,consumer); check(pending==1&&norm==1&&mix==1&&consumers==1&&owner.rows()==2,"full batch dispatch");
 auto prior=retained;
 auto first_capture=[&](const Owner::View& v,uint32_t){failed=std::make_shared<Owner::View>(v);};
 auto second_fail=[&](const Owner::View& v,uint32_t){check(v.rows()==2 && prior->rows()==2,"two consumers"); throw std::runtime_error("second consumer");};
 bool second_threw=false; try {owner.run(4,0,*producer,{s0,s1},first_capture,second_fail);} catch(...) {second_threw=true;}
 auto stale=[](const std::shared_ptr<Owner::View>& v){try { (void)v->read(Owner::View::Field::mixed); return false; } catch(const std::runtime_error&) {return true;}};
 check(second_threw && stale(failed) && prior->rows()==2 && prior->read(Owner::View::Field::mixed).size()==5120,"failed second consumer invalidation");
 for(int stage=1;stage<=5;++stage){fail=stage;consumers=0;bool threw=false;try{owner.run(4,0,*producer,{s0,s1},consumer,consumer);}catch(...){threw=true;}check(threw&&owner.rows()==2,"failed publication");}
 fail=0;consumers=0;expected=1;owner.run(4,1,*producer,{s1},capture,consumer);check(owner.rows()==1&&consumers==1 && stale(prior),"tail publication");
 bool rejected=false;try{owner.run(4,0,*producer,{s1},consumer,consumer);}catch(...){rejected=true;}check(rejected&&owner.rows()==1,"snapshot identity");
 bool reset_rejected=false; auto reset_consumer=[&](const Owner::View& v,uint32_t){retained=std::make_shared<Owner::View>(v); try{owner.reset();}catch(...){reset_rejected=true;}};
 owner.run(4,1,*producer,{s1},reset_consumer,consumer); check(reset_rejected,"active reset accepted");
 // Reject a bad second row before dispatch or publication, not just a bad first row.
 expected=2; const int before_pending=pending;
 rejected=false; try { owner.run(4,0,*producer,{s0,s0},consumer,consumer); } catch(...) { rejected=true; }
 check(rejected && pending==before_pending && owner.rows()==1,"row1 failure published or dispatched");
 auto other=PortableQsaLayer::create(3,f.views,options,error); check(bool(other),error); check(other->reset(error),error);
 rejected=false; try { owner.run(4,0,*other,{s0,s1},consumer,consumer); } catch(...) { rejected=true; }
 check(rejected && pending==before_pending && owner.rows()==1,"wrong owner accepted");
 for(int layer : {0,1}) { rejected=false; try { owner.run(layer,0,*producer,{s0,s1},consumer,consumer); } catch(...) { rejected=true; } check(rejected,"layer0/PLE accepted"); }
 check(producer->reset(error),error);
 rejected=false; try { owner.run(4,0,*producer,{s0,s1},consumer,consumer); } catch(...) { rejected=true; }
 check(rejected && pending==before_pending && owner.rows()==1,"stale generation accepted");
 const auto bytes=owner.bytes(); owner.reset(); check(!owner.rows() && owner.bytes()==bytes && stale(retained),"reset publication/scratch");
 { Owner temporary(cpu.get(),nw,iw,down,up,20*1024*1024,nullptr,dispatch);
   expected=1; auto capture_destroyed=[&](const Owner::View& v,uint32_t){destroyed=std::make_shared<Owner::View>(v);};
   // Producer was reset above; use a fresh snapshot.
   check(producer->step(0,f.input(0),qsa_fixture::experts(),route,error),error); s1=producer->pending_snapshot(error);
   temporary.run(4,0,*producer,{s1},capture_destroyed,consumer); check(destroyed->rows()==1,"live temporary"); }
 check(stale(destroyed),"destruction left view valid");
 // Exercise the same composed control path with real GDN-issued snapshots.
 Fixture gf; auto views=gf.views; for(auto& v:views) v.name.replace(0,6,"blk.1.");
 PortableGdnOptions go; go.execution=options.execution; go.defer_ffn_hc_write=true;
 auto gp=PortableGdnLayer::create(1,views,go,error); check(bool(gp),error); check(gp->reset(error),error);
 auto gexperts=[](int64_t, const PortableRoute& r, std::vector<float>& rows, std::string& e){return experts()(0,r,rows,e);};
 check(gp->step(gf.input(0),gexperts,route,error),error); auto g0=gp->pending_snapshot(error);
 check(gp->step(gf.input(1),gexperts,route,error),error); auto g1=gp->pending_snapshot(error); check(bool(g0)&&bool(g1),error);
 int calls=0, gnorm=0, gmix=0; Owner::Dispatch gd;
 gd.pending=[&](const Owner::Bank& bank){++calls; std::vector<float> x(20480); ggml_backend_tensor_get(bank.input,x.data(),0,x.size()*4); check(std::equal(g0->tuple().residual.begin(),g0->tuple().residual.end(),x.begin()) && std::equal(g1->tuple().residual.begin(),g1->tuple().residual.end(),x.begin()+10240),"GDN staged rows");};
 gd.norm=[&](const Owner::Bank&,uint32_t n){++gnorm; check(n==2,"GDN rows");};
 gd.mix=[&](const Owner::Bank& bank){++gmix; std::vector<float> mixed(5120,1.25f); std::fill(mixed.begin()+2560,mixed.end(),-2.5f); ggml_backend_tensor_set(bank.mixed,mixed.data(),0,mixed.size()*sizeof(float));};
 Owner gowner(cpu.get(),nw,iw,down,up,20*1024*1024,nullptr,gd); int gc=0; auto consume_g=[&](const Owner::View&,uint32_t n){check(n==2,"GDN consumer"); ++gc;};
 weights.reset(); ctx.reset();
 std::vector<unsigned char> copied(payload.size()); ggml_backend_tensor_get(gowner.role_bank()->role(2),copied.data(),0,copied.size()); check(copied==payload,"role snapshot after source destruction");
 std::shared_ptr<Owner::View> good_g;
 auto capture_g=[&](const Owner::View& v,uint32_t n){consume_g(v,n); good_g=std::make_shared<Owner::View>(v);};
 gowner.run(2,0,*gp,{g0,g1},capture_g,consume_g); check(calls==1&&gnorm==1&&gmix==1&&gc==2&&gowner.rows()==2,"GDN composed run");
 const auto good_values=good_g->read(Owner::View::Field::mixed);
 std::vector<float> expected_values(5120,1.25f); std::fill(expected_values.begin()+2560,expected_values.end(),-2.5f);
 check(good_values==expected_values,"GDN mixed canaries");
 check(g1->valid_for(*gp,2,1),"committed GDN snapshot invalid");
 auto chunk_rows=gf.input(2); const auto original_rows=chunk_rows; std::vector<PortableRoute> chunk_routes; bool chunk_dispatch=false;
 check(!gp->run_chunk({2},gp->tokens(),chunk_rows,[&](int64_t,const auto&,auto&,auto&){chunk_dispatch=true; return false;},chunk_routes,error)
       && gp->poisoned() && !chunk_dispatch && chunk_rows==original_rows && chunk_routes.empty()
       && error=="native HC serial: chunk API unsupported","deferred GDN chunk rejection");
 check(!g1->valid_for(*gp,2,1) && !gp->pending_snapshot(error),"poisoned GDN snapshot remained valid");
 rejected=false; try {gowner.run(2,1,*gp,{g1},consume_g,consume_g);} catch(...) {rejected=true;}
 check(rejected && calls==1 && gnorm==1 && gmix==1 && gc==2 && gowner.rows()==2 &&
       good_g->rows()==2 && good_g->read(Owner::View::Field::mixed)==good_values,"poisoned GDN dispatched or changed publication");
 check(gp->reset(error),error); rejected=false; try {gowner.run(2,0,*gp,{g0,g1},consume_g,consume_g);} catch(...) {rejected=true;} check(rejected&&calls==1&&gowner.rows()==2,"GDN stale generation");
 puts("composed HC CPU control path passed");return 0;
 }catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
