// Reuse the actual executor and empty-Dispatch HC CPU component mocks.
#define main existing_ple_integration_main
#include "portable_native_ple_integration_test.cpp"
#undef main
#include "portable_native_ple_composition.hpp"
#include "native_component_bound_fake.hpp"

int main() { try {
 using Composition=PortableNativePleComposition; using HC=PortableNativeHcComposed;
 auto throws=[](auto f){bool caught=false;try{f();}catch(const std::exception&){caught=true;}check(caught,"expected exception");};
 std::string error; auto table=ple_fixture::small_table(error); check(bool(table),error.c_str());
 auto b=std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)>(ggml_backend_cpu_init(),ggml_backend_free);
 auto c=std::unique_ptr<ggml_context,decltype(&ggml_free)>(ggml_init({16384,nullptr,true}),ggml_free);
 auto k=ggml_new_tensor_2d(c.get(),GGML_TYPE_Q8_0,2560,10240); auto v=ggml_new_tensor_1d(c.get(),GGML_TYPE_I32,2560*2560/2);
 auto buf=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>(ggml_backend_alloc_ctx_tensors(c.get(),b.get()),ggml_backend_buffer_free);
 std::vector<uint8_t> kr(ggml_nbytes(k)); std::vector<uint16_t> vr(2560*2560),cw(40960,0x3c00); vr[0]=0x3f80; vr[2560]=0x4000;
 ggml_backend_tensor_set(k,kr.data(),0,kr.size()); ggml_backend_tensor_set(v,vr.data(),0,vr.size()*2);
 std::array<std::vector<float>,3> gam{std::vector<float>(10240,1),std::vector<float>(10240,1),std::vector<float>(10240,1)};
 auto root=std::filesystem::temp_directory_path()/("ple-composition-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
 PortableNativePleExecutor::Roles roles{b.get(),table.get(),k,kr.data(),kr.size(),v,vr.data(),vr.size(),{gam[0].data(),gam[1].data(),gam[2].data()},cw.data(),cw.size()*2,1,root.string()};
 auto hctx=std::unique_ptr<ggml_context,decltype(&ggml_free)>(ggml_init({8192,nullptr,true}),ggml_free);
 std::array<const ggml_tensor*,4> src{ggml_new_tensor_1d(hctx.get(),GGML_TYPE_F32,10240),ggml_new_tensor_1d(hctx.get(),GGML_TYPE_I32,20480),ggml_new_tensor_1d(hctx.get(),GGML_TYPE_I32,1638400),ggml_new_tensor_1d(hctx.get(),GGML_TYPE_I32,1638400)};
 auto hbuf=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>(ggml_backend_alloc_ctx_tensors(hctx.get(),b.get()),ggml_backend_buffer_free);
 for(size_t i=0;i<4;++i){hc_roles[i].assign(ggml_nbytes(src[i]),0x31+i);ggml_backend_tensor_set(const_cast<ggml_tensor*>(src[i]),hc_roles[i].data(),0,hc_roles[i].size());}
 std::string prefix=(root/"hc").string(); HC::ComponentCaps caps{113,227,331};
 auto bounds=PortableNativePleAdmission::calculate(b.get(),src,roles.directory,prefix.size(),caps); size_t total=bounds.total;
 check(bounds.hc_host==HC::host_bound(2,prefix.size()) && HC::host_breakdown(2).state==sizeof(HC)+3*512,"factory includes bounded HC shared allocations");
 auto before=executor_creates; auto hb=hc_creates; auto dispatch=quant;
 bound_calls.clear();
 throws([&]{Composition rejected(roles,src,b.get(),caps,PortableNativePleAdmission::estimate_host_bound()-1,prefix);});
 check(bound_calls.empty()&&executor_creates==before&&hc_creates==hb&&!std::filesystem::exists(root),"preflight side effects");
 throws([&]{Composition rejected(roles,src,b.get(),caps,total-1,prefix);});
 check(executor_creates==before&&hc_creates==hb&&quant==dispatch&&!std::filesystem::exists(root),"budget rejected after side effects");
 bound_overflow=true;
 throws([&]{Composition rejected(roles,src,b.get(),caps,SIZE_MAX,prefix);});
 bound_overflow=false;
 check(executor_creates==before&&hc_creates==hb&&quant==dispatch&&!std::filesystem::exists(root),"component overflow after side effects");
 throws([&]{Composition rejected(roles,src,b.get(),{SIZE_MAX,1,1},SIZE_MAX,prefix);});
 auto invalid=roles; invalid.conv_bytes=0; throws([&]{Composition rejected(invalid,src,b.get(),caps,SIZE_MAX,prefix);});
 auto badsrc=src; badsrc[0]=nullptr; throws([&]{Composition rejected(roles,badsrc,b.get(),caps,SIZE_MAX,prefix);});
 check(executor_creates==before&&hc_creates==hb&&quant==dispatch&&!std::filesystem::exists(root),"invalid descriptor side effects");
 Composition comp(roles,src,b.get(),caps,total,prefix);
 check(executor_creates>before&&hc_creates==std::array<int,3>{1,1,1}&&hc_budgets==std::array<size_t,3>{113,227,331},"factory caps");
 // Change the source bank after construction: HC must use its private copies.
 for(auto t:src) fill_hc(const_cast<ggml_tensor*>(t),0);
 comp.reset({3,5}); std::vector<std::array<uint32_t,16>> indices;
 check(table->gather_batch({7,9},{{3,5},{5,7}},indices,expected_embedding,error),error.c_str()); norm_call=0;
 std::vector<float> input(20480,0.25f);
 auto q0=quant; throws([&]{comp.stage(0,{7,-1},input);}); auto nan_input=input;nan_input[0]=std::numeric_limits<float>::quiet_NaN();throws([&]{comp.stage(0,{7,9},nan_input);});check(quant==q0&&comp.position()==0,"invalid stage dispatched");
 auto h=comp.stage(0,{7,9},input); check(quant==q0+1,"T2 dispatch count");
 check(comp.position()==0&&h.predecessors()==std::vector<std::array<int32_t,2>>{{3,5},{5,7}},"stage transaction");
 auto output=h.residual(), history=h.history(); for(float x:output)check(std::isfinite(x),"nonfinite output");
 int consumers=0; auto first=[&](const HC::View& view,uint32_t n){++consumers;check(n==2&&view.read(HC::View::Field::residual)==output,"materialized residual");auto a=view.read(HC::View::Field::normalized),m=view.read(HC::View::Field::mixed);for(size_t r=0;r<2;++r)for(size_t i=0;i<2560;++i)check(std::isfinite(m[r*2560+i])&&m[r*2560+i]==a[r*10240+i]+20,"materialized mix");};
 comp.commit_to_hc(h,first,[&](const HC::View&,uint32_t){++consumers;});
 check(consumers==2&&comp.position()==2&&comp.previous()==std::array<int32_t,2>{7,9}&&comp.history()==history&&hc_runs==std::array<int,3>{0,1,1},"composition commit");
 throws([&]{comp.commit_to_hc(h,first,first);}); comp.reset({3,5}); check(comp.position()==0&&comp.previous()==std::array<int32_t,2>{3,5},"reset"); for(float x:comp.history())check(x==0,"reset history");
 struct Result { std::vector<float> ple, mixed, history; };
 auto run=[&](bool batch){
   comp.reset({3,5}); Result result;
   auto step=[&](std::vector<int32_t> tokens){
     auto pos=comp.position(); auto previous=comp.previous(); std::vector<std::array<int32_t,2>> pred;
     for(auto t:tokens){pred.push_back(previous);previous={previous[1],t};}
     check(table->gather_batch(tokens,pred,indices,expected_embedding,error),error.c_str()); norm_call=0;
     std::vector<float> in(tokens.size()*10240,-987.f);
     for(size_t r=0;r<tokens.size();++r) for(size_t i=0;i<10240;++i) in[r*10240+i]=0.25f+0.25f*(pos+r)+float(i%13)*0.001f;
     auto before_history=comp.history(); auto candidate=comp.stage(pos,tokens,in); auto out=candidate.residual();
     check(candidate.predecessors()==pred&&comp.position()==pos&&comp.history()==before_history,"candidate isolation");
     for(float x:out)check(std::isfinite(x),"output canary"); result.ple.insert(result.ple.end(),out.begin(),out.end());
     comp.commit_to_hc(candidate,[&](const HC::View& view,uint32_t rows){check(rows==tokens.size()&&view.read(HC::View::Field::residual)==out,"batch materialization");auto m=view.read(HC::View::Field::mixed);for(float x:m)check(std::isfinite(x),"mixed canary");result.mixed.insert(result.mixed.end(),m.begin(),m.end());},[](const HC::View&,uint32_t){});
   };
   if(batch){step({7,9});step({11});}else{step({7});step({9});step({11});}
   check(comp.position()==3&&comp.previous()==std::array<int32_t,2>{9,11},"comparison position");result.history=comp.history();return result;
 };
 auto batch=run(true), serial=run(false);check(batch.ple==serial.ple&&batch.mixed==serial.mixed&&batch.history==serial.history,"composition T2+T1 versus T1");
 comp.reset({3,5}); check(table->gather_batch({7,9},{{3,5},{5,7}},indices,expected_embedding,error),error.c_str());norm_call=0;auto pending=comp.stage(0,{7,9},input);
 throws([&]{comp.commit_to_hc(pending,[](const HC::View&,uint32_t){throw std::runtime_error("consumer");},[](const HC::View&,uint32_t){});});check(comp.position()==0,"failed consumer committed");
 throws([&]{comp.commit_to_hc(pending,[](const HC::View&,uint32_t){},[](const HC::View&,uint32_t){});});comp.reset({3,5});check(comp.position()==0,"poison reset");
 std::cout<<"concrete composition CPU mock passed\n";return 0;
 } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;} }
