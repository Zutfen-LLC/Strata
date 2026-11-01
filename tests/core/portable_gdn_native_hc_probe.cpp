// Physical seam probe: approved caller-supplied HC inputs plus existing synthetic
// full-GDN fixture. No model, new reference kernel or full-HC parity claim.
#include "portable_gdn_fixture.hpp"
#include "../kernels/native_hc_fixture.hpp"
#include "ggml-vulkan.h"
#include <filesystem>
#include <fstream>
using namespace strata::core;
using namespace gdn_fixture;
namespace {
std::vector<uint8_t> load(const std::filesystem::path&p,size_t n){check(std::filesystem::file_size(p)==n,"bounded operand size");std::ifstream f(p,std::ios::binary);std::vector<uint8_t>x(n);check(bool(f.read(reinterpret_cast<char*>(x.data()),n)),"operand read");return x;}
void save(const std::filesystem::path&p,const std::vector<float>&x){check(!std::filesystem::exists(p),"fresh output");native_hc_fixture::floats(p.string(),x);}
struct Result{std::vector<float> residual,state,history;};
Result run(const Fixture&f,ggml_backend_t backend,bool enabled,const std::vector<float>&input,const std::filesystem::path&out){
 PortableGdnOptions o;o.execution.threads=1;o.execution.accelerator=backend;o.execution.request_accelerator=true;o.execution.allow_cpu_fallback=false;o.resident_attention_weights=true;o.resident_state=true;
 std::string error;o.scratch=PortableGdnScratch::create(o.execution,error);check(bool(o.scratch),error);
 // Warm the reusable scheduler with a normal graph before borrowing it for HC.
 auto views=f.views;for(auto&v:views)v.name.replace(0,6,"blk.4.");
 auto warm=PortableGdnLayer::create(4,views,o,error);check(bool(warm),error);
 o.native_hc_norm_inject=enabled;o.native_hc_receipt_prefix=(out/"hc").string();
 auto layer=PortableGdnLayer::create(0,f.views,o,error);check(bool(layer),error);check(layer->reset(error),error);
 PortableRoute route;PortableGdnTrace a;
 check(layer->step(input,experts(),route,error,&a),error);
 check(layer->native_hc_calls()==(enabled?1u:0u),"actual HC participation");
 auto first=layer->last_residual(),state=layer->recurrent_state(),history=layer->convolution_history();
 if(enabled){
  check(a.native_hc_norm_inject&&a.hc_residual==input,"post-graph input preservation/upload");
  save(out/"normalized.f32le",a.hc_normalized);save(out/"rs.f32le",a.hc_rs);save(out/"inject.f32le",a.hc_inject);
 }
 check(layer->reset(error),error);PortableGdnTrace b;
 check(layer->step(input,experts(),route,error,&b),error);
 check(first==layer->last_residual()&&state==layer->recurrent_state()&&history==layer->convolution_history(),"reset/replay state/residual differs");
 if(enabled)check(a.hc_normalized==b.hc_normalized&&a.hc_rs==b.hc_rs&&a.hc_inject==b.hc_inject,"repeat HC differs");
 check(layer->reset(error),error);const auto before=layer->recurrent_state();
 PortableExpertRows refusal=[](int64_t,const PortableRoute&,std::vector<float>&,std::string&e){e="intentional fixture provider refusal";return false;};
 check(!layer->step(input,refusal,route,error),"provider failure accepted");check(layer->tokens()==0&&layer->last_residual().empty()&&layer->recurrent_state()==before,"provider failure committed state");
 check(layer->reset(error),error);
 printf("{\"native_layer0\":%s,\"actual_component_calls\":%llu,\"weight_bytes\":%zu,\"graph_bytes\":%zu,\"reset_repeat_exact\":true,\"provider_rollback\":true}\n",enabled?"true":"false",(unsigned long long)layer->native_hc_calls(),layer->weight_bytes(),layer->graph_bytes());
 return {first,state,history};
}
}
int main(int argc,char**argv){try{
 check(argc==4,"usage: portable_gdn_native_hc_probe OPERANDS FRESH_OUTPUT EXPECT_DEVICE");const std::filesystem::path d=argv[1],out=argv[2];check(std::filesystem::create_directory(out),"fresh output directory");
 auto norm=load(d/"portable-hc_attn_norm.weight.bin",40960),inj=load(d/"portable-hc_attn_inject.weight.bin",43520),raw=load(d/"portable-embedding.f32le",40960);
 std::vector<float>input(10240);memcpy(input.data(),raw.data(),40960);Fixture f;
 for(auto&v:f.views){if(v.name=="blk.0.hc_attn_norm.weight")v.data=norm.data();if(v.name=="blk.0.hc_attn_inject.weight")v.data=inj.data();}
 check(ggml_backend_vk_get_device_count()>0,"real Vulkan required");char desc[256]{};ggml_backend_vk_get_device_description(0,desc,sizeof(desc));check(std::string(desc).find(argv[3])!=std::string::npos,"device identity");
 std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)>backend(ggml_backend_vk_init(0),ggml_backend_free);check(bool(backend),"Vulkan initialization");
 const auto active=run(f,backend.get(),true,input,out);
 std::filesystem::create_directory(out/"disabled-a");std::filesystem::create_directory(out/"disabled-b");
 const auto a=run(f,backend.get(),false,input,out/"disabled-a"),b=run(f,backend.get(),false,input,out/"disabled-b");
 check(a.residual==b.residual&&a.state==b.state&&a.history==b.history,"disabled repeated path changed");
 native_hc_fixture::Fixture fixture;fixture.t=1;fixture.x=input;fixture.w.resize(10240);memcpy(fixture.w.data(),norm.data(),40960);fixture.wi=strata::core::portable_detail::native_hc_q8_bf16(inj.data(),inj.size(),40960);
 native_hc_fixture::write_fixture((out/"fixture.bin").string(),fixture);
 std::ofstream report(out/"report.json");report<<"{\"tokens\":1,\"guards\":true,\"finite_normal_domain\":true,\"repeat_exact\":true,\"qualified\":false,\"serial_layer0_integration\":true,\"full_hc_parity\":false,\"reset_repeat_exact\":true,\"provider_rollback\":true,\"disabled_repeat_exact\":true}\n";
 return 0;
}catch(const std::exception&e){fprintf(stderr,"STOP: %s\n",e.what());return 1;}}
