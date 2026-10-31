// Actual GDN graph/coordinator/owner with mocked device HC arithmetic. No GPU proof.
#include "portable_gdn_fixture.hpp"
#include "strata_vk_native_hc.hpp"
#include "native_hc_layout.hpp"
#include "ggml-cpu.h"
#include <filesystem>
#include <fstream>
#include <chrono>
using namespace strata::core;
using namespace strata::core::portable_detail;
using namespace gdn_fixture;
namespace {
size_t owners=0,calls=0;bool fail_run=false,fail_create=false;float inject_bias=0,norm_scale=1;
std::vector<float> uploaded,expected_xn,expected_rs,expected_inject;
std::vector<uint16_t> expected_weights;
const ggml_tensor* bound_weight=nullptr;
std::vector<float> get(const ggml_tensor* t){std::vector<float>x(size_t(ggml_nelements(t)));ggml_backend_tensor_get(t,x.data(),0,x.size()*4);return x;}
void put(ggml_tensor*t,const std::vector<float>&x){ggml_backend_tensor_set(t,x.data(),0,x.size()*4);}
}
namespace strata::core::portable_detail {
struct StrataVkNativeHc{};
StrataVkNativeHc* strata_vk_native_hc_create(ggml_backend_t,const char* path,size_t budget){check(!fail_create,"injected create refusal");check(budget>=512,"component budget");std::ofstream(path)<<"mock; not shader evidence\n";++owners;return new StrataVkNativeHc;}
size_t strata_vk_native_hc_bytes(const StrataVkNativeHc*){return 512;}
void strata_vk_native_hc_destroy(StrataVkNativeHc*p)noexcept{if(p)--owners;if(!owners)bound_weight=nullptr;delete p;}
void strata_vk_native_hc_run(StrataVkNativeHc*,const ggml_tensor*x,const ggml_tensor*w,const ggml_tensor*wi,ggml_tensor*xn,ggml_tensor*rs,ggml_tensor*inject,uint32_t t){
 native_hc_environment();check(t==1&&wi->type==GGML_TYPE_I32&&ggml_nbytes(wi)==81920,"native weight/T binding");
 check(!fail_run,"injected HC failure");
 std::vector<uint16_t>weight(40960);ggml_backend_tensor_get(wi,weight.data(),0,81920);check(weight==expected_weights,"BF16 replacement differs from original pack conversion");
 if(bound_weight)check(wi==bound_weight,"weight tensor identity changed");else bound_weight=wi;
 uploaded=get(x);auto out=native_hc_oracle(uploaded,get(w),weight,1);for(float&v:out.inject)v+=inject_bias;for(float&v:out.xn)v*=norm_scale;
 expected_xn=out.xn;expected_rs=out.rs;expected_inject=out.inject;
 put(xn,out.xn);put(rs,out.rs);put(inject,out.inject);++calls;
}
}
int main(){try{
 Fixture f;std::string error;auto root=std::filesystem::temp_directory_path()/("hc-serial-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));check(std::filesystem::create_directory(root),"fresh root");
 for(const auto&v:f.views)if(v.name=="blk.0.hc_attn_inject.weight")expected_weights=native_hc_q8_bf16(static_cast<const uint8_t*>(v.data),v.bytes,40960);
 std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(),ggml_backend_free);
 PortableGdnOptions o;o.execution.threads=1;o.execution.accelerator=cpu.get();o.execution.request_accelerator=true;o.execution.allow_cpu_fallback=false;o.resident_attention_weights=true;
 auto baseline=PortableGdnLayer::create(0,f.views,o,error);check(bool(baseline),error);check(baseline->reset(error),error);PortableRoute route;PortableGdnTrace ordinary;
 check(baseline->step(f.input(0),experts(),route,error,&ordinary),error);const auto disabled=baseline->last_residual();check(!calls&&!ordinary.native_hc_norm_inject,"disabled mode invoked native HC");
 o.native_hc_norm_inject=true;o.native_hc_receipt_prefix=(root/"hc").string();
 // Explicit refusals occur before device arithmetic and do not manufacture a mode.
 for(int invalid=0;invalid<5;++invalid){auto bad=o;if(invalid==0)bad.execution.allow_cpu_fallback=true;if(invalid==1)bad.native_hc_receipt_prefix.clear();if(invalid==2)bad.batch_input_projections=true;if(invalid==3)bad.policy=PortableGdnPolicy::device_production;if(invalid==4)bad.resident_attention_weights=false;check(!PortableGdnLayer::create(0,f.views,bad,error),"invalid native mode accepted");check(!owners,"invalid mode leaked owner");}
 {auto bad=o;bad.execution.weight_budget=baseline->weight_bytes()+38399;check(!PortableGdnLayer::create(0,f.views,bad,error),"converted weight cap ignored");check(!owners,"weight refusal leaked owner");}
 {auto bad=o;bad.execution.graph_budget=1;check(!PortableGdnLayer::create(0,f.views,bad,error),"graph cap ignored");check(!owners,"graph refusal leaked owner");}
 fail_create=true;check(!PortableGdnLayer::create(0,f.views,o,error),"create failure accepted");check(!owners,"failed create leaked owner");fail_create=false;
 o.scratch=PortableGdnScratch::create(o.execution,error);check(bool(o.scratch),error);
 auto layer=PortableGdnLayer::create(0,f.views,o,error);check(bool(layer),error);check(owners==1,"owner count");check(layer->weight_bytes()==baseline->weight_bytes()+38400,"injection replacement accounting (no duplicate Q8 bank)");
 check(layer->reset(error),error);const auto initial=layer->recurrent_state();
 fail_run=true;check(!layer->step(f.input(0),experts(),route,error),"component failure accepted");check(layer->tokens()==0&&layer->recurrent_state()==initial&&!layer->native_hc_calls(),"failed component committed GDN state");fail_run=false;
 check(!layer->prepare_chunk(1,error),"chunk planning silently accepted");
 PortableGdnTrace a;check(layer->step(f.input(0),experts(),route,error,&a),error);
 check(uploaded==f.input(0),"native HC missed current residual upload");check(a.native_hc_norm_inject&&a.hc_residual==uploaded&&a.hc_normalized==expected_xn&&a.hc_rs==expected_rs&&a.hc_inject==expected_inject,"native buffers did not survive graph consumers");
 check(layer->native_hc_calls()==1&&layer->tokens()==1,"successful native counter");const auto first=layer->last_residual();
 check(layer->reset(error),error);PortableGdnTrace repeat;check(layer->step(f.input(0),experts(),route,error,&repeat),error);check(first==layer->last_residual()&&a.hc_inject==repeat.hc_inject,"reset/replay changed result");
 check(layer->reset(error),error);inject_bias=1;check(layer->step(f.input(0),experts(),route,error),error);check(layer->last_residual()!=first,"graph ignored native injection leaf");inject_bias=0;
 check(layer->reset(error),error);norm_scale=2;PortableGdnTrace changed;check(layer->step(f.input(0),experts(),route,error,&changed),error);check(changed.qkv!=a.qkv,"graph ignored native normalized leaf");norm_scale=1;
 check(layer->reset(error),error);check(layer->step(f.input(1),experts(),route,error),error);check(uploaded==f.input(1),"subsequent residual stale");
 layer.reset();check(!owners,"owner not released");
 o.resident_state=true;layer=PortableGdnLayer::create(0,f.views,o,error);check(bool(layer),error);check(layer->reset(error),error);
 check(layer->step(f.input(0),experts(),route,error),error);const auto resident=layer->last_residual();const auto state=layer->recurrent_state();
 check(layer->reset(error),error);check(layer->step(f.input(0),experts(),route,error),error);check(layer->last_residual()==resident&&layer->recurrent_state()==state,"resident state reset/replay changed");layer.reset();check(!owners,"resident owner not released");o.resident_state=false;

 o.native_hc_norm_inject=false;o.scratch.reset();auto off=PortableGdnLayer::create(0,f.views,o,error);check(bool(off),error);check(off->reset(error),error);check(off->step(f.input(0),experts(),route,error),error);check(off->last_residual()==disabled,"default-disabled equivalence lost");
 std::filesystem::remove_all(root);
 puts("serial layer0 HC seam upload/consumption/liveness/reset/accounting/refusal pass; mocked component, no GPU qualification");return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
