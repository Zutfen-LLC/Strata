// Real GGML allocator, no device mock or GPU execution. This is the QSA
// combine boundary: gated WO input plus an independently materialized R output.
#include "portable_graph.hpp"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include <iostream>
using namespace strata::core;
using namespace strata::core::portable_detail;
int main(){try{
 Backend cpu(ggml_backend_cpu_init(),ggml_backend_free);
 require(bool(cpu),"CPU backend");PortableFfnOptions o;o.graph_budget=256u<<20;
 for(bool gated_leaf:{false,true}){
  std::unique_ptr<ggml_context,decltype(&ggml_free)>c(ggml_init({1u<<20,nullptr,true}),ggml_free);
  auto*w=ggml_new_tensor_2d(c.get(),GGML_TYPE_F32,6144,2560);
  std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>weights(ggml_backend_alloc_ctx_tensors(c.get(),cpu.get()),ggml_backend_buffer_free);
  require(bool(weights),"resident WO weights");ggml_backend_buffer_set_usage(weights.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  auto*x=ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,6144);ggml_set_input(x);
  auto*residual=ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,10240);ggml_set_input(residual);ggml_set_output(residual);
  ggml_tensor*gated=x;
  if(!gated_leaf){auto*gate=ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,6144);ggml_set_input(gate);gated=ggml_mul(c.get(),x,ggml_sigmoid(c.get(),gate));}
  ggml_set_output(gated);auto*wo=ggml_mul_mat(c.get(),w,gated);ggml_set_output(wo);
  auto*graph=ggml_new_graph(c.get());for(auto*t:{wo,gated,residual})ggml_build_forward_expand(graph,t);
  size_t bytes=0;std::vector<PortablePlacement>placement;
  auto sched=schedule(graph,gated_leaf?"qsa-combine-native-gated-leaf":"qsa-combine-control",o,cpu,placement,bytes,true,{wo,gated,residual});
  require(residual->buffer&&gated->buffer&&wo->buffer,"unconsumed residual/output leaf storage missing");
  require(bytes==ggml_backend_sched_get_buffer_size(sched.get(),cpu.get()),"actual allocator bytes not charged");
  require(bytes<=o.graph_budget,"resource cap changed");
  std::cout<<(gated_leaf?"native gated leaf":"materialized gate control")<<" allocated="<<bytes<<"\n";
 }
 // Minimal form of the same bug: output leaf with no computation consumer.
 std::unique_ptr<ggml_context,decltype(&ggml_free)>c(ggml_init({1u<<20,nullptr,true}),ggml_free);
 auto*r=ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,10240);ggml_set_input(r);ggml_set_output(r);auto*g=ggml_new_graph(c.get());ggml_build_forward_expand(g,r);
 size_t bytes=0;std::vector<PortablePlacement>placement;auto sched=schedule(g,"unconsumed-output",o,cpu,placement,bytes,false,{r});
 require(r->buffer&&bytes>=40960,"output leaf omitted from actual allocation");
 std::cout<<"unconsumed output leaf actual allocation admitted and charged\n";
 return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
