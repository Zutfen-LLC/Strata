#include "portable_native_gdn_output_rows.hpp"
#include "ggml-cpu.h"
#include <chrono>
#include <cstdio>
#include <fstream>
using namespace strata::core::portable_detail;
namespace {
void need(bool ok,const char* m){if(!ok)throw std::runtime_error(m);}
size_t quant_calls=0,dot_calls=0;bool bad_second=false,bad_nonfinite=false,fail_dot=false;
const ggml_tensor* packed_seen=nullptr;
}
namespace strata::core::portable_detail {
struct StrataVkNativeStages {};struct StrataVkQ8Wo {};
StrataVkNativeStages* strata_vk_native_stages_create_quant_rcp_observation(ggml_backend_t,const char* p,size_t){std::ofstream(p)<<"mock";return new StrataVkNativeStages;}
size_t strata_vk_native_stages_bytes(const StrataVkNativeStages*){return 512;}
void strata_vk_native_stages_destroy(StrataVkNativeStages*p)noexcept{delete p;}
StrataVkQ8Wo* strata_vk_gdn_output_rows_create(ggml_backend_t,const char* p,size_t){std::ofstream(p)<<"mock";return new StrataVkQ8Wo;}
size_t strata_vk_q8_wo_bytes(const StrataVkQ8Wo*){return 512;}
void strata_vk_q8_wo_destroy(StrataVkQ8Wo*p)noexcept{delete p;}
void strata_vk_gdn_output_quantize_bounded(StrataVkNativeStages*,const ggml_tensor* x,ggml_tensor* q,size_t rows){
 ++quant_calls;need(ggml_nbytes(x)==rows*6144*4&&ggml_nbytes(q)==rows*6912,"quant span");
 std::vector<float> v(rows*6144);ggml_backend_tensor_get(x,v.data(),0,v.size()*4);need(v[0]==3&&(rows==1||v[6144]==5),"input row offsets");
 std::vector<uint8_t> bytes(rows*6912);for(size_t r=0;r<rows;++r)for(size_t b=0;b<192;++b)bytes[r*6912+b*36+4]=uint8_t(17+r);
 if(bad_second&&rows==2)bytes[6912+36]=1; // second row, second block: half 0x0001
 if(bad_nonfinite&&rows==2){bytes[6912+36]=0;bytes[6912+37]=0x7c;}
 ggml_backend_tensor_set(q,bytes.data(),0,bytes.size());packed_seen=q;
}
void strata_vk_gdn_output_run_bounded(StrataVkQ8Wo*,const ggml_tensor*,const ggml_tensor* q,ggml_tensor* y,size_t rows){
 const size_t columns=2560;
 ++dot_calls;need(q==packed_seen&&columns==2560&&ggml_nbytes(y)==rows*columns*4,"dot span/shared packed");
 std::vector<uint8_t> bytes(rows*6912);ggml_backend_tensor_get(q,bytes.data(),0,bytes.size());for(size_t r=0;r<rows;++r)need(bytes[r*6912+4]==17+r,"packed stride");
 std::vector<float> values(rows*columns);for(size_t r=0;r<rows;++r)for(size_t i=0;i<columns;++i)values[r*columns+i]=float(1+r*columns+i);
 ggml_backend_tensor_set(y,values.data(),0,values.size()*4);if(fail_dot)throw std::runtime_error("injected write then throw");
}
}
int main(int argc,char**argv){try{
 need(argc==2,"scratch path");
 need(PortableNativeGdnOutputRows::construction_host_bound()==65536 &&
      PortableNativeGdnOutputRows::runtime_host_bound()==32768+2*6912+2*2560*sizeof(float) &&
      PortableNativeGdnOutputRows::host_bound()>=PortableNativeGdnOutputRows::construction_host_bound() &&
      PortableNativeGdnOutputRows::host_bound()>=PortableNativeGdnOutputRows::runtime_host_bound(),"host phase bounds");
 std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(),ggml_backend_free);
 std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({16384,nullptr,true}),ggml_free);
 auto*weight=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_Q8_0,6144,2560);
 std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> wb(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()),ggml_backend_buffer_free);need(bool(wb),"weight allocation");
 std::vector<uint8_t> role(ggml_nbytes(weight));ggml_backend_tensor_set(weight,role.data(),0,role.size());
 auto dir=std::filesystem::path(argv[1])/("gdn-output-rows-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
 PortableNativeGdnOutputRows owner(backend.get(),weight,role.data(),role.size(),dir.string(),PortableNativeGdnOutputRows::estimate(backend.get()));
 std::vector<float> input(12288);input[0]=3;input[6144]=5;owner.run(input,2);need(owner.valid()&&quant_calls==1&&dot_calls==1,"T2 counts");
 std::vector<float> out(5120);ggml_backend_tensor_get(owner.output(2),out.data(),0,out.size()*4);need(out[0]==1&&out[2560]==2561,"output row stride");
 // GDN output, unlike GDN input, accepts a finite half-subnormal scale.
 bad_second=true;owner.run(input,2);need(owner.valid()&&dot_calls==2,"GDN output subnormal rejected");
 bad_second=false;bad_nonfinite=true;bool rejected=false;try{owner.run(input,2);}catch(const std::exception&){rejected=true;}need(rejected&&!owner.valid()&&dot_calls==2,"second-row nonfinite scale reached dot");
 bad_nonfinite=false;fail_dot=true;rejected=false;try{owner.run(input,2);}catch(const std::exception&){rejected=true;}need(rejected&&!owner.valid()&&dot_calls==3,"write then throw invalidates output");
 fail_dot=false;input.resize(6144);owner.run(input,1);need(owner.valid()&&dot_calls==4&&ggml_nbytes(owner.output(1))==2560*4,"T1 tail");
 std::puts("GDN output row owner/domain mock passed");return 0;
}catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
