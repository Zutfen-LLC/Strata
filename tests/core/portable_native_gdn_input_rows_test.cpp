#include "portable_native_gdn_input_rows.hpp"
#include "ggml-cpu.h"
#include <chrono>
#include <cstdio>
#include <fstream>
using namespace strata::core::portable_detail;
namespace {
void need(bool ok,const char* m){if(!ok)throw std::runtime_error(m);}
size_t quant_calls=0,dot_calls=0;bool bad_second=false,fail_dot=false;
const ggml_tensor* packed_seen=nullptr;
}
namespace strata::core::portable_detail {
struct StrataVkNativeStages {};struct StrataVkQ8Wo {};
StrataVkNativeStages* strata_vk_native_stages_create_quant_rcp_observation(ggml_backend_t,const char* p,size_t){std::ofstream(p)<<"mock";return new StrataVkNativeStages;}
size_t strata_vk_native_stages_bytes(const StrataVkNativeStages*){return 512;}
void strata_vk_native_stages_destroy(StrataVkNativeStages*p)noexcept{delete p;}
StrataVkQ8Wo* strata_vk_gdn_input_create(ggml_backend_t,const char* p,size_t){std::ofstream(p)<<"mock";return new StrataVkQ8Wo;}
size_t strata_vk_q8_wo_bytes(const StrataVkQ8Wo*){return 512;}
void strata_vk_q8_wo_destroy(StrataVkQ8Wo*p)noexcept{delete p;}
void strata_vk_gdn_input_quantize_bounded(StrataVkNativeStages*,const ggml_tensor* x,ggml_tensor* q,size_t rows){
 ++quant_calls;need(ggml_nbytes(x)==rows*2560*4&&ggml_nbytes(q)==rows*2880,"quant span");
 std::vector<float> v(rows*2560);ggml_backend_tensor_get(x,v.data(),0,v.size()*4);need(v[0]==3&&(rows==1||v[2560]==5),"input row offsets");
 std::vector<uint8_t> bytes(rows*2880);for(size_t r=0;r<rows;++r)for(size_t b=0;b<80;++b)bytes[r*2880+b*36+4]=uint8_t(17+r);
 if(bad_second&&rows==2)bytes[2880+36]=1; // second row, second block: half 0x0001
 ggml_backend_tensor_set(q,bytes.data(),0,bytes.size());packed_seen=q;
}
void strata_vk_gdn_input_run_bounded(StrataVkQ8Wo*,const ggml_tensor*,const ggml_tensor* q,ggml_tensor* y,size_t columns,size_t rows){
 ++dot_calls;need(q==packed_seen&&(columns==10240||columns==6144)&&ggml_nbytes(y)==rows*columns*4,"dot span/shared packed");
 std::vector<uint8_t> bytes(rows*2880);ggml_backend_tensor_get(q,bytes.data(),0,bytes.size());for(size_t r=0;r<rows;++r)need(bytes[r*2880+4]==17+r,"packed stride");
 std::vector<float> values(rows*columns);for(size_t r=0;r<rows;++r)for(size_t i=0;i<columns;++i)values[r*columns+i]=float(1+r*columns+i);
 ggml_backend_tensor_set(y,values.data(),0,values.size()*4);if(fail_dot)throw std::runtime_error("injected write then throw");
}
}
int main(int argc,char**argv){try{
 need(argc==2,"scratch path");std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(),ggml_backend_free);
 std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({16384,nullptr,true}),ggml_free);
 std::array<ggml_tensor*,2> weights{ggml_new_tensor_2d(ctx.get(),GGML_TYPE_Q8_0,2560,10240),ggml_new_tensor_2d(ctx.get(),GGML_TYPE_Q8_0,2560,6144)};
 std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> wb(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()),ggml_backend_buffer_free);need(bool(wb),"weight allocation");
 std::array<std::vector<uint8_t>,2> role;for(size_t i=0;i<2;++i){role[i].resize(ggml_nbytes(weights[i]));ggml_backend_tensor_set(weights[i],role[i].data(),0,role[i].size());}
 auto dir=std::filesystem::path(argv[1])/("gdn-input-rows-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
 PortableNativeGdnInputRows owner(backend.get(),weights,{role[0].data(),role[1].data()},{role[0].size(),role[1].size()},dir.string(),PortableNativeGdnInputRows::estimate(backend.get()));
 std::vector<float> input(5120);input[0]=3;input[2560]=5;owner.run(input,2);need(owner.valid()&&quant_calls==1&&dot_calls==2,"T2 counts");
 for(size_t i=0;i<2;++i){auto*t=owner.output(i,2);size_t n=i?6144:10240;std::vector<float> values(2*n);ggml_backend_tensor_get(t,values.data(),0,values.size()*4);need(values[0]==1&&values[n]==float(n+1),"output row stride");}
 bad_second=true;bool rejected=false;try{owner.run(input,2);}catch(const std::exception&){rejected=true;}need(rejected&&!owner.valid()&&quant_calls==2&&dot_calls==2,"second-row subnormal must reject before either dot");
 bad_second=false;fail_dot=true;rejected=false;try{owner.run(input,2);}catch(const std::exception&){rejected=true;}need(rejected&&!owner.valid()&&dot_calls==3,"write then throw invalidates output");
 fail_dot=false;input.resize(2560);owner.run(input,1);need(owner.valid()&&dot_calls==5&&ggml_nbytes(owner.output(0,1))==10240*4,"T1 tail");
 std::puts("GDN input row owner and second-row scale refusal passed");return 0;
}catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
