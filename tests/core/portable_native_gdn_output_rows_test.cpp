#include "portable_native_gdn_output_rows.hpp"
#include "ggml-cpu.h"
#include "gdn_f16_test_support.hpp"
#include <chrono>
#include <cstdio>
#include <fstream>
using namespace strata::core::portable_detail;
namespace {
void need(bool ok,const char* m){if(!ok)throw std::runtime_error(m);}
size_t quant_calls=0,dot_calls=0;bool bad_second=false,bad_nonfinite=false,fail_dot=false;
const ggml_tensor* packed_seen=nullptr;

// A small CPU feasibility check for the F16-weight/F32-RHS operation needed
// by the selected native-prefill method. This is not an owning-layer test or
// a comparison with a native GPU reduction.
void f16_graph_probe(ggml_backend_t backend){
 // Probe conversion primitives with known RN-even bit patterns, including
 // signed Q8_0 codes, a half-subnormal scale, a halfway case and overflow.
 auto decoded=[](uint16_t scale,int code){return ggml_fp32_to_fp16(ggml_fp16_to_fp32(scale)*float(code));};
 need(decoded(0x3c00,-128)==0xd800 && decoded(0x0001,-128)==0x8080,"F16 probe signed code/scale");
 need(ggml_fp32_to_fp16(1.0f+0x1p-11f)==0x3c00 && ggml_fp32_to_fp16(1.0f+0x1.8p-10f)==0x3c02,"F16 probe tie rounding");
 need((decoded(0x7bff,2)&0x7fff)==0x7c00,"F16 probe overflow");
 std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({512u<<10,nullptr,true}),ggml_free);
 need(bool(ctx),"F16 probe metadata");
 auto*w=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F16,32,2);
 ggml_tensor*x[2]{},*y[2]{};ggml_cgraph*g[2]{};
 for(size_t r=1;r<=2;++r){x[r-1]=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,32,r);y[r-1]=ggml_mul_mat(ctx.get(),w,x[r-1]);g[r-1]=ggml_new_graph(ctx.get());
  need(ggml_backend_supports_op(backend,y[r-1]),"F16 probe unsupported");ggml_build_forward_expand(g[r-1],y[r-1]);}
 std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buf(ggml_backend_alloc_ctx_tensors(ctx.get(),backend),ggml_backend_buffer_free);need(bool(buf),"F16 probe allocation");
 // Exact half encodings: -2, +0.5, +1 and -0.25. Remaining terms are zero.
 std::vector<uint16_t> weights(64);weights[0]=0xc000;weights[1]=0x3800;weights[32]=0x3c00;weights[33]=0xb400;
 ggml_backend_tensor_set(w,weights.data(),0,weights.size()*2);
 auto run=[&](size_t rows,float a,float b,float c,float d){
  std::vector<float> input(rows*32);input[0]=a;input[1]=b;if(rows==2){input[32]=c;input[33]=d;}
  ggml_backend_tensor_set(x[rows-1],input.data(),0,input.size()*sizeof(float));
  need(ggml_backend_graph_compute(backend,g[rows-1])==GGML_STATUS_SUCCESS,"F16 probe compute");ggml_backend_synchronize(backend);
  std::vector<float> result(rows*2);ggml_backend_tensor_get(y[rows-1],result.data(),0,result.size()*sizeof(float));
  need(result[0]==-2*a+0.5f*b && result[1]==a-0.25f*b,"F16 probe row 0");
  if(rows==2)need(result[2]==-2*c+0.5f*d && result[3]==c-0.25f*d,"F16 probe row 1");
 };
 run(2,1,2,-3,4);run(1,5,-2,0,0);run(2,-1,6,2,-4);
}
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
 f16_graph_probe(backend.get());
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
 gdn_f16_test::fill(role,6144);ggml_backend_tensor_set(weight,role.data(),0,role.size());
 auto method=PortableGdnProjectionMethod::native_f16;size_t cap=PortableNativeGdnOutputRows::estimate(backend.get(),method);auto fdir=dir.string()+"-f16";
 auto reject=[&](ggml_backend_t b,ggml_tensor*w,const void*p,size_t size,size_t budget){bool denied=false;try{PortableNativeGdnOutputRows bad(b,w,p,size,fdir,budget,method);}catch(const std::exception&){denied=true;}need(denied&&!std::filesystem::exists(fdir),"F16 output preflight");};
 reject(backend.get(),weight,role.data(),role.size(),cap-1);
 reject(backend.get(),weight,role.data(),role.size()-1,cap);
 reject(nullptr,weight,role.data(),role.size(),cap);
 // Source mismatch, nonfinite scale and finite-scale product overflow must be
 // refused before the decoded allocation and receipt directory.
 role[2]^=1;reject(backend.get(),weight,role.data(),role.size(),cap);role[2]^=1;
 uint8_t original0=role[0],original1=role[1],original2=role[2];
 role[0]=0;role[1]=0x7c;ggml_backend_tensor_set(weight,role.data(),0,34);
 reject(backend.get(),weight,role.data(),role.size(),cap);
 role[0]=0xff;role[1]=0x7b;role[2]=2;ggml_backend_tensor_set(weight,role.data(),0,34);
 reject(backend.get(),weight,role.data(),role.size(),cap);
 role[0]=original0;role[1]=original1;role[2]=original2;ggml_backend_tensor_set(weight,role.data(),0,34);
 {PortableNativeGdnOutputRows dense(backend.get(),weight,role.data(),role.size(),fdir,cap,method);need(dense.bytes()<=cap,"F16 output admitted bytes");
  gdn_f16_test::exercise(dense,6144,{2560},[&](size_t,size_t r){return dense.output(r);});}
 std::puts("GDN output row owner/domain mock and F16 graph passed");return 0;
}catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
