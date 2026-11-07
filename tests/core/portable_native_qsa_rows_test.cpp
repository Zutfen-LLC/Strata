#include "portable_native_qsa_qkv_rows.hpp"
#include "portable_native_qsa_wo_rows.hpp"
#include "portable_native_qsa_indexer_rows.hpp"
#include "ggml-cpu.h"
#include <chrono>
#include <cstdio>
#include <fstream>
using namespace strata::core::portable_detail;
namespace {void need(bool x,const char*m){if(!x)throw std::runtime_error(m);}size_t quant=0,qkv=0,wo=0,bf=0;bool subnormal=false,fail=false;const ggml_tensor* shared=nullptr;}
namespace strata::core::portable_detail {
struct StrataVkNativeStages{};struct StrataVkQ8Wo{};struct StrataVkNativeShared{};
StrataVkNativeStages* strata_vk_native_stages_create_quant_rcp_observation(ggml_backend_t,const char*p,size_t){std::ofstream(p)<<"mock";return new StrataVkNativeStages;}
void strata_vk_native_stages_destroy(StrataVkNativeStages*p)noexcept{delete p;}size_t strata_vk_native_stages_bytes(const StrataVkNativeStages*){return 512;}
StrataVkQ8Wo* strata_vk_qsa_qkv_rows_create(ggml_backend_t,const char*p,size_t){std::ofstream(p)<<"mock";return new StrataVkQ8Wo;}
StrataVkQ8Wo* strata_vk_qsa_wo_rows_create(ggml_backend_t,const char*p,size_t){std::ofstream(p)<<"mock";return new StrataVkQ8Wo;}
void strata_vk_q8_wo_destroy(StrataVkQ8Wo*p)noexcept{delete p;}size_t strata_vk_q8_wo_bytes(const StrataVkQ8Wo*){return 512;}
void quantize(const ggml_tensor*x,ggml_tensor*y,size_t rows,size_t width){++quant;std::vector<float>in(rows*width);ggml_backend_tensor_get(x,in.data(),0,in.size()*4);need(in[0]==3&&(rows==1||in[width]==5),"input row offsets");std::vector<uint8_t>out(rows*width/32*36);if(subnormal)out[rows*width/32*36-36]=1;ggml_backend_tensor_set(y,out.data(),0,out.size());shared=y;}
void strata_vk_qsa_input_quantize_bounded(StrataVkNativeStages*,const ggml_tensor*x,ggml_tensor*y,size_t r){quantize(x,y,r,2560);}
void strata_vk_qsa_wo_quantize_bounded(StrataVkNativeStages*,const ggml_tensor*x,ggml_tensor*y,size_t r){quantize(x,y,r,6144);}
void emit(ggml_tensor*y,size_t n,size_t r){std::vector<float>v(n*r);for(size_t j=0;j<r;++j)std::fill(v.begin()+j*n,v.begin()+(j+1)*n,float(j+1));ggml_backend_tensor_set(y,v.data(),0,v.size()*4);if(fail)throw std::runtime_error("write then fail");}
void strata_vk_qsa_qkv_run_bounded(StrataVkQ8Wo*,const ggml_tensor*,const ggml_tensor*x,ggml_tensor*y,size_t n,size_t r){++qkv;need(x==shared&&(n==12288||n==512),"QKV packed sharing");emit(y,n,r);}
void strata_vk_qsa_wo_run_bounded(StrataVkQ8Wo*,const ggml_tensor*,const ggml_tensor*x,ggml_tensor*y,size_t r){++wo;need(x==shared,"WO packed");emit(y,2560,r);}
StrataVkNativeShared* strata_vk_qsa_indexer_bf16_rows_create(ggml_backend_t,const char*p,size_t){std::ofstream(p)<<"mock";return new StrataVkNativeShared;}
void strata_vk_native_shared_destroy(StrataVkNativeShared*p)noexcept{delete p;}size_t strata_vk_native_shared_bytes(const StrataVkNativeShared*){return 512;}
void strata_vk_native_shared_run(StrataVkNativeShared*,uint32_t mode,const ggml_tensor*w,const ggml_tensor*x,ggml_tensor*y,ggml_tensor*,uint32_t ni,uint32_t no,uint32_t rows){++bf;need(mode==1&&ni==2560&&(no==128||no==512),"BF16 geometry");std::vector<uint16_t>weights(no*ni);std::vector<float>input(rows*ni),output;ggml_backend_tensor_get(w,weights.data(),0,weights.size()*2);ggml_backend_tensor_get(x,input.data(),0,input.size()*4);for(size_t r=0;r<rows;++r){auto v=qsa_indexer_bf16_control(input.data()+r*ni,weights.data(),no);output.insert(output.end(),v.begin(),v.end());}ggml_backend_tensor_set(y,output.data(),0,output.size()*4);if(fail)throw std::runtime_error("write then fail");}
}
int main(int argc,char**argv){try{need(argc==2,"scratch path");std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)>b(ggml_backend_cpu_init(),ggml_backend_free);std::unique_ptr<ggml_context,decltype(&ggml_free)>c(ggml_init({16384,nullptr,true}),ggml_free);
 std::array<ggml_tensor*,3> q{ggml_new_tensor_2d(c.get(),GGML_TYPE_Q8_0,2560,12288),ggml_new_tensor_2d(c.get(),GGML_TYPE_Q8_0,2560,512),ggml_new_tensor_2d(c.get(),GGML_TYPE_Q8_0,2560,512)};auto*w=ggml_new_tensor_2d(c.get(),GGML_TYPE_Q8_0,6144,2560);std::array<ggml_tensor*,2> ix{ggml_new_tensor_1d(c.get(),GGML_TYPE_I32,128*2560/2),ggml_new_tensor_1d(c.get(),GGML_TYPE_I32,512*2560/2)};
 std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>buf(ggml_backend_alloc_ctx_tensors(c.get(),b.get()),ggml_backend_buffer_free);need(bool(buf),"weights");std::array<std::vector<uint8_t>,3> qr;for(size_t i=0;i<3;++i){qr[i].resize(ggml_nbytes(q[i]));ggml_backend_tensor_set(q[i],qr[i].data(),0,qr[i].size());}std::vector<uint8_t>wr(ggml_nbytes(w));ggml_backend_tensor_set(w,wr.data(),0,wr.size());std::array<std::vector<uint8_t>,2>ir;for(size_t i=0;i<2;++i){ir[i].resize(ggml_nbytes(ix[i]));for(size_t row=0;row<(i?512u:128u);++row){ir[i][row*5120]=0x80;ir[i][row*5120+1]=0x3f;}ggml_backend_tensor_set(ix[i],ir[i].data(),0,ir[i].size());}
 auto root=std::filesystem::path(argv[1])/("qsa-rows-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));need(std::filesystem::create_directory(root),"root");
 PortableNativeQsaQkvRows a(b.get(),q,{qr[0].data(),qr[1].data(),qr[2].data()},{qr[0].size(),qr[1].size(),qr[2].size()},(root/"qkv").string(),PortableNativeQsaQkvRows::estimate(b.get()));
 PortableNativeQsaWoRows z(b.get(),w,wr.data(),wr.size(),(root/"wo").string(),PortableNativeQsaWoRows::estimate(b.get()));
 PortableNativeQsaIndexerRows i(b.get(),ix,{ir[0].data(),ir[1].data()},{ir[0].size(),ir[1].size()},(root/"ix").string(),PortableNativeQsaIndexerRows::estimate(b.get()));
 std::vector<float>x(5120);x[0]=3;x[2560]=5;subnormal=true;a.run(x,2);need(a.valid()&&quant==1&&qkv==3,"QKV count/subnormal");i.run(x,2);need(i.valid()&&bf==2,"BF16 counts");for(size_t role=0;role<2;++role){size_t n=role?512:128;std::vector<float>v(2*n);ggml_backend_tensor_get(i.output(role,2),v.data(),0,v.size()*4);for(size_t j=0;j<n;++j)need(v[j]==3.f&&v[n+j]==5.f,"BF16 independent sparse-weight reference");}for(size_t role=0;role<3;++role){size_t n=role?512:12288;std::vector<float>v(2*n);ggml_backend_tensor_get(a.output(role,2),v.data(),0,v.size()*4);need(v[0]==1&&v[n]==2,"QKV row canaries");}
 std::vector<float>g(2*6144);g[0]=3;g[6144]=5;bool refused=false;try{z.run(g,2);}catch(...){refused=true;}need(refused&&!z.valid()&&wo==0,"WO subnormal refusal");subnormal=false;z.run(g,2);need(z.valid()&&wo==1,"WO T2");{std::vector<float>v(5120);ggml_backend_tensor_get(z.output(2),v.data(),0,v.size()*4);need(v[0]==1&&v[2560]==2,"WO row canaries");}fail=true;refused=false;try{a.run(x,2);}catch(...){refused=true;}need(refused&&!a.valid(),"QKV failure invalidation");refused=false;try{i.run(x,2);}catch(...){refused=true;}need(refused&&!i.valid(),"BF16 failure invalidation");refused=false;try{z.run(g,2);}catch(...){refused=true;}need(refused&&!z.valid(),"WO failure invalidation");fail=false;x.resize(2560);a.run(x,1);i.run(x,1);g.resize(6144);z.run(g,1);need(a.valid()&&i.valid()&&z.valid(),"T1 tail");puts("QSA projection row owners: mock dispatch and domains passed");return 0;
}catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
