#include "portable_native_gdn_input_factory.hpp"
#include "ggml-cpu.h"
#include "gdn_f16_test_support.hpp"
#include <chrono>
#include <cstdio>
#include <fstream>
using namespace strata::core::portable_detail;
namespace {
void need(bool ok,const char* m){if(!ok)throw std::runtime_error(m);}
size_t quant_calls=0,dot_calls=0,quant_live=0;bool bad_second=false,fail_dot=false,fail_create=false;
const ggml_tensor* packed_seen=nullptr;
void write_model(const std::filesystem::path& path,const std::array<strata::core::PortableTensorView,2>& views){
 std::vector<uint8_t> header;auto put=[&](const void* p,size_t n){auto b=static_cast<const uint8_t*>(p);header.insert(header.end(),b,b+n);};
 auto u32=[&](uint32_t x){put(&x,4);};auto u64=[&](uint64_t x){put(&x,8);};auto str=[&](const std::string& x){u64(x.size());put(x.data(),x.size());};
 u32(0x46554747);u32(3);u64(2);u64(0);size_t offset=0;
 for(const auto& v:views){str(v.name);u32(2);for(auto d:v.shape)u64(uint64_t(d));u32(v.type);u64(offset);offset=(offset+v.bytes+31)/32*32;}
 header.resize((header.size()+31)/32*32);std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(header.data()),header.size());
 offset=0;for(const auto& v:views){f.seekp(header.size()+offset);f.write(static_cast<const char*>(v.data),v.bytes);offset=(offset+v.bytes+31)/32*32;}need(bool(f),"GGUF fixture write");
}
}
namespace strata::core::portable_detail {
struct StrataVkNativeStages {};struct StrataVkQ8Wo {};
StrataVkNativeStages* strata_vk_native_stages_create_quant_rcp_observation(ggml_backend_t,const char* p,size_t){std::ofstream(p)<<"mock";++quant_live;return new StrataVkNativeStages;}
size_t strata_vk_native_stages_bytes(const StrataVkNativeStages*){return 512;}
void strata_vk_native_stages_destroy(StrataVkNativeStages*p)noexcept{if(p)--quant_live;delete p;}
StrataVkQ8Wo* strata_vk_gdn_input_create(ggml_backend_t,const char* p,size_t){if(fail_create)throw std::runtime_error("injected construction failure");std::ofstream(p)<<"mock";return new StrataVkQ8Wo;}
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
 std::array<strata::core::PortableTensorView,2> views{{
  {"blk.2.attn_qkv.weight",{2560,10240},8,role[0].data(),role[0].size()},
  {"blk.2.attn_gate.weight",{2560,6144},8,role[1].data(),role[1].size()}}};
 auto factory_dir=dir.string()+"-factory";
 auto reject=[&](int layer,const auto& v,size_t host,size_t device){bool failed=false;try{PortableNativeGdnInputOwner bad(layer,backend.get(),v,factory_dir,host,device);}catch(const std::exception&){failed=true;}need(failed&&!std::filesystem::exists(factory_dir),"factory preflight or cleanup");};
 const size_t host=PortableNativeGdnInputOwner::host_bound();
 const size_t device=ggml_backend_buffer_get_size(wb.get())+PortableNativeGdnInputRows::estimate(backend.get());
 reject(3,views,host,device);reject(2,views,host-1,device);reject(2,views,host,1);
 auto bad=views;bad[0].name="blk.4.attn_qkv.weight";reject(2,bad,host,device);
 bad=views;bad[0].type=0;reject(2,bad,host,device);
 bad=views;--bad[1].bytes;reject(2,bad,host,device);
 bad=views;bad[1].shape[1]++;reject(2,bad,host,device);
 fail_create=true;bool failed=false;try{PortableNativeGdnInputOwner partial(2,backend.get(),views,dir.string()+"-partial",host,device);}catch(const std::exception& e){failed=std::string(e.what())=="injected construction failure";}fail_create=false;
 need(failed&&quant_live==0,"partial factory did not release quant owner");
 auto model_path=dir.string()+".gguf";write_model(model_path,views);
 std::unique_ptr<PortableNativeGdnInputOwner> factory;
 { auto model=strata::GgufModel::open(model_path);factory=std::make_unique<PortableNativeGdnInputOwner>(2,backend.get(),model,factory_dir,host,device); }
 need(factory->layer()==2,"factory layer");std::vector<float> factory_input(5120);factory_input[0]=3;factory_input[2560]=5;
 factory->rows().run(factory_input,2);need(factory->rows().valid()&&quant_calls==1&&dot_calls==2,"factory execution after unmapping");
 quant_calls=dot_calls=0;
 PortableNativeGdnInputRows owner(backend.get(),weights,{role[0].data(),role[1].data()},{role[0].size(),role[1].size()},dir.string(),PortableNativeGdnInputRows::estimate(backend.get()));
 std::vector<float> input(5120);input[0]=3;input[2560]=5;owner.run(input,2);need(owner.valid()&&quant_calls==1&&dot_calls==2,"T2 counts");
 for(size_t i=0;i<2;++i){auto*t=owner.output(i,2);size_t n=i?6144:10240;std::vector<float> values(2*n);ggml_backend_tensor_get(t,values.data(),0,values.size()*4);need(values[0]==1&&values[n]==float(n+1),"output row stride");}
 bad_second=true;bool rejected=false;try{owner.run(input,2);}catch(const std::exception&){rejected=true;}need(rejected&&!owner.valid()&&quant_calls==2&&dot_calls==2,"second-row subnormal must reject before either dot");
 bad_second=false;fail_dot=true;rejected=false;try{owner.run(input,2);}catch(const std::exception&){rejected=true;}need(rejected&&!owner.valid()&&dot_calls==3,"write then throw invalidates output");
 fail_dot=false;input.resize(2560);owner.run(input,1);need(owner.valid()&&dot_calls==5&&ggml_nbytes(owner.output(0,1))==10240*4,"T1 tail");
 // Production opt-in F16 projections, with the actual two role shapes.
 for(size_t i=0;i<2;++i){gdn_f16_test::fill(role[i],2560);ggml_backend_tensor_set(weights[i],role[i].data(),0,role[i].size());}
 auto method=PortableGdnProjectionMethod::native_f16;size_t cap=PortableNativeGdnInputRows::estimate(backend.get(),method);
 auto fdir=dir.string()+"-f16";
 bool denied=false;try{PortableNativeGdnInputRows small(backend.get(),weights,{role[0].data(),role[1].data()},{role[0].size(),role[1].size()},fdir,cap-1,method);}catch(const std::exception&){denied=true;}
 need(denied&&!std::filesystem::exists(fdir),"F16 input budget preflight");
 {PortableNativeGdnInputRows dense(backend.get(),weights,{role[0].data(),role[1].data()},{role[0].size(),role[1].size()},fdir,cap,method);
  need(dense.bytes()<=cap,"F16 input admitted bytes");gdn_f16_test::exercise(dense,2560,{10240,6144},[&](size_t i,size_t r){return dense.output(i,r);});}
 std::puts("GDN input row owner and F16 graph passed");return 0;
}catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
