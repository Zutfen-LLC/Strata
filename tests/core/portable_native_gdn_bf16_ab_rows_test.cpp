#include "portable_native_gdn_bf16_ab_rows.hpp"
#include "ggml-cpu.h"
#include <cstdio>
#include <limits>
using namespace strata::core::portable_detail;
static void need(bool x,const char*m){if(!x)throw std::runtime_error(m);}
static std::vector<float> get(ggml_tensor*t){std::vector<float>v(ggml_nbytes(t)/4);ggml_backend_tensor_get(t,v.data(),0,v.size()*4);return v;}
int main(){try{
 std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)>b(ggml_backend_cpu_init(),ggml_backend_free);
 std::unique_ptr<ggml_context,decltype(&ggml_free)>c(ggml_init({16384,nullptr,true}),ggml_free);
 std::array<ggml_tensor*,2>w{ggml_new_tensor_1d(c.get(),GGML_TYPE_I32,61440),ggml_new_tensor_1d(c.get(),GGML_TYPE_I32,61440)};
 std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>buf(ggml_backend_alloc_ctx_tensors(c.get(),b.get()),ggml_backend_buffer_free);
 std::vector<ggml_bf16_t> a(2560*48),bb(a.size());a[0]=ggml_fp32_to_bf16(2);a[47*2560+2559]=ggml_fp32_to_bf16(4);bb[0]=ggml_fp32_to_bf16(-3);bb[47*2560+2559]=ggml_fp32_to_bf16(8);
 ggml_backend_tensor_set(w[0],a.data(),0,a.size()*2);ggml_backend_tensor_set(w[1],bb.data(),0,bb.size()*2);
 auto cap=PortableNativeGdnBf16AbRows::estimate_with_borrowed(b.get(),ggml_backend_buffer_get_size(buf.get()));bool refused=false;try{PortableNativeGdnBf16AbRows bad(b.get(),w,cap-1);}catch(...){refused=true;}need(refused,"short budget accepted");
 auto alias=w;alias[1]=alias[0];refused=false;try{PortableNativeGdnBf16AbRows bad(b.get(),alias,cap);}catch(...){refused=true;}need(refused,"overlap accepted");
 // Refuse the operand itself, even when a large activation could produce a normal result.
 for(uint16_t bits:{uint16_t(0x0001),uint16_t(0x7f80),uint16_t(0x7fc1)}){ggml_backend_tensor_set(w[0],&bits,0,2);refused=false;try{PortableNativeGdnBf16AbRows bad(b.get(),w,cap);}catch(...){refused=true;}need(refused,"invalid BF16 weight accepted");}ggml_backend_tensor_set(w[0],a.data(),0,a.size()*2);
 PortableNativeGdnBf16AbRows p(b.get(),w,cap);
 for(float bad:{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::max(),std::numeric_limits<float>::denorm_min()}){std::vector<float>v(2560);v[0]=bad;refused=false;try{p.run(v,1);}catch(...){refused=true;}need(refused&&!p.valid(),"invalid activation accepted");}
 std::vector<float>x(5120);x[0]=1.00390625f;x[2559]=2;x[2560]=-1.01171875f;x[5119]=-4;auto original=x;
 p.run(x,2);auto aa=get(p.output(0,2)),ba=get(p.output(1,2));need(x==original,"input changed");need(aa[0]==2&&ba[0]==-3&&aa[47]==8&&ba[47]==16,"first row");need(aa[48]==-2.03125f&&ba[48]==3.046875f&&aa[95]==-16&&ba[95]==-32,"second row ties/sentinels");
 std::vector<float>one(2560);one[0]=4;one[2559]=1;p.run(one,1);need(get(p.output(0,1))[0]==8&&get(p.output(1,1))[47]==8,"T1 changed");
 bool stale=false;try{p.output(0,2);}catch(...){stale=true;}need(stale,"stale T2");
 size_t calls=0;ggml_backend_cpu_set_abort_callback(b.get(),[](void*v){++*static_cast<size_t*>(v);return true;},&calls);bool aborted=false;try{p.run(x,2);}catch(...){aborted=true;}ggml_backend_cpu_set_abort_callback(b.get(),nullptr,nullptr);need(aborted&&calls&&!p.valid(),"graph abort");p.run(x,2);need(get(p.output(0,2))==aa&&p.graph_attempts()==4,"T2 retry");
 puts("BF16 AB CPU product graph arithmetic and abort passed");return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
