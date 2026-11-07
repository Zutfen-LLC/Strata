#include "portable_native_hc_bf16_products.hpp"
#include "ggml-cpu.h"
#include <cstdio>
#include <limits>
using namespace strata::core::portable_detail;
static void need(bool v,const char*m){if(!v)throw std::runtime_error(m);}
template<class F> static bool fails(F f){try{f();return false;}catch(...){return true;}}
static std::vector<float> get(ggml_tensor*t){std::vector<float>v(ggml_nbytes(t)/4);ggml_backend_tensor_get(t,v.data(),0,v.size()*4);return v;}
int main(){try{
 using P=PortableNativeHcBf16Products;
 std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> b(ggml_backend_cpu_init(),ggml_backend_free);
 std::unique_ptr<ggml_context,decltype(&ggml_free)> c(ggml_init({16384,nullptr,true}),ggml_free);
 std::array<ggml_tensor*,3>w{ggml_new_tensor_1d(c.get(),GGML_TYPE_I32,1638400),ggml_new_tensor_1d(c.get(),GGML_TYPE_I32,1638400),ggml_new_tensor_1d(c.get(),GGML_TYPE_I32,20480)};
 auto* image=ggml_new_tensor_2d(c.get(),GGML_TYPE_F32,10240,2);auto* low_image=ggml_new_tensor_2d(c.get(),GGML_TYPE_F32,320,2);
 std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buf(ggml_backend_alloc_ctx_tensors(c.get(),b.get()),ggml_backend_buffer_free);
 // Literal BF16 weights; no production rounding helper is used as an oracle.
 std::vector<uint16_t> d(10240*320),u(320*10240),inj(10240*4);
 d[0]=0x4000;d.back()=0x4080;u[0]=0xc040;u.back()=0x4100;inj[0]=0x3f80;inj.back()=0xc000;
 ggml_backend_tensor_set(w[0],d.data(),0,d.size()*2);ggml_backend_tensor_set(w[1],u.data(),0,u.size()*2);ggml_backend_tensor_set(w[2],inj.data(),0,inj.size()*2);
 std::array<const ggml_tensor*,3> roles{w[0],w[1],w[2]};size_t cap=P::device_bound(b.get(),ggml_backend_buffer_get_size(buf.get()));
 need(fails([&]{P x(b.get(),roles,P::host_bound()-1,cap);}),"host budget");
 need(fails([&]{P x(b.get(),roles,P::host_bound(),cap-1);}),"device budget");
 auto alias=roles;alias[1]=roles[0];need(fails([&]{P x(b.get(),alias,P::host_bound(),cap);}),"overlap");
 auto wrong=roles;wrong[0]=roles[2];need(fails([&]{P x(b.get(),wrong,P::host_bound(),cap);}),"role extent");
 auto original_type=w[0]->type;w[0]->type=GGML_TYPE_F32;need(fails([&]{P x(b.get(),roles,P::host_bound(),cap);}),"role type");w[0]->type=original_type;
 auto original_data=w[0]->data;w[0]->data=reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(buf.get()))+ggml_backend_buffer_get_size(buf.get())-4);need(fails([&]{P x(b.get(),roles,P::host_bound(),cap);}),"role span");w[0]->data=original_data;
 for(uint16_t bits:{uint16_t(1),uint16_t(0x7f80),uint16_t(0x7fc1)}){ggml_backend_tensor_set(w[0],&bits,0,2);need(fails([&]{P x(b.get(),roles,P::host_bound(),cap);}),"weight domain");}
 ggml_backend_tensor_set(w[0],d.data(),0,d.size()*2);
 P p(b.get(),roles,P::host_bound(),cap);
 // Graph weights are independent copies of the immutable source bank.
 uint16_t zero=0;ggml_backend_tensor_set(w[0],&zero,0,2);
 std::vector<float>x(20480),low(640);x[0]=1.00390625f;x[10239]=2;x[10240]=-1.01171875f;x[20479]=-4;
 // Already-computed SiLU values: this test does not exercise SiLU.
 low[0]=1.00390625f;low[319]=2;low[320]=-1.01171875f;low[639]=-4;auto oldx=x,oldlow=low;
 p.down(x,2);auto down=get(p.output(0));need(down[0]==2&&down[319]==8&&down[320]==-2.03125f&&down[639]==-16,"down T2");
 p.up(low);need(fails([&]{p.output(0);}),"stale down");auto up=get(p.output(1));need(up[0]==-3&&up[10239]==16&&up[10240]==3.046875f&&up[20479]==-32,"up T2");
 p.inject();auto iv=get(p.output(2));need(iv[0]==1&&iv[3]==-4&&iv[4]==-1.015625f&&iv[7]==8,"inject T2");need(x==oldx&&low==oldlow,"input changed");
 std::vector<float>one(10240),lo(320);one[0]=4;one.back()=1;lo[0]=2;lo.back()=3;
 p.down(one,1);need(get(p.output(0))[0]==8,"T1 down");p.up(lo);need(get(p.output(1)).back()==24,"T1 up");p.inject();need(get(p.output(2)).back()==-2,"T1 inject");
 x[0]=2;x[10240]=4;p.down(x,2);need(get(p.output(0))[0]==4&&get(p.output(0))[320]==8,"changed T2");p.up(low);p.inject();
 for(float bad:{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::max(),std::numeric_limits<float>::denorm_min()}){one[0]=bad;need(fails([&]{p.down(one,1);})&&fails([&]{p.output(2);}),"invalid input/stale output");}
 one[0]=4;need(fails([&]{p.up(lo);}),"order refusal");
 size_t calls=0;ggml_backend_cpu_set_abort_callback(b.get(),[](void*v){++*static_cast<size_t*>(v);return true;},&calls);need(fails([&]{p.down(x,2);})&&calls&&fails([&]{p.output(0);}),"graph abort");ggml_backend_cpu_set_abort_callback(b.get(),nullptr,nullptr);
 p.down(x,2);p.up(low);p.inject();need(get(p.output(2))[0]==2,"recovery");
 p.down(x,2);ggml_backend_cpu_set_abort_callback(b.get(),[](void*v){++*static_cast<size_t*>(v);return true;},&calls);need(fails([&]{p.up(low);})&&fails([&]{p.output(0);}),"up abort/stale");ggml_backend_cpu_set_abort_callback(b.get(),nullptr,nullptr);
 p.down(x,2);p.up(low);p.inject();need(get(p.output(2))[4]==4,"up abort recovery");
 p.down(x,2);auto badlow=low;badlow[0]=std::numeric_limits<float>::infinity();need(fails([&]{p.up(badlow);})&&fails([&]{p.output(0);}),"low domain/stale");
 p.down(x,2);p.up(low);ggml_backend_cpu_set_abort_callback(b.get(),[](void*v){++*static_cast<size_t*>(v);return true;},&calls);need(fails([&]{p.inject();})&&fails([&]{p.output(1);}),"inject abort/stale");ggml_backend_cpu_set_abort_callback(b.get(),nullptr,nullptr);p.down(x,2);p.up(low);p.inject();need(get(p.output(2))[4]==4,"inject abort recovery");
 // Widened BF16 images model scalar-shader output without exercising its arithmetic.
 std::vector<float> xi(20480),li(640);xi[0]=1;xi[10239]=2;xi[10240]=-1.015625f;xi[20479]=-4;li[0]=1;li[319]=2;li[320]=-1.015625f;li[639]=-4;
 ggml_backend_tensor_set(image,xi.data(),0,xi.size()*4);ggml_backend_tensor_set(low_image,li.data(),0,li.size()*4);
 p.down_image(image,2);need(get(p.output(0))==down,"image down");p.up_image(low_image);need(get(p.output(1))==up,"image up");p.inject();need(get(p.output(2))==iv,"image injection");
 xi[0]=1.00390625f;ggml_backend_tensor_set(image,xi.data(),0,xi.size()*4);need(fails([&]{p.down_image(image,2);})&&fails([&]{p.output(2);}),"unrounded image refusal");
 xi[0]=1;ggml_backend_tensor_set(image,xi.data(),0,xi.size()*4);p.down_image(image,2);li[0]=std::numeric_limits<float>::infinity();ggml_backend_tensor_set(low_image,li.data(),0,li.size()*4);need(fails([&]{p.up_image(low_image);})&&fails([&]{p.output(0);}),"low image domain/stale");
 li[0]=1;ggml_backend_tensor_set(low_image,li.data(),0,li.size()*4);p.down_image(image,2);p.up_image(low_image);p.inject();need(get(p.output(2))==iv,"image recovery");
 p.down_image(image,2);auto* internal=p.output(0);need(fails([&]{p.up_image(internal);})&&fails([&]{p.output(0);}),"internal image alias/stale");
 puts("HC BF16 CPU product graphs passed (SiLU and device arithmetic not tested)");return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
