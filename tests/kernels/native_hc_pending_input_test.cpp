#include "native_hc_pending_fixture.hpp"
#include <limits>
using namespace native_hc_pending_fixture;
template<class F>void refuses(F f){bool bad=false;try{f();}catch(const std::invalid_argument&){bad=true;}native_hc_need(bad,"HC pending: expected refusal");}
int main(){try{
 // Independently pinned exact dyadic product: (1+2^-23)*(1-2^-23)-1=-2^-46.
 const float b=value(0x3f800001),g=value(0x3f7ffffe),r=-1.f;
 native_hc_need(bits(native_hc_pending_fma(b,g,r))==0xa8800000,"pending FMA cancellation golden");
 volatile float product=b*g;native_hc_need(bits(float(product)+r)!=0xa8800000,"golden fails to distinguish separate multiply/add");
 auto f=synthetic("cancel");auto original=f.norm.x;auto out=native_hc_pending_control(f.norm.x,f.bo,f.inj);
 for(float x:out.residual)native_hc_need(bits(x)==0,"exact zero cancellation");
 for(float x:out.gate)native_hc_need(x==1.f,"zero injection gate identity");
 native_hc_need(f.norm.x==original,"control modified pending tuple");
 // Stream/column indexing: bo[d] is shared across all four streams.
 std::vector<float> x(10240),bo(2560),inj(4,0.f);for(size_t d=0;d<2560;++d)bo[d]=float(d%17)*0.03125f;
 for(size_t c=0;c<4;++c)for(size_t d=0;d<2560;++d)x[c*2560+d]=float(c+1);
 auto got=native_hc_pending_control(x,bo,inj);for(size_t c=0;c<4;++c)for(size_t d=0;d<2560;++d)native_hc_need(got.residual[c*2560+d]==float(c+1)+bo[d],"pending independent stream/column golden");
 refuses([&]{auto bad=x;bad.pop_back();native_hc_pending_control(bad,bo,inj);});
 refuses([&]{auto bad=inj;bad[0]=INFINITY;native_hc_pending_control(x,bo,bad);});
 refuses([&]{auto bad=inj;bad[0]=400.f;native_hc_pending_control(x,bo,bad);});
 refuses([&]{auto bad=bo;bad[0]=std::numeric_limits<float>::denorm_min();native_hc_pending_control(x,bad,inj);});
 refuses([&]{native_hc_pending_fma(std::numeric_limits<float>::max(),2.f,1.f);});
 refuses([&]{native_hc_pending_fma(value(0x00800000),value(0x3f7ffffe),-value(0x00800000));});
 // Existing no-pending API remains a refusal; this does not silently widen it.
 refuses([]{native_hc_shape(2560,4,1,1e-6f,true);});
 puts("pending T1 independent FMA/index/zero/lifetime and refusal controls passed");return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
