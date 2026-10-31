#include "native_hc_write_fixture.hpp"
#include <limits>
using namespace native_hc_write_fixture;
template<class F> void refuses(F f){bool bad=false;try{f();}catch(const std::invalid_argument&){bad=true;}native_hc_need(bad,"write expected refusal");}
int main(){try{
 // Independent dyadic FMA golden: a rounded product loses the negative 2^-46.
 native_hc_need(bits(native_hc_pending_fma(value(0x3f800001),value(0x3f7ffffe),-1.f))==0xa8800000,"write FMA golden");
 // Native scale includes +0, turning scaled negative zero into positive zero.
 native_hc_need(bits(std::fma(.25f,-0.f,0.f))==0,"write positive zero bias");
 std::vector<float> r(10240),b(2560),i(4,-0.f);
 for(size_t c=0;c<4;++c)for(size_t d=0;d<2560;++d)r[c*2560+d]=float(c+1);
 for(size_t d=0;d<2560;++d)b[d]=float(d%17)/32.f;
 const auto original=r,bo=b,inj=i;auto candidate=native_hc_write_control(r,b,i);
 for(float g:candidate.gate)native_hc_need(bits(g)==0x3f800000,"write gate independent identity");
 for(size_t c=0;c<4;++c)for(size_t d=0;d<2560;++d)native_hc_need(candidate.residual[c*2560+d]==float(c+1)+float(d%17)/32.f,"write independent layout golden");
 // Candidate is not committed when the next consumer refuses. No borrowed BO.
 {auto rejected=candidate.residual;rejected.clear();}native_hc_need(r==original&&b==bo&&i==inj,"write rollback changed producer");
 r=candidate.residual;native_hc_need(r==candidate.residual,"write explicit accepted candidate");r=original;
 native_hc_need(native_hc_write_control(r,b,i).residual==candidate.residual,"write reset/replay");
 refuses([&]{auto x=i;x.pop_back();native_hc_write_control(r,b,x);});
 for(float bad:{INFINITY,std::numeric_limits<float>::denorm_min(),400.f})refuses([&]{auto x=i;x[0]=bad;native_hc_write_control(r,b,x);});
 refuses([&]{auto x=b;x[0]=std::numeric_limits<float>::max();auto y=i;y[0]=4.f;native_hc_write_control(r,x,y);});
 puts("write independent FMA/layout/zero-bias/candidate/reset and domain tests passed");return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
