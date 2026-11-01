#include "native_gdn_t1_fixture.hpp"
using namespace native_gdn_t1;
int main(int argc,char**argv){if(argc==3&&std::string(argv[1])=="--validate-fixture"){(void)native_gdn_t1::read(argv[2]);puts("GDN complete immutable fixture input domain pass; actual primitive outputs still require checking");return 0;}if(argc==4&&(std::string(argv[1])=="--check"||std::string(argv[1])=="--check-model")){
 bool model=std::string(argv[1])=="--check-model";
 auto f=native_gdn_t1::read(argv[2]);std::filesystem::path d=argv[3];auto get=[&](const char*name,size_t n){auto p=d/(std::string(name)+".f32le");native_hc_need(std::filesystem::file_size(p)==n*4,"GDN result shape");std::ifstream in(p,std::ios::binary);std::vector<float>v(n);for(float&x:v)x=value(word(in));return v;};
 auto ref=control(f,get("decay-exp",48),get("rs",49),get("sigmoid",6144));exact(ref.state,get("next-state",786432),"GDN independent own-exp state");exact(ref.oc,get("oc",6144),"GDN output reduction");if(model){auto path=d/"derived-norm-arg.f32le";native_hc_need(!std::filesystem::exists(path),"GDN fresh derived norm argument");floats(path.string(),ref.arg);}else exact(ref.arg,get("norm-arg",48),"GDN norm argument");exact(ref.y,get("y",6144),"GDN own-scale/sigmoid output");puts("GDN source-order offline control pass");return 0;
 }native_hc_need(argc==1,"GDN host control arguments");auto f=native_gdn_t1::synthetic(1);std::vector<float>eg(48,1),rs(49,1),sig(6144,.5f);rs[48]=value(0x3db504f3);auto out=control(f,eg,rs,sig);
 // Pinned hand calculation: k0=.25,v0=2,beta=.5 -> state(row0,head0,col0)=.25.
 native_hc_need(bits(out.state[0])==0x3e800000&&bits(out.state[1])==0,"GDN independent sparse state vector");native_hc_need(bits(out.oc[0])==0x3c3504f3&&bits(out.y[0])==0x3bb504f3,"GDN independently pinned output order");
 // Dense independent integer/dyadic oracle pins row/head/column indexing and
 // all four 32-row groups. Intrinsic controls deliberately set exp=1 here.
 auto dense=native_gdn_t1::synthetic(2);auto d=control(dense,eg,rs,sig);
 const size_t index[]={0,6145,192737,198881,393215,393612,586560,786431};
 const uint32_t expected[]={0xbbc37170,0x3bc53dc0,0xbba85580,0x3b75eaa0,0xbb8e2c80,0x3b5dcbc0,0x3bdd0480,0x3b20bb00};
 for(size_t i=0;i<8;++i)native_hc_need(bits(d.state[index[i]])==expected[i],"GDN independent dense row-group/indexing vector");
 auto wrong=out.state;wrong[1]=1;bool rejected=false;try{exact(wrong,out.state,"bad identical-backend output");}catch(...){rejected=true;}native_hc_need(rejected,"GDN wrong common result accepted");
 auto underflow=native_gdn_t1::synthetic(1);underflow[2][23]=-138.10023498535156f;validate(underflow);
 // Derived class boundaries, not fitted to a model extremum. Every finite
 // nonpositive normal decay log is preflightable; actual subnormal outputs refuse.
 for(float x:{-87.33651733398438f,-88.f,-104.f,-128.f,-138.10023498535156f,-std::numeric_limits<float>::max(),-0.f})native_gdn_decay_input(x);
 for(float x:{0.f,87.33651733398438f,88.f,89.f,104.f,138.f,-89.f,-128.f})native_gdn_sigmoid_input(x);
 for(float x:{1.f,INFINITY,NAN}){bool bad=false;try{native_gdn_decay_input(x);}catch(...){bad=true;}native_hc_need(bad,"GDN invalid decay class accepted");}
 for(float x:{-88.f,INFINITY,NAN,std::numeric_limits<float>::denorm_min()}){bool bad=false;try{native_gdn_sigmoid_input(x);}catch(...){bad=true;}native_hc_need(bad,"GDN unsupported sigmoid class accepted");}
 auto zero_decay=eg;zero_decay[0]=0;auto reset_state=control(f,zero_decay,rs,sig);native_hc_need(bits(reset_state.state[0])==0x3e800000,"GDN zero decay changed update association");
 f[2][0]=INFINITY;rejected=false;try{validate(f);}catch(...){rejected=true;}native_hc_need(rejected,"GDN nonfinite accepted");
 puts("GDN pinned sparse vector, wrong common output and invalid input rejection pass");}
