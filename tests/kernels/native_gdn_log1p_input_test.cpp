#include "native_gdn_log1p_fixture.hpp"
using namespace native_gdn_log1p;
int main(int argc,char**argv){try{
 if(argc==4&&std::string(argv[1])=="--synthetic"){write(argv[3],native_gdn_log1p::synthetic(std::stoul(argv[2])));return 0;}
 if(argc==3&&std::string(argv[1])=="--contract"){contract(argv[2]);return 0;}
 if(argc==5&&std::string(argv[1])=="--check"){
  auto x=read(argv[2]);std::array<std::vector<float>,6>v;
  for(size_t j=0;j<6;++j){std::ifstream in(std::filesystem::path(argv[3])/(std::string(fields[j])+".f32le"),std::ios::binary);native_hc_need(bool(in),"log1p missing output");v[j].resize(count);for(float&a:v[j])a=value(word(in));native_hc_need(in.peek()==EOF,"log1p output shape");}
  for(size_t i=0;i<count;++i){volatile float u=1.f+x[i];native_hc_need(bits(v[2][i])==bits(u),"log1p rounded u");if(u==1.f){native_hc_need(bits(v[1][i])==bits(x[i])&&bits(v[3][i])==0&&bits(v[4][i])==0&&bits(v[5][i])==0,"log1p tiny identity/intermediates");}else{volatile float n=v[3][i]*x[i],d=u-1.f;native_hc_need(bits(v[4][i])==bits(n)&&bits(v[5][i])==bits(d),"log1p multiplication/subtraction order");}if(std::string(argv[4])=="vulkan")native_hc_need(bits(v[0][i])==bits(v[1][i]),"log1p Vulkan target changed");}return 0;
 }
 for(unsigned m=0;m<3;++m)validate(native_gdn_log1p::synthetic(m));
 for(float bad:{-1.f,INFINITY,NAN,value(1),value(0x4e000001)}){bool no=false;try{native_gdn_log1p_input(bad);}catch(...){no=true;}native_hc_need(no,"log1p invalid domain accepted");}
 native_hc_need(bits(1.f+value(0x33800000))==0x3f800000&&bits(1.f+value(0x33800001))==0x3f800001,"log1p RNE branch boundary");return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
