#include "native_qsa_attention_fixture.hpp"
using namespace native_qsa_attn;
int main(int argc,char**argv){try{
 if(argc==4&&std::string(argv[1])=="--synthetic"){write(argv[3],native_qsa_attn::synthetic(std::stoul(argv[2])));return 0;}
 if(argc==4&&std::string(argv[1])=="--contract"){contract(read(argv[2]),argv[3]);return 0;}
 if(argc==4&&std::string(argv[1])=="--check"){check(read(argv[2]),argv[3]);return 0;}
 if(argc==4&&std::string(argv[1])=="--oracle"){auto f=read(argv[2]);std::filesystem::path d=argv[3];native_hc_need(std::filesystem::create_directory(d),"fresh QSA attention oracle");std::filesystem::copy_file(argv[2],d/"fixture.bin");save(d,control(f));return 0;}
 native_hc_need(argc==1,"QSA attention control arguments");auto a=control(native_qsa_attn::synthetic(1));native_hc_need(a.out[3][0]==bits(-.5f)&&a.out[5][0]==bits(-.25f),"independent single-cell value/gate vector");native_hc_need(a.out[3][17]==bits(0x1p-24f)&&a.out[5][17]==bits(0x1p-25f),"independent exact FP16 subnormal expansion");auto b=control(native_qsa_attn::synthetic(2));native_hc_need(b.out[3][0]==bits(-.5f)&&b.out[5][0]==bits(-.25f),"independent 64/65-cell chunk/merge boundary");
 auto masked=control(native_qsa_attn::synthetic(5));for(auto v:masked.out[3])native_hc_need(v==0,"all-masked native attention must be zero");for(auto v:masked.out[5])native_hc_need(v==0,"all-masked gated output must be zero");
 auto c=control(native_qsa_attn::synthetic(6));native_hc_need(c.out[1][0]==bits(.0625f),"independent eight-term FMA cancellation source vector");int sum=0;for(int cell=0;cell<64;++cell){int at=((cell/4)*2*4+cell%4)*256;sum+=(at*13)%37-18;}native_hc_need(c.out[5][0]==bits(float(sum)/2048.f),"independent uniform-softmax value/page/head chronology");
 auto bad=native_qsa_attn::synthetic(0);bad.in[4][0]=1;bool refused=false;try{validate(bad);}catch(...){refused=true;}native_hc_need(refused,"out-of-range selected cell accepted");bad=native_qsa_attn::synthetic(0);bad.in[3][0]=64;refused=false;try{validate(bad);}catch(...){refused=true;}native_hc_need(refused,"out-of-range page accepted");
 puts("native QSA independent single-cell/chunk-edge/masked/FMA/value controls and invalid addressing refusals pass");return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
