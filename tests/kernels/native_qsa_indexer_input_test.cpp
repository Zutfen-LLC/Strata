#include "native_qsa_indexer_fixture.hpp"
using namespace native_qsa_idx;
int main(int argc,char**argv){try{
 if(argc==4&&std::string(argv[1])=="--synthetic"){write(argv[3],native_qsa_idx::synthetic(std::stoul(argv[2])));return 0;}
 if(argc==4&&std::string(argv[1])=="--contract"){contract(native_qsa_idx::read(argv[2]),argv[3]);return 0;}
 if(argc==4&&std::string(argv[1])=="--check"){check(native_qsa_idx::read(argv[2]),argv[3]);return 0;}
 if(argc==4&&std::string(argv[1])=="--oracle"){auto f=native_qsa_idx::read(argv[2]);oracle(f,argv[3]);std::filesystem::copy_file(argv[2],std::filesystem::path(argv[3])/"fixture.bin");check(f,argv[3]);return 0;}
 native_hc_need(argc==1,"indexer input arguments");
 // Expected binary16 words are independent pinned IEEE ties/limits, not an
 // expected vector produced by the conversion implementation under test.
 const float in[]={1.00048828125f,1.00146484375f,-1.00048828125f,65504.f,0x1p-24f,0x1p-25f,-0x1p-25f,-0.f};const uint16_t out[]={0x3c00,0x3c02,0xbc00,0x7bff,1,0,0x8000,0x8000};for(size_t i=0;i<8;++i)native_hc_need(ggml_fp32_to_fp16(in[i])==out[i],"independent F16 tie/subnormal/negative-zero words");
 auto f=native_qsa_idx::synthetic(0);f.steps=1;f.raw.assign(128,1.f);std::vector<float>d(386);for(size_t i=0;i<256;++i)d[i]=1;d[256]=1.f+1e-6f;d[257]=.5f;for(size_t i=0;i<32;++i)d[322+i]=1;auto a=control(f,0,f.before,d);for(size_t i=0;i<128;++i){native_hc_need(a[i]==0x3f800000&&a[384+i]==0x3f000000&&a[512+i]==0x3f000000,"independent repeated-cell-zero spare");}native_hc_need(a.back()==0&&a[128]==0,"spare must not invent tail/block updates");
 f.first=3;f.before.assign(f.before.size(),0);for(size_t i=0;i<384;++i)f.before[i]=bits(float(i/128+1));f.raw.assign(128,4);for(size_t i=0;i<128;++i){d[i]=4;d[128+i]=2.5f;}d[256]=6.25f+1e-6f;auto b=control(f,0,f.before,d);native_hc_need(b[0]==bits(1.f)&&b[128]==bits(2.f)&&b[256]==bits(3.f),"completed native tail remains retained, not cleared");for(size_t i=0;i<128;++i)native_hc_need(b[640+i]==0,"completed spare copies persistent dead row");
 for(uint32_t cells:{255u,256u,257u,2048u,2049u,2050u,2051u,2052u,4095u,4096u})native_qsa_indexer_geometry(cells,int32_t(cells-1),0,f.theta);
 bool refused=false;try{native_qsa_indexer_geometry(4096,0,4,f.theta);}catch(...){refused=true;}native_hc_need(refused,"unsupported absolute context admitted");
 refused=false;try{native_qsa_indexer_geometry(4097,0,0,f.theta);}catch(...){refused=true;}native_hc_need(refused,"unsupported capacity admitted");
 puts("indexer independent F16/pool/spare/native-tail and geometry controls pass");return 0;
 }catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
