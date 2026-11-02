#include "native_gdn_input_fixture.hpp"
using namespace native_gdn_input_fixture;
int main(int argc,char**argv){try{
 if(argc==2&&std::string(argv[1])=="--layout"){
  for(size_t n:rows){gdn_input_require_layout(n,n*gdn_input_row_bytes,gdn_input_packed_bytes,n*4);for(unsigned i=0;i<4;++i){bool refused=false;try{gdn_input_require_layout(n+(i==0),n*gdn_input_row_bytes-(i==1),gdn_input_packed_bytes-(i==2),n*4-(i==3));}catch(...){refused=true;}need(refused,"GDN input: invalid shape accepted");}}return 0;}
 if(argc==3&&std::string(argv[1])=="--synthetic"){std::filesystem::path d=argv[2];need(std::filesystem::create_directory(d),"GDN input: fixture directory exists");for(unsigned i=0;i<2;++i)q::write((d/(i?"dense.bin":"input.bin")).string(),input(i));for(size_t i=0;i<2;++i){auto w=weights(i);save(d/(std::string(names[i])+".bin"),w.data(),w.size());}return 0;}
 if(argc==5&&std::string(argv[1])=="--contract"){contract(argv[2],argv[3],argv[4]);return 0;}
 throw std::runtime_error("GDN input: expected --synthetic DIR or --contract WEIGHTS INPUT OUTPUT");
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
