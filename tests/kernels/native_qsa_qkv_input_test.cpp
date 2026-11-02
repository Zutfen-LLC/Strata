#include "native_qsa_qkv_fixture.hpp"
using namespace native_qsa_qkv_fixture;
int main(int argc,char**argv){try {
    if(argc==3&&std::string(argv[1])=="--synthetic") {
        std::filesystem::path dir=argv[2];need(std::filesystem::create_directory(dir),"QSA QKV: fixture directory exists");
        q::write((dir/"input.bin").string(),input());
        for(size_t role=0;role<3;++role){auto w=weights(role);save(dir/(std::string(names[role])+".bin"),w.data(),w.size());}
        return 0;
    }
    if(argc==5&&std::string(argv[1])=="--contract"){contract(argv[2],argv[3],argv[4]);return 0;}
    throw std::runtime_error("QSA QKV: expected --synthetic DIR or --contract WEIGHTS INPUT OUTPUT");
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
