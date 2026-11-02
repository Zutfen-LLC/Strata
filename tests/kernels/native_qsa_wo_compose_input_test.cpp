#include "native_qsa_wo_compose_fixture.hpp"
using namespace native_qsa_wo_compose;
int main(int argc,char** argv){try{
    if(argc==3 && std::string(argv[1])=="--synthetic-input") {q::write(argv[2],deterministic());return 0;}
    if(argc==5 && std::string(argv[1])=="--contract") {contract(argv[2],argv[3],argv[4]);return 0;}
    if(argc==5 && std::string(argv[1])=="--validate-packed") {
        auto ww=weights(argv[2]);(void)q::read(argv[3]);auto packed=w::load(argv[4],q::bytes);
        packed_domain(ww,packed);return 0;
    }
    throw std::runtime_error("composition: invalid validator arguments");
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
