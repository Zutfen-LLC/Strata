#include "native_qsa_qkv_fixture.hpp"
#ifdef STRATA_QKV_Q81_SCALE_SUITE
#include "native_q8_wo_fixture.hpp"
#endif
using namespace native_qsa_qkv_fixture;
int main(int argc,char**argv){try {
#ifdef STRATA_QKV_Q81_SCALE_SUITE
    if(argc==2&&std::string(argv[1])=="--self-test") {
        for(unsigned h=1;h<=1023;++h)need(native_q8_wo_fixture::input_half(h)==std::ldexp(float(h),-24),"QSA QKV: half decoder");
        bool refused=false;try{(void)native_q8_wo_fixture::input_half(0x7c00);}catch(...){refused=true;}need(refused,"QSA QKV: nonfinite input scale accepted");
        auto direct=direct_packed();for(size_t i=0;i<3;++i){auto w=weights(i);auto ref=[&](size_t row){return native_q8_wo_fixture::reference_blocks(w.data(),w.size(),direct.data(),direct.size(),rows[i],row,qsa_qkv_blocks).value;};
            need(ref(1)==std::ldexp(23.f,-24)&&ref(3)==-std::ldexp(23.f*1023.f,-24)&&ref(4)==-std::ldexp(23.f,-38),"QSA QKV: direct consumer literal differs");}
        return 0;
    }
    if(argc==3&&std::string(argv[1])=="--scale-suite") {
        std::filesystem::path dir=argv[2];need(std::filesystem::create_directory(dir),"QSA QKV: fixture directory exists");
        for(size_t i=0;i<scale_cases;++i)q::write((dir/case_name(i)).string(),scale_input(i));
        for(size_t role=0;role<3;++role){auto w=weights(role);save(dir/(std::string(names[role])+".bin"),w.data(),w.size());}
        return 0;
    }
#endif
    if(argc==3&&std::string(argv[1])=="--synthetic") {
        std::filesystem::path dir=argv[2];need(std::filesystem::create_directory(dir),"QSA QKV: fixture directory exists");
        q::write((dir/"input.bin").string(),input());
        for(size_t role=0;role<3;++role){auto w=weights(role);save(dir/(std::string(names[role])+".bin"),w.data(),w.size());}
        return 0;
    }
    if(argc==5&&std::string(argv[1])=="--contract"){contract(argv[2],argv[3],argv[4]);return 0;}
    throw std::runtime_error("QSA QKV: expected --synthetic DIR or --contract WEIGHTS INPUT OUTPUT");
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
