#include "native_shared_fixture.hpp"
using namespace native_shared_fixture;
template<class F> void rejects(F fn) {bool caught=false;try{fn();}catch(const std::invalid_argument&){caught=true;}need(caught,"expected input rejection");}
int main(int argc,char** argv) {try {
    if(argc==6 && std::string(argv[1])=="--write") {
        write_fixture(argv[2],synthetic(uint32_t(std::stoul(argv[3])),uint32_t(std::stoul(argv[4])),uint32_t(std::stoul(argv[5]))));return 0;
    }
    need(argc==1,"usage: native_shared_input_test [--write NEW_FILE H F T]");
    for(uint32_t t=1;t<=8;++t){auto f=synthetic(2560,640,t);auto image=padded(f);need(image[X].size()==prefix(X)+2560*t+9,"padded count");}
    need(native_shared_mmvf_block(2560)==256 && native_shared_mmvf_block(640)==160,"native MMVF block selection");
    need(native_shared_promoted_bf16(-0.f)==0x8000,"signed zero BF16");
    rejects([]{native_shared_shape(2561,640,1);});rejects([]{native_shared_shape(2560,640,9);});
    rejects([]{native_shared_shape(2560,640,1,12);});rejects([]{native_shared_promoted_bf16(1.00001f);});
    rejects([]{native_shared_promoted_bf16(value(0x7f800000));});
    auto f=synthetic(64,32,1);f.data[WS][0]=0x7f80;bool rejected=false;try{validate(f);}catch(const std::runtime_error&){rejected=true;}need(rejected,"nonfinite BF16 reject");
    puts("native shared fixture and binding checks passed");return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
