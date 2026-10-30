#include "native_router_fixture.hpp"
#include <cfenv>
#include <chrono>
using namespace native_router_fixture;
using namespace strata::core::portable_detail;
int main(int argc,char** argv){try {
    need(std::fesetround(FE_TONEAREST)==0,"router host RNE required");
    const auto f=synthetic();std::array<float,512> p{};p.fill(1.f/512);
    const auto tie=select(p);for(size_t k=0;k<10;++k)need(tie.ids[k]==int(k),"lower ID tie rule");
    p.fill(0);for(int k=0;k<10;++k)p[7+k*32]=1.f-float(k)/32;
    const auto collision=select(p);for(int k=0;k<10;++k)need(collision.ids[k]==7+k*32,"same-lane picks lost");
    // Reduction witness: rank-order sum loses two half-ULPs; winning-lane
    // XOR groups the small picks together and retains one ULP.
    p.fill(0);p[0]=0.5f;p[1]=p[3]=0x1p-25f;
    const auto order=select(p);float rank=0;
    for(float x:order.probabilities){volatile float next=rank+x;rank=next;}
    need(bits(rank)!=bits(order.selected_sum),"fixture fails to distinguish rank accumulation");
    p.fill(0);need(select(p).selected_sum==0x1p-14f,"normalization clamp");
    // Probabilities, not logits, determine ties (underflow leaves ascending IDs).
    p.fill(0);p[511]=1;const auto underflow=select(p);
    need(underflow.ids[0]==511 && underflow.ids[1]==0 && underflow.ids[9]==8,"computed probability tie");
    need(native_router_parameters(8,{12,20,28})==std::array<uint32_t,4>{8,3,5,7},"view offset units");
    for(auto shape:{std::array<uint32_t,3>{256,8,1},{512,10,0},{512,10,9},{512,8,1}}) {
        bool refused=false;try{native_router_shape(shape[0],shape[1],shape[2]);}catch(...){refused=true;}need(refused,"unsupported shape");
    }
    for(size_t i=0;i<3;++i){auto bias=std::array<uint32_t,3>{12,20,28};++bias[i];bool refused=false;
        try{native_router_parameters(1,bias);}catch(...){refused=true;}need(refused,"misalignment");}
    auto invalid=f;invalid.logits[0]=std::numeric_limits<float>::infinity();bool refused=false;
    try{validate(invalid);}catch(...){refused=true;}need(refused,"nonfinite fixture");
    for(uint32_t tokens:{1u,2u,8u}) {
        auto subset=f;subset.tokens=tokens;subset.logits.resize(tokens*512);validate(subset);
        const auto views=padded(subset);need(views[1].size()==prefixes[1]+tokens*10+9,"T1/T2/T8 padded span");
    }
    const auto temporary=std::filesystem::temp_directory_path()/(
        "native-router-host-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    need(std::filesystem::create_directory(temporary),"router test directory collision");
    struct Cleanup {std::filesystem::path path;~Cleanup(){std::error_code ignored;std::filesystem::remove_all(path,ignored);}} cleanup{temporary};
    const auto fixture_path=(temporary/"fixture.bin").string();write_fixture(fixture_path,f);
    const auto roundtrip=read_fixture(fixture_path);need(roundtrip.tokens==f.tokens && roundtrip.logits.size()==f.logits.size(),"fixture roundtrip geometry");
    for(size_t i=0;i<f.logits.size();++i)need(bits(roundtrip.logits[i])==bits(f.logits[i]),"fixture roundtrip bits");
    refused=false;try{write_fixture(fixture_path,f);}catch(...){refused=true;}need(refused,"fixture overwrite");
    {std::ofstream trailing(fixture_path,std::ios::binary|std::ios::app);trailing.put('x');}
    refused=false;try{read_fixture(fixture_path);}catch(...){refused=true;}need(refused,"fixture trailing byte");
    reserve((temporary/"receipt").string());refused=false;
    try{reserve((temporary/"receipt").string());}catch(...){refused=true;}need(refused,"receipt overwrite");
    if(argc==3 && std::string(argv[1])=="--write-fixture")write_fixture(argv[2],f);
    else need(argc==1,"usage: native_router_input_test [--write-fixture NEW_FILE]");
    printf("router host ordering/shape witnesses pass; CUDA transcendental arithmetic not emulated\n");return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
