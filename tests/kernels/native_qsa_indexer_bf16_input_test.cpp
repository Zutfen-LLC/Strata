#include "native_qsa_indexer_bf16_fixture.hpp"
#include <cstdio>
#include <limits>
#include <vector>
using namespace strata::core::portable_detail;
template<class F> void rejects(F f){bool bad=false;try{f();}catch(const std::invalid_argument&){bad=true;}native_shared_need(bad,"QSA indexer BF16: expected refusal");}
int main(int argc,char**argv){try{
    namespace F=native_qsa_indexer_bf16_fixture;
    if(argc==3&&std::string(argv[1])=="--write"){F::write(argv[2],F::synthetic());return 0;}
    if(argc==3&&std::string(argv[1])=="--write-dense"){F::write(argv[2],F::dense());return 0;}
    if(argc==4&&std::string(argv[1])=="--contract"){F::contract(argv[2],argv[3]);return 0;}
    if(argc==5&&std::string(argv[1])=="--check"){auto f=F::read(argv[2]);std::array<std::vector<float>,2>out;for(size_t i=0;i<2;++i){auto data=F::read_file(argv[3+i],F::rows[i]*4);out[i].resize(F::rows[i]);std::memcpy(out[i].data(),data.data(),data.size());auto expected=qsa_indexer_bf16_control(f.x.data(),f.w[i].data(),F::rows[i]);native_shared_need(std::memcmp(expected.data(),out[i].data(),data.size())==0,"QSA indexer BF16 exact output differs");}F::literals(f,out);return 0;}
    native_shared_need(argc==1,"QSA indexer BF16 input: invalid arguments");
    native_shared_need(native_shared_mmvf_block(2560)==256,"native block size differs");
    {auto dense=F::dense();for(float v:dense.x)native_shared_need(v!=0.f,"dense input missing");for(const auto&role:dense.w)for(uint16_t v:role)native_shared_need(v!=0,"dense weight missing");std::array<std::vector<float>,2> out;for(size_t i=0;i<2;++i)out[i]=qsa_indexer_bf16_control(dense.x.data(),dense.w[i].data(),F::rows[i]);F::literals(dense,out);}

    // PLE value uses the same CUDA kernel at 2,560 rows. Exercise every
    // pair and every output row, then a separate order-sensitive sentinel.
    {
        std::vector<float> dense_input(2560,1.f);
        std::vector<uint16_t> dense_weight(size_t(2560)*2560,0x3f80);
        auto dense_result=native_bf16_matrix_control(dense_input.data(),dense_weight.data(),2560);
        for(float v:dense_result) native_shared_need(v==2560.f,"PLE value dense literal differs");
        std::vector<float> order(2560,0.f);
        std::fill(dense_weight.begin(),dense_weight.end(),0);
        order[0]=16777216.f;order[512]=1.f;order[513]=-16777216.f;order[2558]=3.f;order[2559]=5.f;
        dense_weight[0]=dense_weight[512]=dense_weight[513]=0x3f80;
        dense_weight[2559*2560+2558]=0x4000;
        dense_weight[2559*2560+2559]=0xbf80;
        auto ordered=native_bf16_matrix_control(order.data(),dense_weight.data(),2560);
        native_shared_need(ordered[0]==0.f && ordered[2559]==1.f,"PLE value FMA order/last row literal differs");
        rejects([&]{qsa_indexer_bf16_control(order.data(),dense_weight.data(),2560);});
    }
    std::vector<float>x(2560,0.f);std::vector<uint16_t>w(2560*128,0);
    x[0]=16777216.f;x[512]=1.f;x[513]=-16777216.f;x[2558]=3.f;x[2559]=5.f;
    auto put=[&](size_t row,size_t column,uint16_t value){w[row*2560+column]=value;};
    // Lane zero: 2^24, then +1 followed by -2^24 gives 0. A pair sum gives 1.
    put(0,0,0x3f80);put(0,512,0x3f80);put(0,513,0x3f80);
    put(1,0,0x3f80);put(127,2558,0x4000);put(127,2559,0xbf80);
    auto result=qsa_indexer_bf16_control(x.data(),w.data(),128);
    native_shared_need(result[0]==0.f && result[1]==16777216.f && result[127]==1.f,"independent ordered-FMA/row sentinels differ");
#ifdef STRATA_INDEXER_HOST_MXCSR
    struct RestoreMxcsr{unsigned value;~RestoreMxcsr(){_mm_setcsr(value);}} restore{_mm_getcsr()};
    _mm_setcsr(restore.value|(1u<<15));rejects([&]{qsa_indexer_bf16_control(x.data(),w.data(),128);});_mm_setcsr(restore.value);
#endif
    x[1]=std::numeric_limits<float>::denorm_min();rejects([&]{qsa_indexer_bf16_control(x.data(),w.data(),128);});x[1]=0;
    x[1]=std::numeric_limits<float>::infinity();rejects([&]{qsa_indexer_bf16_control(x.data(),w.data(),128);});x[1]=0;
    w[1]=1;rejects([&]{qsa_indexer_bf16_control(x.data(),w.data(),128);});w[1]=0;
    w[1]=0x7f80;rejects([&]{qsa_indexer_bf16_control(x.data(),w.data(),128);});w[1]=0;
    x[0]=std::numeric_limits<float>::max();w[0]=0x4000;rejects([&]{qsa_indexer_bf16_control(x.data(),w.data(),128);});
    rejects([&]{qsa_indexer_bf16_control(x.data(),w.data(),127);});
    puts("QSA indexer BF16 ordered pair, row sentinels and finite-domain refusals pass");return 0;
}catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
