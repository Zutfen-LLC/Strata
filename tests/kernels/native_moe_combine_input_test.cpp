#include "native_moe_combine_fixture.hpp"
#include <cfenv>
using namespace native_moe_fixture;
using namespace strata::core::portable_detail;
int main(int argc,char** argv){try {
    need(std::fesetround(FE_TONEAREST)==0,"RNE rounding unavailable");
    const auto f=synthetic();const auto expected=reference(f);
    need(bits(expected[0])==bits(0x1p-46f),"first rounded-product fixture changed");
    need(bits(expected[1])==bits(-0x1.000002p-23f),"FMA fixture changed");
    need(expected[H]==1.f && expected[H+1]==1.f,"order/shared-last fixture changed");
    size_t separate_diff=0,reorder_diff=0,shared_early_diff=0,first_fused_diff=0;
    for(size_t t=0;t<f.tokens;++t)for(size_t h=0;h<H;++h) {
        volatile float first=f.parts[t*K*H+h]*f.weights[t*K];float separate=first,reverse=first,early=first+f.shared[t*H+h];
        for(size_t k=1;k<K;++k){volatile float product=f.parts[(t*K+k)*H+h]*f.weights[t*K+k];separate+=product;
            early=std::fma(f.parts[(t*K+k)*H+h],f.weights[t*K+k],early);}
        for(size_t k=K-1;k>0;--k)reverse=std::fma(f.parts[(t*K+k)*H+h],f.weights[t*K+k],reverse);
        volatile float second=f.parts[(t*K+1)*H+h]*f.weights[t*K+1];
        float wrong_first=std::fma(f.parts[t*K*H+h],f.weights[t*K],second);
        for(size_t k=2;k<K;++k)wrong_first=std::fma(f.parts[(t*K+k)*H+h],f.weights[t*K+k],wrong_first);
        const float ref=expected[t*H+h];
        separate_diff+=bits(separate+f.shared[t*H+h])!=bits(ref);
        reorder_diff+=bits(reverse+f.shared[t*H+h])!=bits(ref);
        shared_early_diff+=bits(early)!=bits(ref);
        first_fused_diff+=bits(wrong_first+f.shared[t*H+h])!=bits(ref);
    }
    need(separate_diff && reorder_diff && shared_early_diff && first_fused_diff,"combine fixture misses an arithmetic-order negative control");
    const auto params=native_moe_combine_parameters(H,2,{16,12,32,48});need(params==std::array<uint32_t,6>{H,2,1,3,2,3},"combine view units");
    for(auto shape:{std::pair<uint32_t,uint32_t>{H,0},{H,9},{H-4,1}}){bool rejected=false;try{native_moe_combine_shape(shape.first,shape.second);}catch(...){rejected=true;}need(rejected,"combine unsupported shape accepted");}
    for(size_t i:{0u,1u,2u,3u}){std::array<uint32_t,4> bias{16,12,32,48};bias[i]+=2;bool rejected=false;
        try{native_moe_combine_parameters(H,2,bias);}catch(...){rejected=true;}need(rejected,"combine misaligned view accepted");}
    if(argc==3 && std::string(argv[1])=="--write-fixture")write_fixture(argv[2],f);
    else need(argc==1,"usage: native_moe_combine_input_test [--write-fixture NEW_FILE]");
    printf("combine host fixture passes: first_product=%zu separate=%zu reordered=%zu shared_early=%zu; no GPU execution\n",first_fused_diff,separate_diff,reorder_diff,shared_early_diff);return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
