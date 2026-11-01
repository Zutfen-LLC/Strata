#include "native_q8_geometry_fixture.hpp"
#include "../../src/core/native_q8_gu_layout.hpp"
#include "../../src/core/native_q8_down_layout.hpp"
using namespace native_q8_fixture;
namespace geo=native_q8_fixture::geometry;
int main(){try {
    const auto weights=geo::weights();need(weights.size()==5222400,"production expert bytes");
    const auto old=consumer::weights();uint64_t old_fingerprint=14695981039346656037ull;
    for(uint8_t b:old)old_fingerprint=(old_fingerprint^b)*1099511628211ull;
    need(old.size()==26112 && old_fingerprint==0xe580f286ee045325ull,"old fixture bytes changed");
    need(geo::input().size()==7680 && geo::INPUT_BYTES==8640 && geo::HQ_BYTES==2160,"production stage shape");
    for(float f:geo::input())need(std::isfinite(f),"nonfinite input recipe");
    using namespace strata::core::portable_detail;
    Q8GuLayout gu{2560,640,3,3,2,{0,0,0},geo::STARTS,geo::TOKENS_BY_ENTRY};
    Q8DownLayout down{2560,640,3,3,2,{3481600,3481600,3481600},geo::STARTS,geo::DESTINATIONS};
    const auto gm=q8_gu_metadata(gu,weights.size()),dm=q8_down_metadata(down,weights.size());
    need(gm[0]==3 && gm[STRATA_Q8_START_WORD+2]==2 && gm[STRATA_Q8_START_WORD+3]==3 && gm[STRATA_Q8_ID_WORD]==2 && dm[STRATA_Q8_ID_WORD]==1,"empty group/entry maps lost");
    bool rejected=false;try{(void)q8_down_metadata(down,weights.size()-4);}catch(...){rejected=true;}
    need(rejected,"truncated production weights accepted");
    puts("H2560/F640 finite fixture, native layout and metadata bounds checked; no GPU proof");return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
