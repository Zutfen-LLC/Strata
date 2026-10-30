#include "native_q8_geometry_fixture.hpp"
#include "../../src/core/native_q8_gu_layout.hpp"
#include "../../src/core/native_q8_down_layout.hpp"
using namespace native_q8_fixture;
namespace geo=native_q8_fixture::geometry;
using namespace strata::core::portable_detail;
int main(){try {
    need(geo::ENTRIES==80 && geo::TOKENS==8 && geo::GROUPS==24,"capacity fixture shape");
    need(geo::INPUT_BYTES==23040 && geo::HQ_BYTES==57600 && geo::OUTPUT_FLOATS==204800,"capacity boundary sizes");
    const auto weights=geo::weights();need(weights.size()==geo::WEIGHT_BYTES,"capacity weights changed");
    for(float x:geo::input())need(std::isfinite(x),"capacity nonfinite input");
    Q8GuLayout gu{geo::H,geo::F,geo::ENTRIES,geo::TOKENS,geo::GRID_GROUPS,
        std::vector<uint32_t>(geo::GROUPS,0),geo::STARTS,geo::TOKENS_BY_ENTRY};
    Q8DownLayout down{geo::H,geo::F,geo::ENTRIES,geo::ENTRIES,geo::GRID_GROUPS,
        std::vector<uint32_t>(geo::GROUPS,geo::DOWN_OFFSET),geo::STARTS,geo::DESTINATIONS};
    const auto gm=q8_gu_metadata(gu,weights.size()),dm=q8_down_metadata(down,weights.size());
    need(gm.size()==130 && dm.size()==130 && gm[49]==geo::STARTS.back() && gm[129]==geo::TOKENS_BY_ENTRY.back() && dm[129]==geo::DESTINATIONS.back(),"capacity fixture final metadata");
    need(geo::STARTS[11]==geo::STARTS[12] && geo::GRID_GROUPS<geo::GROUPS,"strided/empty group fixture");
    std::vector<bool> seen(80,false);
    for(int e=geo::STARTS.front();e<geo::STARTS.back();++e){
        need(geo::TOKENS_BY_ENTRY[e]==(e*5+3)%8,"nontrivial token map");
        need(geo::DESTINATIONS[e]==79-e && !seen[geo::DESTINATIONS[e]],"unique destination permutation");seen[geo::DESTINATIONS[e]]=true;
    }
#ifdef STRATA_Q8_PADDED_FIXTURE
    need(geo::STARTS.front()==1 && geo::STARTS.back()==79 && !seen.front() && !seen.back(),"padded fixture interval");
#else
    need(geo::STARTS.front()==0 && geo::STARTS.back()==80 && std::all_of(seen.begin(),seen.end(),[](bool x){return x;}),"full 80 active entries");
#endif
    puts("80-entry composed fixture maps/boundaries passed; no GPU proof");return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
