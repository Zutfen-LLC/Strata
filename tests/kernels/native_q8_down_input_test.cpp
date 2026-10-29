#include "../../src/core/native_q8_down_layout.hpp"
#include "native_q8_gu_fixture.hpp"
using namespace strata::core::portable_detail;
using native_q8_fixture::need;
namespace {template<class F>void rejects(F f){bool failed=false;try{f();}catch(const std::runtime_error&){failed=true;}need(failed,"invalid down metadata accepted");}}
int main(){try {
    Q8DownLayout l{256,32,2,4,1,{17408},{0,2},{0,1}};
    auto m=q8_down_metadata(l,26112);need(m[0]==1 && m[1]==17408 && m[18]==2 && m[34]==1,"native down mapping");
    auto bad=l;bad.weight_offsets[0]+=4;rejects([&]{q8_down_metadata(bad,26112);});
    bad=l;bad.weight_offsets[0]=17410;rejects([&]{q8_down_metadata(bad,26112);});
    bad=l;bad.destination_ids[1]=4;rejects([&]{q8_down_metadata(bad,26112);});
    bad=l;bad.starts={2,1};rejects([&]{q8_down_metadata(bad,26112);});
    bad=l;bad.f=31;rejects([&]{q8_down_metadata(bad,26112);});
    bad=l;bad.h=255;rejects([&]{q8_down_metadata(bad,26112);});
    bad=l;bad.grid_groups=0;rejects([&]{q8_down_metadata(bad,26112);});
    l.starts={1,2};l.destination_ids={-1,2};m=q8_down_metadata(l,26112);
    need(m[17]==1 && m[18]==2 && m[33]==-1 && m[34]==2,"entry/destination distinction lost");
    l.weight_offsets={17408,0,17408};l.starts={0,1,1,2};l.destination_ids={3,1};l.grid_groups=2;
    m=q8_down_metadata(l,26112);need(m[0]==3 && m[20]==2 && m[33]==3,"empty group/strided grid mapping");
    l.weight_offsets.clear();l.starts={1};l.destination_ids={-1,-1};m=q8_down_metadata(l,26112);need(m[0]==0,"empty down groups");
    // A same-group repeated destination stays a direct ordered overwrite.
    l.weight_offsets={17408};l.starts={0,2};l.destination_ids={1,1};q8_down_metadata(l,26112);
    puts("native down metadata tests passed; no GPU/numerical proof");return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
