#include "../../src/core/native_q8_gu_layout.hpp"
#include "native_q8_consumer_fixture.hpp"
using namespace strata::core::portable_detail;
using native_q8_fixture::need;
namespace {template<class F>void rejects(F f){bool failed=false;try{f();}catch(const std::runtime_error&){failed=true;}need(failed,"invalid GU metadata accepted");}}
int main(){try {
    Q8GuLayout l{256,32,2,2,1,{0},{0,2},{0,1}};
    auto m=q8_gu_metadata(l,26112);
    need(m[0]==1 && m[1]==0 && m[17]==0 && m[18]==2 && m[33]==0 && m[34]==1,"native grouped metadata mapping");
    auto bad=l;bad.token_ids[1]=2;rejects([&]{q8_gu_metadata(bad,26112);});
    bad=l;bad.starts={2,1};rejects([&]{q8_gu_metadata(bad,26112);});
    bad=l;bad.weight_offsets[0]=4;rejects([&]{q8_gu_metadata(bad,17408);});
    bad=l;bad.weight_offsets[0]=2;rejects([&]{q8_gu_metadata(bad,26112);});
    bad=l;bad.h=32;rejects([&]{q8_gu_metadata(bad,26112);});
    bad=l;bad.f=31;rejects([&]{q8_gu_metadata(bad,26112);});
    bad=l;bad.grid_groups=0;rejects([&]{q8_gu_metadata(bad,26112);});
    // Empty groups and inactive entry/token slots preserve native meanings.
    Q8GuLayout sparse{256,32,4,2,1,{0,0,0},{1,2,2,3},{-1,1,0,-1}};
    m=q8_gu_metadata(sparse,26112);need(m[0]==3 && m[17]==1 && m[20]==3 && m[34]==1 && m[35]==0,"sparse native mapping");
    Q8GuLayout empty{256,32,2,2,1,{}, {1},{-1,-1}};
    m=q8_gu_metadata(empty,26112);need(m[0]==0 && m[17]==1,"zero-group metadata");
    std::puts("Q8 GU metadata tests passed; no numerical/GPU proof");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
