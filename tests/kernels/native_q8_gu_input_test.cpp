#include "../../src/core/native_q8_gu_layout.hpp"
#include "native_q8_consumer_fixture.hpp"
using namespace strata::core::portable_detail;
using native_q8_fixture::need;
namespace {template<class F>void rejects(F f){bool failed=false;try{f();}catch(const std::runtime_error&){failed=true;}need(failed,"invalid GU metadata accepted");}}
int main(){try {
    Q8GuLayout l{256,32,2,2,1,{0},{0,2},{0,1}};
    auto m=q8_gu_metadata(l,26112);
    need(m[0]==1 && m[1]==0 && m[25]==0 && m[26]==2 && m[50]==0 && m[51]==1,"native grouped metadata mapping");
    auto bad=l;bad.token_ids[1]=2;rejects([&]{q8_gu_metadata(bad,26112);});
    bad=l;bad.starts={2,1};rejects([&]{q8_gu_metadata(bad,26112);});
    bad=l;bad.weight_offsets[0]=4;rejects([&]{q8_gu_metadata(bad,17408);});
    bad=l;bad.weight_offsets[0]=2;rejects([&]{q8_gu_metadata(bad,26112);});
    bad=l;bad.h=32;rejects([&]{q8_gu_metadata(bad,26112);});
    bad=l;bad.f=31;rejects([&]{q8_gu_metadata(bad,26112);});
    bad=l;bad.grid_groups=0;rejects([&]{q8_gu_metadata(bad,26112);});
    // Empty groups and inactive entry/token slots preserve native meanings.
    Q8GuLayout sparse{256,32,4,2,1,{0,0,0},{1,2,2,3},{-1,1,0,-1}};
    m=q8_gu_metadata(sparse,26112);need(m[0]==3 && m[25]==1 && m[28]==3 && m[51]==1 && m[52]==0,"sparse native mapping");
    Q8GuLayout empty{256,32,2,2,1,{}, {1},{-1,-1}};
    m=q8_gu_metadata(empty,26112);need(m[0]==0 && m[25]==1,"zero-group metadata");
    Q8GuLayout full{256,32,64,64,7,std::vector<uint32_t>(24,0),std::vector<int32_t>(25),std::vector<int32_t>(64)};
    for(int i=0;i<25;++i)full.starts[i]=i<24?i:64;
    for(int i=0;i<64;++i)full.token_ids[i]=63-i;
    full.weight_offsets.back()=8704;
    m=q8_gu_metadata(full,26112);
    need(m[24]==8704 && m[25]==0 && m[49]==64 && m[50]==63 && m[113]==0 && m[114]==0,"24-group GU metadata overlap");
    auto stage=q8_stage_metadata(full.starts.data(),full.starts.size());need(stage[24]==64 && stage[31]==24 && stage[25]==0,"24-group activation starts/count overlap");
    full.weight_offsets.push_back(0);full.starts.push_back(64);rejects([&]{q8_gu_metadata(full,26112);});
    rejects([&]{q8_stage_metadata(full.starts.data(),full.starts.size());});
    std::puts("Q8 GU metadata tests passed; no numerical/GPU proof");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
