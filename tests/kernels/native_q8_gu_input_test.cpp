#include "../../src/core/native_q8_gu_layout.hpp"
#include "native_q8_consumer_fixture.hpp"
using namespace strata::core::portable_detail;
using native_q8_fixture::need;
namespace {template<class F>void rejects(F f){bool failed=false;try{f();}catch(const std::runtime_error&){failed=true;}need(failed,"invalid GU metadata accepted");}}
int main(){try {
    Q8GuLayout l{256,32,2,2,1,{0},{0,2},{0,1}};
    auto m=q8_gu_metadata(l,26112);
    need(m[0]==1 && m[1]==0 && m[STRATA_Q8_START_WORD]==0 && m[STRATA_Q8_START_WORD+1]==2 && m[STRATA_Q8_ID_WORD]==0 && m[STRATA_Q8_ID_WORD+1]==1,"native grouped metadata mapping");
    auto bad=l;bad.token_ids[1]=2;rejects([&]{q8_gu_metadata(bad,26112);});
    bad=l;bad.starts={2,1};rejects([&]{q8_gu_metadata(bad,26112);});
    bad=l;bad.weight_offsets[0]=4;rejects([&]{q8_gu_metadata(bad,17408);});
    bad=l;bad.weight_offsets[0]=2;rejects([&]{q8_gu_metadata(bad,26112);});
    bad=l;bad.h=32;rejects([&]{q8_gu_metadata(bad,26112);});
    bad=l;bad.f=31;rejects([&]{q8_gu_metadata(bad,26112);});
    bad=l;bad.grid_groups=0;rejects([&]{q8_gu_metadata(bad,26112);});
    // Empty groups and inactive entry/token slots preserve native meanings.
    Q8GuLayout sparse{256,32,4,2,1,{0,0,0},{1,2,2,3},{-1,1,0,-1}};
    m=q8_gu_metadata(sparse,26112);need(m[0]==3 && m[STRATA_Q8_START_WORD]==1 && m[STRATA_Q8_START_WORD+3]==3 && m[STRATA_Q8_ID_WORD+1]==1 && m[STRATA_Q8_ID_WORD+2]==0,"sparse native mapping");
    Q8GuLayout empty{256,32,2,2,1,{}, {1},{-1,-1}};
    m=q8_gu_metadata(empty,26112);need(m[0]==0 && m[STRATA_Q8_START_WORD]==1,"zero-group metadata");
    // Exercise the final ID, all 25 starts and IDs above 63. GU token
    // capacity is independent of entry capacity and remains 64.
    for(uint32_t n:{64u,65u,79u,80u}) {
        Q8GuLayout full{256,32,n,64,7,std::vector<uint64_t>(24,0),std::vector<int32_t>(25),std::vector<int32_t>(n)};
        for(int i=0;i<25;++i)full.starts[i]=i<24?i:n;
        for(uint32_t i=0;i<n;++i)full.token_ids[i]=(n-1-i)%64;
        struct Guarded {uint32_t before=0x12345678;Q8GroupMetadata data{};uint32_t after=0x87654321;} guarded;
        guarded.data=q8_gu_metadata(full,26112);m=guarded.data;
        need(guarded.before==0x12345678 && guarded.after==0x87654321 && sizeof(m)==q8_metadata_words*4,"metadata guard/upload size");
        need(m[STRATA_Q8_ADDRESS_WORD+2*23]==0 && m[STRATA_Q8_START_WORD]==0 && m[STRATA_Q8_START_WORD+24]==int32_t(n) && m[STRATA_Q8_ID_WORD]==full.token_ids[0] && m[STRATA_Q8_ID_WORD+n-1]==0,"24-group gu metadata overlap");
        for(size_t i=STRATA_Q8_ID_WORD+n;i<m.size();++i)need(m[i]==0,"metadata tail not zero");
        auto stage=q8_stage_metadata(full.starts.data(),full.starts.size());
        need(stage[24]==int32_t(n) && stage[STRATA_Q8_STAGE_COUNT_WORD]==24 && stage[25]==0,"activation starts/count overlap");
        auto bad=full;bad.token_ids.back()=64;rejects([&]{q8_gu_metadata(bad,26112);});
        // Padded inactive endpoints and an empty group retain their IDs, ignored.
        full.starts[0]=1;full.starts[1]=1;full.starts.back()=n-1;
        full.token_ids.front()=-1;full.token_ids.back()=-1;
        q8_gu_metadata(full,26112);
        bad=full;bad.entries=81;bad.token_ids.resize(81);rejects([&]{q8_gu_metadata(bad,26112);});
        bad=full;bad.tokens=65;rejects([&]{q8_gu_metadata(bad,26112);});
        full.weight_offsets.resize(81,0);full.starts.resize(82,n-1);rejects([&]{q8_gu_metadata(full,26112);});
        rejects([&]{q8_stage_metadata(full.starts.data(),full.starts.size());});
    }
    std::puts("Q8 GU metadata tests passed; no numerical/GPU proof");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
