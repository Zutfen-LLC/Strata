#include "../../src/core/native_q8_down_layout.hpp"
#include "native_q8_gu_fixture.hpp"
using namespace strata::core::portable_detail;
using native_q8_fixture::need;
namespace {template<class F>void rejects(F f){bool failed=false;try{f();}catch(const std::runtime_error&){failed=true;}need(failed,"invalid down metadata accepted");}}
int main(){try {
    Q8DownLayout l{256,32,2,4,1,{17408},{0,2},{0,1}};
    auto m=q8_down_metadata(l,26112);need(m[0]==1 && m[1]==17408 && m[STRATA_Q8_START_WORD+1]==2 && m[STRATA_Q8_ID_WORD+1]==1,"native down mapping");
    auto bad=l;bad.weight_offsets[0]+=4;rejects([&]{q8_down_metadata(bad,26112);});
    bad=l;bad.weight_offsets[0]=17410;rejects([&]{q8_down_metadata(bad,26112);});
    bad=l;bad.destination_ids[1]=4;rejects([&]{q8_down_metadata(bad,26112);});
    bad=l;bad.starts={2,1};rejects([&]{q8_down_metadata(bad,26112);});
    bad=l;bad.f=31;rejects([&]{q8_down_metadata(bad,26112);});
    bad=l;bad.h=255;rejects([&]{q8_down_metadata(bad,26112);});
    bad=l;bad.grid_groups=0;rejects([&]{q8_down_metadata(bad,26112);});
    l.starts={1,2};l.destination_ids={-1,2};m=q8_down_metadata(l,26112);
    need(m[STRATA_Q8_START_WORD]==1 && m[STRATA_Q8_START_WORD+1]==2 && m[STRATA_Q8_ID_WORD]==-1 && m[STRATA_Q8_ID_WORD+1]==2,"entry/destination distinction lost");
    l.weight_offsets={17408,0,17408};l.starts={0,1,1,2};l.destination_ids={3,1};l.grid_groups=2;
    m=q8_down_metadata(l,26112);need(m[0]==3 && m[STRATA_Q8_START_WORD+3]==2 && m[STRATA_Q8_ID_WORD]==3,"empty group/strided grid mapping");
    l.weight_offsets.clear();l.starts={1};l.destination_ids={-1,-1};m=q8_down_metadata(l,26112);need(m[0]==0,"empty down groups");
    // A same-group repeated destination stays a direct ordered overwrite.
    l.weight_offsets={17408};l.starts={0,2};l.destination_ids={1,1};q8_down_metadata(l,26112);
    // Exercise the final ID, all 25 starts and destinations above 63.
    for(uint32_t n:{64u,65u,79u,80u}) {
        Q8DownLayout full{256,32,n,n,7,std::vector<uint64_t>(24,17408),std::vector<int32_t>(25),std::vector<int32_t>(n)};
        for(int i=0;i<25;++i)full.starts[i]=i<24?i:n;
        for(uint32_t i=0;i<n;++i)full.destination_ids[i]=n-1-i;
        struct Guarded {uint32_t before=0x12345678;Q8GroupMetadata data{};uint32_t after=0x87654321;} guarded;
        guarded.data=q8_down_metadata(full,26112);m=guarded.data;
        need(guarded.before==0x12345678 && guarded.after==0x87654321 && sizeof(m)==q8_metadata_words*4,"metadata guard/upload size");
        need(m[STRATA_Q8_ADDRESS_WORD+2*23]==17408 && m[STRATA_Q8_START_WORD]==0 && m[STRATA_Q8_START_WORD+24]==int32_t(n) && m[STRATA_Q8_ID_WORD]==full.destination_ids[0] && m[STRATA_Q8_ID_WORD+n-1]==0,"24-group down metadata overlap");
        for(size_t i=STRATA_Q8_ID_WORD+n;i<m.size();++i)need(m[i]==0,"metadata tail not zero");
        auto stage=q8_stage_metadata(full.starts.data(),full.starts.size());
        need(stage[24]==int32_t(n) && stage[STRATA_Q8_STAGE_COUNT_WORD]==24 && stage[25]==0,"activation starts/count overlap");
        auto bad=full;bad.destination_ids.back()=n;rejects([&]{q8_down_metadata(bad,26112);});
        // Padded inactive endpoints and an empty group retain their IDs, ignored.
        full.starts[0]=1;full.starts[1]=1;full.starts.back()=n-1;
        full.destination_ids.front()=-1;full.destination_ids.back()=-1;
        q8_down_metadata(full,26112);
        bad=full;bad.entries=81;bad.destination_ids.resize(81);rejects([&]{q8_down_metadata(bad,26112);});
        bad=full;bad.destinations=81;rejects([&]{q8_down_metadata(bad,26112);});
        full.weight_offsets.resize(81,0);full.starts.resize(82,n-1);rejects([&]{q8_down_metadata(full,26112);});
        rejects([&]{q8_stage_metadata(full.starts.data(),full.starts.size());});
    }
    puts("native down metadata tests passed; no GPU/numerical proof");return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
