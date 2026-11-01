#include "../../src/core/native_q8_address.hpp"
#include "../../src/core/native_q8_down_layout.hpp"
#include <iostream>
using namespace strata::core::portable_detail;
template<class F>void rejects(F f){bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}q8_gu_need(rejected,"invalid address accepted");}
int main(){try {
    constexpr uint64_t blob=5222400,down=3481600;
    for(size_t slots:{24,25,205,206,480,512,822,823,960,1378}) {
        // Arithmetic-only ownership: no giant host/device buffer allocation.
        const uint64_t bytes=slots*blob,off=(slots-1)*blob;
        auto address=q8_weight_address(0x1000000000,bytes,0,0,bytes,off+down,blob-down);
        q8_gu_need(address==0x1000000000+off+down,"high address/carry lost");
        Q8GroupMetadata m{};q8_store_address(m,79,address);
        const auto restored=uint64_t(uint32_t(m[159]))|(uint64_t(uint32_t(m[160]))<<32);
        q8_gu_need(restored==address,"address ABI narrowed");
    }
    q8_gu_need(q8_weight_address(0xfffffff0,1024,8,12,100,4,96)==0x100000008,"view/owner carry incorrect");
    rejects([]{q8_weight_address(0,100,0,0,100,0,4);});
    rejects([]{q8_weight_address(4,100,8,12,100,0,4);});
    rejects([]{q8_weight_address(4,100,0,0,100,96,8);});
    rejects([]{q8_weight_address(UINT64_MAX-3,100,0,0,100,4,4);});
    rejects([]{q8_weight_address(4,100,UINT64_MAX,4,100,0,4);});
    rejects([]{q8_weight_address(4,100,0,0,100,2,4);});
    for(int features=0;features<4;++features) {
        if(features==3)q8_require_address_features(true,true);
        else rejects([&]{q8_require_address_features(features&1,features&2);});
    }
    q8_gu_need(q8_resident_limit(1ull<<30,4294967292,4294967292)==1ull<<30,"backend policy lost");
    q8_gu_need(q8_resident_limit(1ull<<30,10000,5000)==5000,"allocation limit lost");
    rejects([]{q8_resident_limit(100,0,100);});
    Q8GuLayout gu{2560,640,80,8,80,std::vector<uint64_t>(80),std::vector<int32_t>(81),std::vector<int32_t>(80)};
    Q8DownLayout d{2560,640,80,80,80,std::vector<uint64_t>(80,down),{}, {}};
    for(int i=0;i<81;++i)gu.starts[i]=i;
    for(int i=0;i<80;++i)gu.token_ids[i]=(i*3+1)%8;
    d.starts=gu.starts;for(int i=0;i<80;++i)d.destination_ids.push_back(79-i);
    const auto gm=q8_gu_metadata(gu,blob),dm=q8_down_metadata(d,blob);
    auto stage=q8_stage_metadata(gu.starts.data(),81);
    q8_gu_need(gm[241]==80 && dm[321]==0 && gm[321]==gu.token_ids.back() && stage[80]==80 && stage[81]==80,"80-group ABI overlaps");
    gu.weight_offsets.push_back(0);gu.starts.push_back(80);rejects([&]{q8_gu_metadata(gu,blob);});
    rejects([&]{q8_stage_metadata(gu.starts.data(),82);});
    puts("native Q8 64-bit owner/address/80-group ABI tests passed; no GPU execution");return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
