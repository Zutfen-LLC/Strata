#include "../../src/core/portable_native_capacity.hpp"
#include "strata/core/expert_profile.hpp"
#include <cstdio>
#include <filesystem>
#include <chrono>
using namespace strata::core;using namespace strata::core::portable_detail;
static void need(bool b){if(!b)throw std::runtime_error("native capacity golden mismatch");}
template<class F>static void reject(F f){bool bad=false;try{f();}catch(const std::exception&){bad=true;}need(bad);}
int main(){try {
    constexpr size_t blob=5222400,scratch=2u<<20; // Synthetic allowance, not hardware evidence.
    auto estimate=[](size_t n){return n*blob+scratch;};
    NativeCapacityRoom room{3584ull<<20,8192ull<<20,7168ull<<20,9216ull<<20,512ull<<20,5120ull<<20,128ull<<20};
    need(room.tensor_room()==(4480ull<<20));
    need(native_primary_quota(960,512,std::vector<size_t>(48,blob),room.tensor_room(),estimate)==18);
    // Independent golden native formula: min(request byte ceiling, free room)/sum(strides).
    for(size_t layers:{1u,2u,7u})for(size_t request=1;request<45;++request)for(size_t bytes=0;bytes<250;bytes+=7) {
        std::vector<size_t> stride(layers);size_t sum=0,max=0;
        for(size_t l=0;l<layers;++l){stride[l]=3+l;sum+=stride[l];max=std::max(max,stride[l]);}
        size_t expected=std::min(size_t(8),std::min(request*max,bytes)/sum);
        auto cost=[&](size_t slots){return slots/layers*sum;};
        if(expected)need(native_primary_quota(request,8,stride,bytes,cost)==expected);
        else reject([&]{native_primary_quota(request,8,stride,bytes,cost);});
    }
    auto padded=[](size_t n){return ((n*7+15)/16)*16+9;};
    need(native_fit_count(10,41,padded)==4);need(native_fit_count(10,40,padded)==2);
    for(size_t n=1;n<=6;++n) {need(native_fit_count(n,estimate(n),estimate)==n);need(native_fit_count(n,estimate(n)-1,estimate)==n-1);}
    reject([&]{native_primary_quota(SIZE_MAX,512,{2},100,estimate);});
    reject([&]{native_primary_quota(1,512,{SIZE_MAX,1},100,estimate);});
    reject([&]{native_fit_count(SIZE_MAX,100,estimate,2);});
    auto bad=room;bad.available=bad.reserve;reject([&]{bad.tensor_room();});
    bad=room;bad.available=bad.total+1;reject([&]{bad.tensor_room();});
    bad=room;bad.dense=bad.limit;reject([&]{bad.tensor_room();});
    bad=room;bad.tensor_ceiling=1;need(bad.tensor_room()==1);
    std::vector<std::pair<int32_t,int32_t>> ranks{{0,3},{0,2},{1,2},{0,3},{1,1},{0,1},{1,0}};
    auto one=native_ranked_selection(ranks,2,4,1),two=native_ranked_selection(ranks,2,4,2);
    need(one.primary==std::vector<std::pair<int32_t,int32_t>>({{0,3},{1,2}}));
    need(one.remaining.front()==std::make_pair(0,2) && two.remaining.front()==std::make_pair(0,1));
    for(auto p:one.primary)need(std::find(one.remaining.begin(),one.remaining.end(),p)==one.remaining.end());
    auto short_profile=native_ranked_selection({{0,0},{0,1}},2,4,1);need(short_profile.primary.size()==1 && short_profile.remaining.size()==1);
    reject([&]{native_ranked_selection({},2,4,1);});reject([&]{native_ranked_selection({{0,0}},2,4,1);});
    reject([&]{native_ranked_selection({{2,0}},2,4,1);});
    auto path=std::filesystem::temp_directory_path()/("strata-profile-golden-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    auto write=[&](uint32_t version,uint32_t count,bool truncate){FILE* f=fopen(path.c_str(),"wb");need(f);fwrite("STRP",1,4,f);uint32_t h[]{version,2,4,8,count};fwrite(h,4,5,f);uint16_t pairs[]{0,3,1,2};if(!truncate)fwrite(pairs,2,4,f);fclose(f);};
    std::vector<std::pair<int32_t,int32_t>> parsed;int64_t slots=0;std::string error;
    write(1,2,false);need(read_expert_profile(path.string(),2,4,parsed,slots,error) && parsed.size()==2 && slots==8);
    need(read_expert_profile_bounded(path.string(),2,4,parsed,slots,error,8));
    need(!read_expert_profile_bounded(path.string(),2,4,parsed,slots,error,1));
    write(2,2,false);need(!read_expert_profile(path.string(),2,4,parsed,slots,error));
    write(1,2,true);need(!read_expert_profile(path.string(),2,4,parsed,slots,error));
    write(1,9,true);need(!read_expert_profile(path.string(),2,4,parsed,slots,error));
    std::filesystem::remove(path);
    puts("native sizing formula, aligned exact/short room, rank exclusion/reselection, profile and overflow tests passed; no backend/weight allocation");return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
