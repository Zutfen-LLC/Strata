#include "strata/core/layer0_observer_cuda.hpp"
#include <filesystem>
#include <chrono>
using namespace strata::core;
void need(bool b){if(!b)throw std::runtime_error("native layer0 snapshot test failed");}
int main(){try {
#ifndef _WIN32
    auto path=std::filesystem::temp_directory_path()/("strata-l0cuda-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    setenv("STRATA_LAYER0_OBSERVER",path.c_str(),1);setenv("STRATA_LAYER0_BINDING",std::string(64,'a').c_str(),1);
    {
        NativeLayer0Observer observer;need(observer_mock_allocations==1);
        std::string error_for_skipped;
        for(int capacity:{2,4,8})need(observer.snapshot(capacity,Layer0Observer::state_before,nullptr,nullptr,error_for_skipped));
        need(observer_mock_async==0 && observer_mock_downloads==0);
        std::vector<float> state(Layer0Observer::counts[1],0);std::string error;
        need(observer.snapshot(1,Layer0Observer::state_before,state.data(),nullptr,error));
        need(observer_mock_async==1 && observer_mock_downloads==0);
        observer_mock_fail_copy=true;
        need(!observer.snapshot(1,Layer0Observer::state_after,state.data(),nullptr,error));observer_mock_fail_copy=false;
        for(int i=0;i<12;++i)if(i!=Layer0Observer::state_before && i!=Layer0Observer::router_ids)
            need(observer.snapshot(1,Layer0Observer::Field(i),state.data(),nullptr,error));
        std::array<int32_t,10> ids{};for(int i=0;i<10;++i)ids[i]=i;
        need(observer.snapshot(1,Layer0Observer::router_ids,ids.data(),nullptr,error));
        for(auto field:{Layer0Observer::attention_mixed,Layer0Observer::attention_inject})
            need(observer.snapshot(1,field,state.data(),nullptr,error));
        std::array<int32_t,512> res;res.fill(-1);res[7]=42;
        need(observer.flush(248045,res.data(),error));need(observer.finished());
        need(observer_mock_downloads==14);need(observer.flush(248045,res.data(),error));need(observer_mock_downloads==14);
    }
    need(observer_mock_allocations==0);std::filesystem::remove(path);
    {
        NativeLayer0Observer reduced;reduced.set_helper_reduction(true);
        std::vector<float> zeros(786432,0);std::string error;
        need(!reduced.snapshot(1,Layer0Observer::expert_rows,zeros.data(),nullptr,error));
        for(int i=0;i<12;++i)if(i!=Layer0Observer::expert_rows)
            need(reduced.snapshot(1,Layer0Observer::Field(i),zeros.data(),nullptr,error));
        for(auto field:{Layer0Observer::attention_mixed,Layer0Observer::attention_inject})
            need(reduced.snapshot(1,field,zeros.data(),nullptr,error));
        std::array<int32_t,512> primary;primary.fill(-1);
        need(reduced.flush(248045,primary.data(),error));
        need(reduced.finished() && observer_mock_downloads==27);
    }
    need(observer_mock_allocations==0);std::filesystem::remove(path);
    observer_mock_free=(512u<<20);bool failed=false;
    try{NativeLayer0Observer rejected;}catch(const std::exception&){failed=true;}
    need(failed && observer_mock_allocations==0);std::filesystem::remove(path);
    unsetenv("STRATA_LAYER0_OBSERVER");unsetenv("STRATA_LAYER0_BINDING");
#endif
    puts("native observer fake CUDA: pre-sizing storage, snapshots, explicit download, copy refusal and release passed; no device execution");return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
