#include "strata/core/layer0_observer.hpp"
#include <filesystem>
#include <vector>
#include <chrono>
using namespace strata::core;
void need(bool b){if(!b)throw std::runtime_error("layer0 observer test failed");}
template<class F>void reject(F f){bool bad=false;try{f();}catch(const std::exception&){bad=true;}need(bad);}
int main(){try {
#ifndef _WIN32
    unsetenv("STRATA_LAYER0_OBSERVER");Layer0Observer disabled;disabled.begin(1);need(!disabled.active());
    auto path=std::filesystem::temp_directory_path()/("strata-layer0-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    setenv("STRATA_LAYER0_OBSERVER",path.c_str(),1);unsetenv("STRATA_LAYER0_BINDING");reject([]{Layer0Observer invalid;});
    need(!std::filesystem::exists(path));setenv("STRATA_LAYER0_BINDING",std::string(64,'a').c_str(),1);
    {Layer0Observer o;o.begin(248045);reject([&]{o.begin(1);});reject([&]{o.put(Layer0Observer::embedding,nullptr,1);});
     std::vector<float> f(10240,1);o.put(Layer0Observer::embedding,f.data(),f.size());
     reject([&]{o.put(Layer0Observer::embedding,f.data(),f.size());});reject([&]{o.finish();});
     std::vector<float> zeros(786432,0);
     for(size_t i=0;i<Layer0Observer::counts.size();++i)if(i!=Layer0Observer::token && i!=Layer0Observer::embedding &&
        i!=Layer0Observer::state_after && i!=Layer0Observer::history_after && i!=Layer0Observer::materialized_post_residual)
        o.put(Layer0Observer::Field(i),zeros.data(),Layer0Observer::counts[i]);
     o.finish();need(o.finished()&&!o.active());}
    reject([]{Layer0Observer overwrite;});need(std::filesystem::file_size(path)<Layer0Observer::limit);
    std::filesystem::remove(path);unsetenv("STRATA_LAYER0_OBSERVER");unsetenv("STRATA_LAYER0_BINDING");
#endif
    std::vector<uint32_t> native(786432),canonical(native.size());
    for(size_t i=0;i<native.size();++i)native[i]=uint32_t(i);
    layer0_native_state_view(native.data(),canonical.data(),native.size());
    need(canonical[1]==6144 && canonical[128]==1 && canonical[16384]==128 && canonical.back()==786431);
    std::vector<bool> used(native.size());for(auto word:canonical){need(word<native.size() && !used[word]);used[word]=true;}
    reject([&]{layer0_native_state_view(native.data(),canonical.data(),1);});
    size_t n=0;for(auto c:Layer0Observer::counts)n+=c;need(n*4<Layer0Observer::limit);
    puts("layer0 disabled, exclusive creation, binding, shape, duplicates and storage bound passed");return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
