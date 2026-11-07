#pragma once
// Standalone FFN row arithmetic; not an attention boundary or model admission.
#include "portable_native_ffn_projection.hpp"
#include "portable_native_router.hpp"
#include "portable_native_shared.hpp"
#include "portable_native_moe_combine.hpp"
#include <algorithm>
#include <array>
#include <functional>
#include <utility>

namespace strata::core::portable_detail {
class PortableNativeFfnRows {
public:
    struct Dispatch {
        std::function<void(size_t,const ggml_tensor*,std::vector<PortableRoute>&)> router;
        std::function<std::vector<float>(size_t,const ggml_tensor*)> shared;
        std::function<std::vector<float>(size_t,const std::vector<float>&,const std::vector<float>&,const std::vector<float>&)> combine;
    };
    // All six owners are borrowed, serialized and must outlive this dispatch.
    // The same fixed owners are reused for every call of their shape.
    static Dispatch native_dispatch(const std::array<PortableNativeRouter*,2>& router,
                                    const std::array<PortableNativeShared*,2>& shared,
                                    const std::array<PortableNativeMoeCombine*,2>& combine) {
        for(size_t i=0;i<2;++i)need(router[i]&&shared[i]&&combine[i],"FFN rows: missing native owner");
        return {[router](size_t n,const ggml_tensor* t,std::vector<PortableRoute>& r){router[n-1]->run(t,r);},
                [shared](size_t n,const ggml_tensor* t){return shared[n-1]->run(t);},
                [combine](size_t n,const std::vector<float>& r,const std::vector<float>& w,const std::vector<float>& s){return combine[n-1]->run(r,w,s);}};
    }
    struct Result {std::vector<PortableRoute> routes;std::vector<float> block;};
private:
    PortableNativeFfnProjection& projection_; // borrowed; no concurrent use
    Dispatch dispatch_;
    int64_t layer_;
    bool running_=false,poisoned_=false;
    uint64_t generation_=0;
    static void need(bool b,const char* m){if(!b)throw std::runtime_error(m);}
    static void finite(const std::vector<float>& v){for(float x:v)need(std::isfinite(x),"FFN rows: nonfinite value");}
public:
    PortableNativeFfnRows(int64_t layer,PortableNativeFfnProjection& projection,Dispatch dispatch)
      :projection_(projection),dispatch_(std::move(dispatch)),layer_(layer){
        need(layer>=0&&layer<48&&dispatch_.router&&dispatch_.shared&&dispatch_.combine,"FFN rows: invalid construction");
    }
    PortableNativeFfnRows(const PortableNativeFfnRows&)=delete;
    PortableNativeFfnRows& operator=(const PortableNativeFfnRows&)=delete;
    int64_t layer()const{return layer_;}
    bool poisoned()const{return poisoned_;}
    uint64_t generation()const{return generation_;}
    // Reset only after the enclosing producer/provider has also been reset or replayed.
    void reset(){need(!running_&&generation_!=UINT64_MAX,"FFN rows: active reset or generation exhausted");++generation_;poisoned_=false;}
    Result run(const std::vector<float>& mixed,size_t count,const PortableExpertRowsBatch& provider){
        need(!running_&&!poisoned_&&generation_!=UINT64_MAX&&(count==1||count==2)&&mixed.size()==count*2560&&bool(provider),"FFN rows: unavailable or shape");
        finite(mixed);++generation_;running_=true;struct Guard{bool& flag;~Guard(){flag=false;}}guard{running_};
        try {
            projection_.run(mixed,count);
            Result result;result.routes.resize(count);
            for(size_t t=0;t<count;++t)result.routes[t].activation.assign(mixed.begin()+t*2560,mixed.begin()+(t+1)*2560);
            dispatch_.router(count,projection_.output(count),result.routes);
            need(result.routes.size()==count,"FFN rows: router count");
            std::vector<float> weights;weights.reserve(count*10);
            for(size_t t=0;t<count;++t){auto& r=result.routes[t];need(r.activation.size()==2560 &&
                std::equal(r.activation.begin(),r.activation.end(),mixed.begin()+t*2560) &&
                r.ids.size()==10&&r.weights.size()==10&&r.native_router,"FFN rows: invalid route");
                std::array<bool,512> seen{};bool positive=false;
                for(size_t k=0;k<10;++k){auto id=r.ids[k];auto w=r.weights[k];need(id>=0&&id<512&&!seen[size_t(id)]&&std::isfinite(w)&&w>=0,"FFN rows: invalid route slot");seen[size_t(id)]=true;positive|=w>0;weights.push_back(w);}
                need(positive,"FFN rows: zero route");
            }
            auto shared=dispatch_.shared(count,projection_.mixed(count));need(shared.size()==count*2560,"FFN rows: shared shape");finite(shared);
            for(auto& r:result.routes)r.native_shared=true;
            std::vector<float> rows;std::string error;
            const bool accepted=provider(layer_,result.routes,rows,error);
            need(accepted,error.empty()?"FFN rows: provider rejected":error.c_str());
            need(rows.size()==count*10*2560,"FFN rows: provider shape");finite(rows);
            result.block=dispatch_.combine(count,rows,weights,shared);
            need(result.block.size()==count*2560,"FFN rows: combine shape");finite(result.block);
            return result;
        }catch(...){poisoned_=true;throw;}
    }
};
}
