#include "portable_remote_experts.hpp"
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
namespace strata::core::portable_detail {
bool RemoteExperts::open(worker::GpuWorkerTier::Open open,const worker::GpuPlan& expected,
                         const std::vector<worker::GpuPlan>& local,size_t required_k,Complement complement,SourceGuard guard,std::string& error) {
    error.clear();
    if(ready_){error="remote expert placement already frozen";return false;}
    if(!open.vulkan || !open.native_q8 || !open.entry_rows || !complement || !guard || !required_k || required_k>worker::kMaxK || required_k>open.plan.n_expert){error="remote experts require native Q8 Vulkan indexed rows and a complement";return false;}
    std::vector<uint8_t> a,b;
    if(!worker::encode_gpu_plan(open.plan,a,error)||!worker::encode_gpu_plan(expected,b,error)||a!=b){if(error.empty())error="remote plan/source identity mismatch";return false;}
    for(const auto& p:local) {
        if(!worker::validate_gpu_plan(p,error))return false;
        for(const auto& asset:p.assets)if(open.plan.owns(asset.layer,asset.expert)){error="remote/local expert ownership overlaps";return false;}
    }
    if(!guard(error))return false;
    if(!tier_.open(open,error))return false;
    if(!guard(error)){tier_.close();return false;}
    plan_=std::move(open.plan);required_k_=required_k;complement_=std::move(complement);source_guard_=std::move(guard);ready_=true;return true;
}
bool RemoteExperts::run(int64_t layer,const PortableRoute& route,std::vector<float>& rows,std::string& error) {
    return run_batch(layer,std::vector<PortableRoute>{route},rows,error);
}
bool RemoteExperts::run_batch(int64_t layer,const std::vector<PortableRoute>& routes,std::vector<float>& rows,std::string& error) {
    rows.clear();error.clear();
    try {
        if(!ready_||tier_.dead())throw std::runtime_error("remote expert provider is not ready");
        if(!source_guard_(error))return false;
        const size_t T=routes.size(), H=plan_.n_embd;
        if(layer<0||layer>=plan_.n_layers||T==0||T>worker::kMaxT)throw std::runtime_error("remote route shape invalid");
        const size_t K=routes.front().ids.size();
        if(K!=required_k_)throw std::runtime_error("remote route shape invalid");
        std::vector<int32_t> ids;std::vector<float> weights,x;
        std::vector<uint8_t> remote(T*K),cold(T*K);
        size_t count=0;
        for(size_t t=0;t<T;++t){
            const auto& r=routes[t];
            if(r.ids.size()!=K||r.weights.size()!=K||r.activation.size()!=H)throw std::runtime_error("remote route shape invalid");
            std::set<int32_t> unique;
            for(size_t k=0;k<K;++k){const auto id=r.ids[k];
                if(id<0||id>=plan_.n_expert||!unique.insert(id).second||!std::isfinite(r.weights[k]))throw std::runtime_error("remote route identity invalid");
                const size_t slot=t*K+k;remote[slot]=plan_.owns(int(layer),id);cold[slot]=!remote[slot];count+=remote[slot];
                ids.push_back(id);weights.push_back(r.weights[k]);
            }
            for(float v:r.activation)if(!std::isfinite(v))throw std::runtime_error("remote activation nonfinite");
            x.insert(x.end(),r.activation.begin(),r.activation.end());
        }
        std::vector<float> result(T*K*H,0);
        if(count){
            if(!tier_.post(layer,x.data(),ids.data(),weights.data(),remote.data(),int(T),int(K),error)||!tier_.finish(error))return false;
            if(!tier_.served())throw std::runtime_error("remote reply was not served");
            const auto& reply=tier_.entry_rows();
            if(reply.indices.size()!=count||reply.rows.size()!=count*H)throw std::runtime_error("remote reply shape invalid");
            std::vector<bool> seen(T*K,false);
            for(size_t n=0;n<count;++n){const auto slot=size_t(reply.indices[n]);
                if(slot>=remote.size()||!remote[slot]||seen[slot])throw std::runtime_error("remote reply index invalid");
                seen[slot]=true;
                for(size_t h=0;h<H;++h){float v=reply.rows[n*H+h];if(!std::isfinite(v))throw std::runtime_error("remote reply nonfinite");result[slot*H+h]=v;}
            }
        }
        if(count<T*K){
            std::vector<float> fallback;
            if(!complement_(layer,routes,cold,fallback,error)){if(error.empty())error="remote complement failed";return false;}
            if(fallback.size()!=result.size())throw std::runtime_error("complement reply shape invalid");
            for(size_t i=0;i<cold.size();++i)if(cold[i])for(size_t h=0;h<H;++h){float v=fallback[i*H+h];if(!std::isfinite(v))throw std::runtime_error("complement reply nonfinite");result[i*H+h]=v;}
        }
        if(!source_guard_(error))return false;
        rows=std::move(result);cpu_entries_+=T*K-count;return true;
    }catch(const std::exception& e){rows.clear();error=e.what();return false;}
}
} // namespace strata::core::portable_detail
