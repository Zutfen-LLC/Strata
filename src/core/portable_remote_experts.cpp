#include "portable_remote_experts.hpp"
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
namespace strata::core::portable_detail {
bool RemoteExperts::open(worker::GpuWorkerTier::Open open,const worker::GpuPlan& expected,
                         const std::vector<worker::GpuPlan>& local,size_t required_k,Complement complement,SourceGuard guard,std::string& error) {
    error.clear();
    if(ready_ || group_ || terminal_){error="remote expert placement already frozen or terminal";return false;}
    if(!open.vulkan || !open.native_q8 || !open.entry_rows || !complement || !guard || !required_k || required_k>worker::kMaxK || required_k>open.plan.n_expert){error="remote experts require native Q8 Vulkan indexed rows and a complement";return false;}
    std::vector<uint8_t> a,b;
    if(!worker::encode_gpu_plan(open.plan,a,error)||!worker::encode_gpu_plan(expected,b,error)||a!=b){if(error.empty())error="remote plan/source identity mismatch";return false;}
    for(const auto& p:local) {
        if(!worker::validate_gpu_plan(p,error))return false;
        for(const auto& asset:p.assets)if(open.plan.owns(asset.layer,asset.expert)){error="remote/local expert ownership overlaps";return false;}
    }
    if(!guard(error))return false;
    single_snapshot_.host=open.host;single_snapshot_.port=open.port;single_snapshot_.setup_attempted=true;
    if(!tier_.open(open,error))return false;
    if(!guard(error)){tier_.close();return false;}
    plan_=std::move(open.plan);required_k_=required_k;complement_=std::move(complement);source_guard_=std::move(guard);ready_=true;return true;
}
bool RemoteExperts::open(std::vector<worker::GpuWorkerTier::Open> opens,const std::vector<worker::GpuPlan>& expected,
                         const std::vector<worker::GpuPlan>& local,size_t required_k,Complement complement,SourceGuard guard,std::string& error) {
    error.clear();
    if(group_ || terminal_){error="remote expert provider is terminal";return false;}
    if(opens.size()==1 && expected.size()==1)return open(std::move(opens.front()),expected.front(),local,required_k,std::move(complement),std::move(guard),error);
    if(ready_ || group_ || opens.empty() || opens.size()!=expected.size() || !complement || !guard || !required_k || required_k>worker::kMaxK){error="invalid remote group configuration";return false;}
    for(size_t i=0;i<opens.size();++i) {
        auto& o=opens[i];std::vector<uint8_t> a,b;
        if(!o.vulkan || !o.native_q8 || !o.entry_rows || required_k>o.plan.n_expert ||
           !worker::encode_gpu_plan(o.plan,a,error) || !worker::encode_gpu_plan(expected[i],b,error) || a!=b){
            if(error.empty())error="remote group plan/numerics mismatch";
            return false;
        }
        for(const auto& p:local){if(!worker::validate_gpu_plan(p,error))return false;
            for(const auto& asset:p.assets)if(o.plan.owns(asset.layer,asset.expert)){error="remote/local expert ownership overlaps";return false;}}
    }
    if(!worker::GpuWorkerGroup::validate(opens,error) || !guard(error))return false;
    group_=std::make_unique<worker::GpuWorkerGroup>();
    if(!group_->open(opens,error))return false;
    if(!guard(error)){group_->close();return false;}
    plan_=opens.front().plan;required_k_=required_k;
    complement_=std::move(complement);source_guard_=std::move(guard);ready_=true;return true;
}
bool RemoteExperts::status(worker::GpuCounters& out,std::string& error) {
    if(!group_){const bool ok=tier_.status(out,error);if(ok){single_snapshot_.worker=out;single_snapshot_.worker_status_observed=true;}return ok;}
    std::vector<worker::GpuCounters> values;if(!group_->status(values,error)){terminal_=true;group_->close();return false;}
    out={};for(const auto& v:values){
        out.assigned_bytes+=v.assigned_bytes;out.cached_bytes+=v.cached_bytes;out.transferred_bytes+=v.transferred_bytes;
        out.persisted_bytes+=v.persisted_bytes;out.retained_disk_bytes+=v.retained_disk_bytes;out.verified_bytes+=v.verified_bytes;
        out.reused_bytes+=v.reused_bytes;out.host_staging_peak+=v.host_staging_peak;out.gpu_allocated_bytes+=v.gpu_allocated_bytes;
        out.gpu_peak_bytes+=v.gpu_peak_bytes;out.gpu_weight_bytes+=v.gpu_weight_bytes;out.gpu_scratch_bytes+=v.gpu_scratch_bytes;
        out.gpu_allocation_count+=v.gpu_allocation_count;out.gpu_free_before+=v.gpu_free_before;out.gpu_free_after+=v.gpu_free_after;
        out.gpu_runtime_reserve+=v.gpu_runtime_reserve;out.requests+=v.requests;out.entries+=v.entries;out.failures+=v.failures;out.service_ns+=v.service_ns;
    }
    return true;
}
std::vector<worker::GpuPeerSnapshot> RemoteExperts::snapshots() const {
    if(group_)return group_->snapshots();
    if(single_snapshot_.host.empty())return {};
    auto s=single_snapshot_;s.client=tier_.stats();s.failure=tier_.dead_why();return {s};
}
bool RemoteExperts::run(int64_t layer,const PortableRoute& route,std::vector<float>& rows,std::string& error) {
    return run_batch(layer,std::vector<PortableRoute>{route},rows,error);
}
bool RemoteExperts::run_batch(int64_t layer,const std::vector<PortableRoute>& routes,std::vector<float>& rows,std::string& error) {
    rows.clear();error.clear();bool assembling_remote=false;
    try {
        if(!usable()){
            if(group_ && group_->dead()){terminal_=true;group_->close();}
            throw std::runtime_error("remote expert provider is not ready");
        }
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
                const size_t slot=t*K+k;remote[slot]=group_ ? group_->holds(int(layer),id) : plan_.owns(int(layer),id);cold[slot]=!remote[slot];count+=remote[slot];
                ids.push_back(id);weights.push_back(r.weights[k]);
            }
            for(float v:r.activation)if(!std::isfinite(v))throw std::runtime_error("remote activation nonfinite");
            x.insert(x.end(),r.activation.begin(),r.activation.end());
        }
        std::vector<float> result(T*K*H,0);
        if(count){
            assembling_remote=true;
            if(group_){
                if(!group_->post(layer,x.data(),ids.data(),weights.data(),remote.data(),int(T),int(K),error) || !group_->finish(error)){terminal_=true;return false;}
            } else if(!tier_.post(layer,x.data(),ids.data(),weights.data(),remote.data(),int(T),int(K),error)||!tier_.finish(error)){terminal_=true;return false;}
            std::vector<bool> seen(T*K,false);size_t received=0;
            const size_t peers=group_?group_->size():1;
            for(size_t p=0;p<peers;++p){
                if(group_ && !group_->active(p))continue;
                if(!group_ && !tier_.served())throw std::runtime_error("remote reply was not served");
                const auto& reply=group_?group_->entry_rows(p):tier_.entry_rows();
                if(reply.rows.size()!=reply.indices.size()*H)throw std::runtime_error("remote reply shape invalid");
                for(size_t n=0;n<reply.indices.size();++n){const auto slot=size_t(reply.indices[n]);
                    if(slot>=remote.size()||!remote[slot]||seen[slot]||(group_ && !group_->mask(p)[slot]))throw std::runtime_error("remote reply index invalid");
                    seen[slot]=true;++received;
                    for(size_t h=0;h<H;++h){float v=reply.rows[n*H+h];if(!std::isfinite(v))throw std::runtime_error("remote reply nonfinite");result[slot*H+h]=v;}
                }
            }
            if(received!=count)throw std::runtime_error("remote reply missing entries");
            assembling_remote=false;
        }
        if(count<T*K){
            std::vector<float> fallback;
            if(!complement_(layer,routes,cold,fallback,error)){if(error.empty())error="remote complement failed";return false;}
            if(fallback.size()!=result.size())throw std::runtime_error("complement reply shape invalid");
            for(size_t i=0;i<cold.size();++i)if(cold[i])for(size_t h=0;h<H;++h){float v=fallback[i*H+h];if(!std::isfinite(v))throw std::runtime_error("complement reply nonfinite");result[i*H+h]=v;}
        }
        if(!source_guard_(error))return false;
        if(!usable()){terminal_=true;if(group_)group_->close();else tier_.close();throw std::runtime_error("remote expert provider failed before result publication");}
        rows=std::move(result);complement_entries_+=T*K-count;return true;
    }catch(const std::exception& e){if(assembling_remote){terminal_=true;if(group_)group_->close();else tier_.close();}rows.clear();error=e.what();return false;}
}
} // namespace strata::core::portable_detail
