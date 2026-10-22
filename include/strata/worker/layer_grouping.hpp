// Offline arithmetic/selection diagnostics only. Never included by the engine.
#pragma once
#include "strata/worker/layer_trace.hpp"
#include "strata/worker/expert_worker.hpp"
namespace strata::worker::diagnostic::grouping {
struct Observation { int layer; bool input_bitwise, output_bitwise; };
inline int seed(const std::vector<Observation>& rows) {
    int selected=-1;
    for(const auto& row:rows) if(row.input_bitwise&&!row.output_bitwise&&
            (selected<0||row.layer<selected))selected=row.layer;
    return selected;
}
inline bool exact(const std::vector<float>& a,const std::vector<float>& b) {
    return a.size()==b.size()&&std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0;
}
// Explicit rounding prevents the diagnostic compiler from reassociating groups.
inline float add(float a,float b){volatile float value=a+b;return value;}
inline float multiply(float a,float b){volatile float value=a*b;return value;}
enum class Local { NativeFma, Float64 };
inline const char* name(Local model){return model==Local::NativeFma?"native_fma_shared_last":"float64_shared_last";}
inline std::vector<float> local(const std::vector<float>& rows,const std::vector<float>& weights,
                                const std::vector<float>& shared,Local model) {
    const size_t H=shared.size(),K=weights.size();
    if(!H||!K||K>10||rows.size()!=H*K)throw std::runtime_error("grouping local shape");
    std::vector<float> result(H);
    for(size_t h=0;h<H;++h){
        if(model==Local::NativeFma){
            float sum=multiply(rows[h],weights[0]);
            for(size_t i=1;i<K;++i)sum=std::fma(rows[i*H+h],weights[i],sum);
            result[h]=add(sum,shared[h]);
        }else{
            double sum=0;
            for(size_t i=0;i<K;++i){volatile double product=(double)weights[i]*rows[i*H+h];sum+=product;}
            result[h]=(float)(sum+(double)shared[h]);
        }
    }
    return result;
}
struct Groups {std::vector<float> local,helper,cpu,gpu,remote,final;};
inline size_t moved_cpu_to_gpu(const std::vector<int32_t>& baseline,const std::vector<int32_t>& distributed) {
    if(baseline.empty()||baseline.size()!=distributed.size())throw std::runtime_error("four-way owner shape");
    size_t moved=baseline.size();
    for(size_t i=0;i<baseline.size();++i){
        if(baseline[i]<-1||baseline[i]>3||distributed[i]<-1||distributed[i]>4)
            throw std::runtime_error("four-way baseline/owner bounds");
        if(baseline[i]!=distributed[i]){
            if(moved!=baseline.size()||baseline[i]!=3||distributed[i]!=4)
                throw std::runtime_error("four-way requires one CPU-to-GPU entry and no other ownership change");
            moved=i;
        }
    }
    if(moved==baseline.size())throw std::runtime_error("four-way CPU-to-GPU entry unavailable");
    return moved;
}
inline Groups anchored(const std::vector<float>& rows,const std::vector<float>& weights,
                       const std::vector<int32_t>& owners,const std::vector<float>& captured_local,
                       const std::vector<float>& captured_helper) {
    const size_t H=captured_local.size(),K=weights.size();
    if(!H||!K||K>10||rows.size()!=H*K||owners.size()!=K||captured_helper.size()!=H)
        throw std::runtime_error("four-way anchored shape");
    Groups result;result.local=captured_local;result.helper=captured_helper;
    std::vector<uint8_t> mask(K);
    for(int owner:{3,4}){
        for(size_t i=0;i<K;++i){
            if(owners[i]<-1||owners[i]>4)throw std::runtime_error("four-way owner bounds");
            mask[i]=owners[i]==owner;
        }
        auto& partial=owner==3?result.cpu:result.gpu;partial.resize(H);
        ExpertWorkerService::accumulate_partial(1,(int)K,mask.data(),weights.data(),rows.data(),(int)H,partial.data());
    }
    result.remote.resize(H);result.final.resize(H);
    for(size_t h=0;h<H;++h){
        result.remote[h]=add(add(result.helper[h],result.gpu[h]),result.cpu[h]);
        result.final[h]=add(result.local[h],result.remote[h]);
    }
    return result;
}
struct Effects {
    std::vector<double> row_baseline,row_distributed,group_cpu,group_cuda,row_main,group_main,interaction;
};
inline Effects effects(const std::vector<float>& cpu_baseline,const std::vector<float>& cpu_distributed,
                       const std::vector<float>& cuda_baseline,const std::vector<float>& cuda_distributed) {
    const size_t n=cpu_baseline.size();
    if(!n||cpu_distributed.size()!=n||cuda_baseline.size()!=n||cuda_distributed.size()!=n)
        throw std::runtime_error("four-way effects shape");
    Effects result;
    for(size_t i=0;i<n;++i){
        const double cb=cpu_baseline[i],cg=cpu_distributed[i],gb=cuda_baseline[i],gg=cuda_distributed[i];
        if(!std::isfinite(cb)||!std::isfinite(cg)||!std::isfinite(gb)||!std::isfinite(gg))
            throw std::runtime_error("four-way nonfinite effects");
        const double rb=gb-cb,rg=gg-cg,gc=cg-cb,ggroup=gg-gb;
        result.row_baseline.push_back(rb);result.row_distributed.push_back(rg);
        result.group_cpu.push_back(gc);result.group_cuda.push_back(ggroup);
        result.row_main.push_back((rb+rg)/2);result.group_main.push_back((gc+ggroup)/2);
        result.interaction.push_back(rg-rb);
    }
    return result;
}
inline Groups reduce(const std::vector<float>& rows,const std::vector<float>& weights,
                     const std::vector<int32_t>& owners,const std::vector<float>& shared,
                     Local model,const std::vector<float>* actual_helper=nullptr) {
    const size_t H=shared.size(),K=weights.size();
    if(owners.size()!=K||rows.size()!=H*K||!H||!K||K>10)throw std::runtime_error("grouping partition shape");
    std::vector<float> local_rows(rows.size(),0.f);std::vector<uint8_t> mask(K);
    Groups result;
    for(size_t i=0;i<K;++i){
        if(owners[i]<-1||owners[i]>4)throw std::runtime_error("grouping owner bounds");
        if(owners[i]<=1)std::copy_n(rows.data()+i*H,H,local_rows.data()+i*H);
    }
    result.local=local(local_rows,weights,shared,model);
    for(int owner:{2,3,4}){
        for(size_t i=0;i<K;++i)mask[i]=owners[i]==owner;
        auto& partial=owner==2?result.helper:owner==3?result.cpu:result.gpu;
        partial.resize(H);
        // Reuse the actual worker entry-order reduction, not a new reference.
        ExpertWorkerService::accumulate_partial(1,(int)K,mask.data(),weights.data(),rows.data(),(int)H,partial.data());
    }
    if(actual_helper){if(actual_helper->size()!=H)throw std::runtime_error("helper anchor shape");result.helper=*actual_helper;}
    result.remote.resize(H);result.final.resize(H);
    for(size_t h=0;h<H;++h){
        result.remote[h]=add(add(result.helper[h],result.gpu[h]),result.cpu[h]);
        result.final[h]=add(result.local[h],result.remote[h]);
    }
    return result;
}
struct Delta {size_t equal=0;double actual_max=0,predicted_max=0,residual_max=0;};
inline Delta delta(const std::vector<float>& actual_b,const std::vector<float>& actual_g,
                   const std::vector<float>& predicted_b,const std::vector<float>& predicted_g) {
    const size_t n=actual_b.size();
    if(!n||actual_g.size()!=n||predicted_b.size()!=n||predicted_g.size()!=n)throw std::runtime_error("delta shape");
    Delta result;
    for(size_t i=0;i<n;++i){
        const double actual=(double)actual_g[i]-actual_b[i],predicted=(double)predicted_g[i]-predicted_b[i];
        if(!std::isfinite(actual)||!std::isfinite(predicted))throw std::runtime_error("nonfinite grouping delta");
        result.equal+=actual==predicted;result.actual_max=std::max(result.actual_max,std::fabs(actual));
        result.predicted_max=std::max(result.predicted_max,std::fabs(predicted));
        result.residual_max=std::max(result.residual_max,std::fabs(actual-predicted));
    }
    return result;
}
} // namespace strata::worker::diagnostic::grouping
