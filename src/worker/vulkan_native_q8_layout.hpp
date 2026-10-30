#pragma once
#include "vulkan_expert_graph.hpp"
#include "strata/worker/gpu_worker.hpp"
#include <algorithm>
#include <ostream>
namespace strata::worker::vulkan_detail {
inline void native_q8_metadata_header(std::ostream& out,size_t record,const GpuPlan& p,const InferRequest& r) {
    out<<"record "<<record<<" model "<<p.model_id<<" placement "<<p.placement_id.a<<' '<<p.placement_id.b
       <<" layer "<<r.layer<<" T "<<unsigned(r.T)<<" K "<<unsigned(r.K)<<'\n';
}
inline ggml_tensor* native_q8_view(ggml_context* context,ggml_tensor* base,size_t count) {
    auto* result=ggml_view_1d(context,base,count,0);
    require(ggml_backend_view_init(result)==GGML_STATUS_SUCCESS,"native Q8 backend view initialization failed");
    return result;
}
inline size_t native_q8_weight_bytes(size_t slots,size_t budget) {
    require(slots>=1 && slots<=24,"native Q8 requires 1..24 frozen slots");
    const size_t bytes=slots*size_t(5222400);
    require(bytes<=budget,"native Q8 weights exceed tensor budget");return bytes;
}
// Same first-occurrence group order and stable entry order as gpu_compute.cu.
// Capacities belong to the existing Vulkan components; never split a call.
inline GpuExecutionTrace native_q8_groups(const GpuPlan& p,const InferRequest& r) {
    require(r.T>=1 && r.T<=8 && r.K>=1 && r.K<=10 && r.mask.size()==size_t(r.T)*r.K &&
        r.ids.size()==r.mask.size(),"native Q8 request shape");
    GpuExecutionTrace t;std::vector<int32_t> group(r.mask.size(),-1);
    for(size_t i=0;i<r.mask.size();++i) {
        require(r.mask[i]<=1,"native Q8 nonbinary mask");
        if(!r.mask[i])continue;
        auto it=std::find(t.experts.begin(),t.experts.end(),r.ids[i]);
        if(it==t.experts.end()) {
            const auto* a=p.find(r.layer,r.ids[i]);require(a,"native Q8 unassigned expert");
            t.experts.push_back(r.ids[i]);t.asset_slots.push_back(int32_t(a-p.assets.data()));
            group[i]=int32_t(t.experts.size()-1);
        } else group[i]=int32_t(it-t.experts.begin());
    }
    require(!t.experts.empty() && t.experts.size()<=24,"native Q8 supports 1..24 active groups; no call splitting");
    for(size_t g=0;g<t.experts.size();++g) {
        t.start.push_back(int32_t(t.dst.size()));
        for(size_t i=0;i<group.size();++i)if(group[i]==int32_t(g)) {
            t.original.push_back(int32_t(i));t.dst.push_back(int32_t(t.dst.size()));t.tok.push_back(int32_t(i/r.K));
        }
    }
    t.start.push_back(int32_t(t.dst.size()));
    require(t.dst.size()<=64,"native Q8 supports at most 64 active entries; no call splitting");
    return t;
}
}
