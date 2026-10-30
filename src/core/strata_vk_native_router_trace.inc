#include "strata_vk_native_router_trace.hpp"
#include "strata_native_router_trace_spv.hpp"
namespace strata::core::portable_detail {
StrataVkNativeRouter* strata_vk_native_router_trace_create(ggml_backend_t backend,const char* path) {
    auto result=std::make_unique<StrataVkNativeRouter>();
    result->owner=strata_vk_q8_component_owner(backend,strata_native_router_trace_spv,sizeof(strata_native_router_trace_spv)/4,
        NativeComponentAddressing::descriptor,"strata_native_router_row2_diagnostic",4,4,path);
    return result.release();
}
void strata_vk_native_router_trace_run(StrataVkNativeRouter* s,const ggml_tensor* logits,
    ggml_tensor* ids,ggml_tensor* weights,uint32_t* trace26) {
    logits_guard_need(s && trace26,"router trace: invalid call");
    std::array<uint32_t,128> before,after;before.fill(0xa5a5a5a5);
    ggml_backend_tensor_set(s->owner->status,before.data(),0,sizeof(before));
    // Same T8 dispatch, status reset/check, synchronization and tensor checks.
    strata_vk_native_router_run(s,logits,ids,weights,8);
    ggml_backend_tensor_get(s->owner->status,after.data(),0,sizeof(after));
    for(size_t i=27;i<after.size();++i)logits_guard_need(after[i]==before[i],"router trace: status/trace guard mutation");
    std::copy(after.begin()+1,after.begin()+27,trace26);
}
}
