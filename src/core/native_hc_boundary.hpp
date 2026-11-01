#pragma once
#include <stdexcept>
namespace strata::core::portable_detail {
// Native Verifier::record_window pre/post boundary rules. A partial port must
// materialize at its handoff; neither a head nor PLE consumes a stale tuple.
struct NativeHcBoundary {
    bool pending_attention;
    bool defer_ffn;
};
inline NativeHcBoundary native_hc_boundary(int layer,int layers=48,bool ple=true,
    bool previous_control=false,bool current_control=false,bool handoff=false) {
    if(layers<1 || layer<0 || layer>=layers)throw std::runtime_error("native HC: invalid layer boundary");
    return {layer>0 && !previous_control && !(layer==1 && ple),
            layer+1<layers && !current_control && !handoff && !(layer==0 && ple)};
}
} // namespace strata::core::portable_detail
