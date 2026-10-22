// Native-OFF must link the real shared dispatcher without any worker library.
// An opaque nonnull worker pointer is refused before activation/source/device access.
#include "strata/core/expert_source.hpp"
#include <cstdio>
#include <cstring>

#ifdef STRATA_NATIVE_EXPERTS
#error This regression must be built with STRATA_NATIVE_EXPERTS=OFF
#endif

int main() {
    strata::core::ExpertDispatch dispatch;
    alignas(void*) unsigned char opaque_worker[sizeof(void*)]{};
    dispatch.worker = reinterpret_cast<strata::worker::ExpertWorkerTier*>(opaque_worker);
    dispatch.layers = 7;
    float output[] = {123.0f, -456.0f};
    // Null inputs and no pool/source/plan: touching any of them before refusal would fail.
    strata::core::expert_pool_dispatch_multi(dispatch, nullptr, nullptr, 1, 1, output);
    bool ok = dispatch.failed && dispatch.fail != nullptr &&
                    std::strstr(dispatch.fail, "requires STRATA_NATIVE_EXPERTS=ON") &&
                    dispatch.fail_layer == 7 && dispatch.layers == 7 &&
                    output[0] == 123.0f && output[1] == -456.0f;
    // The network-GPU pointer is opaque too: native-OFF must refuse before dereferencing either tier.
    dispatch.worker = nullptr;
    dispatch.gpu_worker = reinterpret_cast<strata::worker::GpuWorkerTier*>(opaque_worker);
    dispatch.failed = false;
    dispatch.fail = nullptr;
    strata::core::expert_pool_dispatch_multi(dispatch, nullptr, nullptr, 1, 1, output);
    ok = ok && dispatch.failed && dispatch.fail != nullptr &&
         std::strstr(dispatch.fail, "requires STRATA_NATIVE_EXPERTS=ON") &&
         dispatch.fail_layer == 7 && dispatch.layers == 7 &&
         output[0] == 123.0f && output[1] == -456.0f;
    std::puts(ok ? "PASS native-OFF worker dispatch refuses before resource access" :
                   "FAIL native-OFF worker dispatch refusal");
    return ok ? 0 : 1;
}
