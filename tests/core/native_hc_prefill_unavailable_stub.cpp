// Legacy-only CPU fixtures explicitly refuse selected Vulkan scalar creation.
#include "strata_vk_native_hc.hpp"
namespace strata::core::portable_detail {
struct StrataVkNativeHcPrefill {};
StrataVkNativeHcPrefill* strata_vk_native_hc_prefill_create(ggml_backend_t,const char*,size_t){throw std::runtime_error("prefill scalar unavailable in this CPU fixture");}
size_t strata_vk_native_hc_prefill_bytes(const StrataVkNativeHcPrefill*){return 0;}
void strata_vk_native_hc_prefill_destroy(StrataVkNativeHcPrefill*p) noexcept{delete p;}
void strata_vk_native_hc_prefill_run(StrataVkNativeHcPrefill*,uint32_t,uint32_t,const std::array<ggml_tensor*,6>&){throw std::runtime_error("prefill scalar unavailable in this CPU fixture");}
}
