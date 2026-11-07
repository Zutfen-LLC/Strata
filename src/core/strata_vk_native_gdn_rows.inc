#include "strata_vk_native_gdn_rows.hpp"
#include "native_gdn_domain.hpp"
#include "strata_native_gdn_rows_spv.hpp"
namespace strata::core::portable_detail {
struct StrataVkNativeGdnRows {std::unique_ptr<StrataVkLogitsGuard> owner;size_t budget=0;};
StrataVkNativeGdnRows* strata_vk_native_gdn_rows_create(ggml_backend_t b,const char*path,size_t budget){
 native_hc_need(b&&ggml_backend_is_vk(b),"GDN rows: Vulkan backend required");auto*ctx=static_cast<ggml_backend_vk_context*>(b->context);const auto&l=ctx->device->properties.limits;
 native_hc_need(l.maxComputeWorkGroupInvocations>=512&&l.maxComputeWorkGroupSize[0]>=128&&l.maxComputeWorkGroupSize[1]>=4&&l.maxComputeSharedMemorySize>=6160&&l.maxComputeWorkGroupCount[0]>=48&&l.maxPerStageDescriptorStorageBuffers>=11&&l.maxDescriptorSetStorageBuffers>=11,"GDN rows: device limits");
 auto p=std::make_unique<StrataVkNativeGdnRows>();p->budget=budget;p->owner=strata_vk_q8_component_owner(b,strata_native_gdn_rows_spv,sizeof(strata_native_gdn_rows_spv)/4,NativeComponentAddressing::descriptor,"strata_native_gdn_rows",11,15,path,budget);return p.release();
}
void strata_vk_native_gdn_rows_destroy(StrataVkNativeGdnRows*p)noexcept{delete p;}
size_t strata_vk_native_gdn_rows_bytes(const StrataVkNativeGdnRows*p){return p?p->owner->bytes:0;}
#include "strata_vk_native_gdn_rows_run.inc"
}
