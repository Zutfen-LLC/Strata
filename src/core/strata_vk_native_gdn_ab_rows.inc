#include "strata_vk_native_gdn_ab_rows.hpp"
#include "native_llvm_log1p_spirv.hpp"
#include "strata_native_gdn_ab_rows_spv.hpp"
namespace strata::core::portable_detail {
struct StrataVkNativeGdnAbRows { std::unique_ptr<StrataVkLogitsGuard> owner;size_t budget=0; };
StrataVkNativeGdnAbRows* strata_vk_native_gdn_ab_rows_create(ggml_backend_t b,const char*path,size_t budget){
 native_hc_need(b&&ggml_backend_is_vk(b),"GDN AB rows: Vulkan required");auto*ctx=static_cast<ggml_backend_vk_context*>(b->context);const auto&l=ctx->device->properties.limits;
 native_hc_need(l.maxComputeWorkGroupInvocations>=64&&l.maxComputeWorkGroupSize[0]>=64&&l.maxComputeWorkGroupCount[0]>=48&&l.maxComputeWorkGroupCount[1]>=2&&l.maxPerStageDescriptorStorageBuffers>=9&&l.maxDescriptorSetStorageBuffers>=9,"GDN AB rows: unsupported device");
 auto fp=ctx->device->physical_device.getProperties2<vk::PhysicalDeviceProperties2,vk::PhysicalDeviceVulkan12Properties>().get<vk::PhysicalDeviceVulkan12Properties>();bool physical=ctx->device->physical_device.getFeatures().shaderFloat64;bool enabled=static_cast<const StrataVkQ8Device*>(ctx->device.get())->strata_float64_enabled;llvm_log1p_require(physical,enabled,fp);
 printf("native_llvm_log1p physical_float64=%d logical_float64=%d rte64=%d signed64=%d nonfma=true\n",int(physical),int(enabled),int(fp.shaderRoundingModeRTEFloat64),int(fp.shaderSignedZeroInfNanPreserveFloat64));
 auto code=llvm_log1p_modes(strata_native_gdn_ab_rows_spv,sizeof(strata_native_gdn_ab_rows_spv)/4);auto p=std::make_unique<StrataVkNativeGdnAbRows>();p->budget=budget;p->owner=strata_vk_q8_component_owner(b,code.data(),code.size(),NativeComponentAddressing::descriptor,"strata_native_gdn_ab_rows",9,9,path,budget);return p.release();
}
void strata_vk_native_gdn_ab_rows_destroy(StrataVkNativeGdnAbRows*p)noexcept{delete p;}
size_t strata_vk_native_gdn_ab_rows_bytes(const StrataVkNativeGdnAbRows*p){return p?p->owner->bytes:0;}
#include "strata_vk_native_gdn_ab_rows_run.inc"
}
