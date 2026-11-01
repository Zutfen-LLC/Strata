#include "strata_vk_native_llvm_log1p.hpp"
#include "native_gdn_log1p_domain.hpp"
#include "native_llvm_log1p_spirv.hpp"
#include "strata_native_llvm_log1p_spv.hpp"
namespace strata::core::portable_detail {
struct StrataVkNativeLlvmLog1p{std::unique_ptr<StrataVkLogitsGuard>owner;};
StrataVkNativeLlvmLog1p* strata_vk_native_llvm_log1p_create(ggml_backend_t b,const char*path,size_t budget){native_hc_need(b&&ggml_backend_is_vk(b),"LLVM log1p Vulkan required");auto*ctx=static_cast<ggml_backend_vk_context*>(b->context);auto fp=ctx->device->physical_device.getProperties2<vk::PhysicalDeviceProperties2,vk::PhysicalDeviceVulkan12Properties>().get<vk::PhysicalDeviceVulkan12Properties>();bool physical=ctx->device->physical_device.getFeatures().shaderFloat64;bool enabled=static_cast<const StrataVkQ8Device*>(ctx->device.get())->strata_float64_enabled;llvm_log1p_require(physical,enabled,fp);printf("native_llvm_log1p physical_float64=%d logical_float64=%d rte64=%d signed64=%d nonfma=true\n",int(physical),int(enabled),int(fp.shaderRoundingModeRTEFloat64),int(fp.shaderSignedZeroInfNanPreserveFloat64));auto code=llvm_log1p_modes(strata_native_llvm_log1p_spv,sizeof(strata_native_llvm_log1p_spv)/4);auto p=std::make_unique<StrataVkNativeLlvmLog1p>();p->owner=strata_vk_q8_component_owner(b,code.data(),code.size(),NativeComponentAddressing::descriptor,"strata_native_llvm_log1p",4,4,path,budget);return p.release();}
void strata_vk_native_llvm_log1p_destroy(StrataVkNativeLlvmLog1p*p)noexcept{delete p;}
void strata_vk_native_llvm_log1p_run(StrataVkNativeLlvmLog1p*p,ggml_tensor*const t[4]){native_hc_need(p,"GDN log1p: owner");auto&g=*p->owner;auto*ctx=static_cast<ggml_backend_vk_context*>(g.backend->context);std::array<uint32_t,4>off{};
 for(size_t i=0;i<4;++i){auto*x=t[i];native_hc_need(x&&x->type==(i<2?GGML_TYPE_F32:GGML_TYPE_I32)&&ggml_nbytes(x)==(i==2?2048:1024)&&ggml_is_contiguous(x)&&x->buffer&&ggml_backend_buffer_is_vk(x->buffer),"GDN log1p: fixed contiguous F32[256] Vulkan spans");auto*b=static_cast<ggml_backend_vk_buffer_context*>(x->buffer->context);native_hc_need(b->device.lock()==ctx->device,"GDN log1p: foreign device");auto bias=get_misalign_bytes(ctx,x);native_hc_need(bias%4==0,"GDN log1p: alignment");off[i]=bias/4;
  for(size_t j=0;j<i;++j)if(x->buffer==t[j]->buffer){auto a=reinterpret_cast<uintptr_t>(x->data),b=reinterpret_cast<uintptr_t>(t[j]->data);native_hc_need((a>=b?a-b>=ggml_nbytes(t[j]):b-a>=ggml_nbytes(x)),"GDN log1p: overlapping input/output spans");}}
 ggml_backend_vk_synchronize(g.backend);std::array<float,256>input;ggml_backend_tensor_get(t[0],input.data(),0,1024);for(float x:input)native_gdn_log1p_input(x);

 ggml_pipeline_request_descriptor_sets(ctx,g.pipeline,1);auto compute=ggml_vk_get_compute_ctx(ctx);ggml_vk_sync_buffers(ctx,compute);ggml_vk_dispatch_pipeline(ctx,compute,g.pipeline,{ggml_vk_tensor_subbuffer(ctx,t[0],true),ggml_vk_tensor_subbuffer(ctx,t[1],true),ggml_vk_tensor_subbuffer(ctx,t[2],true),ggml_vk_tensor_subbuffer(ctx,t[3],true)},off,{1,1,1});ggml_vk_sync_buffers(ctx,compute);compute.reset();ggml_backend_vk_synchronize(g.backend);
}
}
