#include "strata_vk_native_ple_gate.hpp"
#include "strata_native_ple_gate_t1_spv.hpp"
#include <limits>
namespace strata::core::portable_detail {
struct StrataVkNativePleGate {std::unique_ptr<StrataVkLogitsGuard> owner;size_t budget=0;uint64_t calls=0;};
StrataVkNativePleGate* strata_vk_native_ple_gate_create(ggml_backend_t b,const char* path,size_t budget) {
    native_hc_need(b&&ggml_backend_is_vk(b),"native PLE gate: Vulkan backend required");
    auto* ctx=static_cast<ggml_backend_vk_context*>(b->context);const auto& l=ctx->device->properties.limits;
    native_hc_need(l.maxComputeWorkGroupInvocations>=512&&l.maxComputeWorkGroupSize[0]>=512&&
                   l.maxComputeSharedMemorySize>=528*4&&l.maxComputeWorkGroupCount[0]>=4&&
                   l.maxPerStageDescriptorStorageBuffers>=5&&l.maxDescriptorSetStorageBuffers>=5,
                   "native PLE gate: workgroup/shared/descriptors unsupported");
    auto p=std::make_unique<StrataVkNativePleGate>();p->budget=budget;
    p->owner=strata_vk_q8_component_owner(b,strata_native_ple_gate_t1_spv,sizeof(strata_native_ple_gate_t1_spv)/4,
        NativeComponentAddressing::descriptor,"strata_native_ple_gate_t1_observation",5,5,path,budget,1);
    return p.release();
}
void strata_vk_native_ple_gate_destroy(StrataVkNativePleGate* p) noexcept {delete p;}
size_t strata_vk_native_ple_gate_bytes(const StrataVkNativePleGate* p){return p?p->owner->bytes:0;}
uint64_t strata_vk_native_ple_gate_calls(const StrataVkNativePleGate* p) noexcept{return p?p->calls:0;}
uint32_t strata_vk_native_ple_gate_run(StrataVkNativePleGate* p,const ggml_tensor* key,const ggml_tensor* query,
                                      ggml_tensor* gate,ggml_tensor* diagnostic) {
    native_hc_need(p,"native PLE gate: owner required");auto& g=*p->owner;
    auto* ctx=static_cast<ggml_backend_vk_context*>(g.backend->context);
    const ggml_tensor* tensors[]={key,query,gate,diagnostic};const size_t bytes[]={40960,40960,16,64};
    std::array<uint32_t,5> off{};size_t used=g.bytes;ggml_backend_buffer_t seen[4]{};size_t nseen=0;
    for(size_t i=0;i<4;++i){const auto* t=tensors[i];
        native_hc_need(t&&t->type==GGML_TYPE_F32&&ggml_nbytes(t)==bytes[i]&&ggml_is_contiguous(t)&&t->buffer&&ggml_backend_buffer_is_vk(t->buffer),"native PLE gate: exact contiguous Vulkan F32 span required");
        auto* b=static_cast<ggml_backend_vk_buffer_context*>(t->buffer->context);native_hc_need(b->device.lock()==ctx->device,"native PLE gate: foreign device");
        auto bias=get_misalign_bytes(ctx,t);native_hc_need(bias%4==0,"native PLE gate: alignment");off[i]=bias/4;
        bool duplicate=false;for(size_t j=0;j<nseen;++j)duplicate|=seen[j]==t->buffer;
        if(!duplicate){seen[nseen++]=t->buffer;const size_t n=ggml_backend_buffer_get_size(t->buffer);native_hc_need(n<=std::numeric_limits<size_t>::max()-used,"native PLE gate: budget overflow");used+=n;}
        for(size_t j=0;j<i;++j)if((i>=2||j>=2)&&t->buffer==tensors[j]->buffer){auto x=reinterpret_cast<uintptr_t>(t->data),y=reinterpret_cast<uintptr_t>(tensors[j]->data);native_hc_need(x<=y?y-x>=ggml_nbytes(t):x-y>=ggml_nbytes(tensors[j]),"native PLE gate: output overlap");}
    }
    native_hc_need(used<=p->budget,"native PLE gate: allocation budget exceeded");
    ggml_backend_vk_synchronize(g.backend);std::vector<float> input(10240);
    for(const auto* t:{key,query}){ggml_backend_tensor_get(t,input.data(),0,40960);for(float x:input)native_hc_normal(x);}
    off[4]=get_misalign_bytes(ctx,g.status)/4;const uint32_t zero=0;ggml_backend_tensor_set(g.status,&zero,0,4);
    ggml_pipeline_request_descriptor_sets(ctx,g.pipeline,1);auto compute=ggml_vk_get_compute_ctx(ctx);ggml_vk_sync_buffers(ctx,compute);
    ggml_vk_dispatch_pipeline(ctx,compute,g.pipeline,{ggml_vk_tensor_subbuffer(ctx,key,true),ggml_vk_tensor_subbuffer(ctx,query,true),
        ggml_vk_tensor_subbuffer(ctx,gate,true),ggml_vk_tensor_subbuffer(ctx,diagnostic,true),ggml_vk_tensor_subbuffer(ctx,g.status,true)},off,{4,1,1});
    ggml_vk_sync_buffers(ctx,compute);compute.reset();ggml_backend_vk_synchronize(g.backend);
    uint32_t status=0;ggml_backend_tensor_get(g.status,&status,0,4);if(!status)++p->calls;return status;
}
}
