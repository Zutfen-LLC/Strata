// Included inside strata::core::portable_detail by the native stage wrapper.
static void strata_vk_quantize_rows(StrataVkNativeStages* s,const ggml_tensor* input,ggml_tensor* packed,size_t tokens,size_t width) {
    const size_t blocks=width/32, packed_bytes=blocks*36;
    logits_guard_need(width==2560 || width==6144,"native row quant: unsupported width");
    logits_guard_need(s && s->quant_rcp && s->owner && input && packed && tokens>=1 && tokens<=2,
                      "native PLE key quant: invalid call");
    logits_guard_need(input->type==GGML_TYPE_F32 && packed->type==GGML_TYPE_I32 &&
        ggml_nbytes(input)==tokens*width*sizeof(float) && ggml_nbytes(packed)==tokens*packed_bytes,
        "native PLE key quant: shape/type mismatch");
    auto& g=*s->owner; auto* ctx=static_cast<ggml_backend_vk_context*>(g.backend->context);
    const auto& limits=ctx->device->properties.limits;
    logits_guard_need(blocks<=limits.maxComputeWorkGroupCount[0] && tokens<=limits.maxComputeWorkGroupCount[1],
                      "native PLE key quant: grid exceeds device limit");
    std::array<uint32_t,9> params{0,uint32_t(width),uint32_t(width)};
    const ggml_tensor* tensors[]={input,input,g.status,g.status,input,packed};
    for(size_t i=0;i<6;++i) {
        const auto* t=tensors[i];
        logits_guard_need(t->buffer && ggml_is_contiguous(t) && ggml_backend_buffer_is_vk(t->buffer),
                          "native PLE key quant: contiguous Vulkan tensors required");
        auto* buffer=static_cast<ggml_backend_vk_buffer_context*>(t->buffer->context);
        logits_guard_need(buffer->device.lock()==ctx->device,"native PLE key quant: different device");
        const size_t offset=ggml_vk_tensor_buffer_offset(ctx,t), size=ggml_backend_buffer_get_size(t->buffer);
        logits_guard_need(offset<=size && ggml_nbytes(t)<=size-offset,"native PLE key quant: tensor outside buffer");
        const uint32_t bias=get_misalign_bytes(ctx,t);
        logits_guard_need(bias%4==0 && uint64_t(ggml_nbytes(t))+bias<=limits.maxStorageBufferRange,
                          "native PLE key quant: descriptor range or alignment");
        logits_guard_need(reinterpret_cast<uintptr_t>(t->data)<=UINTPTR_MAX-ggml_nbytes(t),
                          "native PLE key quant: span overflow");
        params[3+i]=bias/4;
    }
    if(input->buffer==packed->buffer) {
        const auto x=reinterpret_cast<uintptr_t>(input->data), y=reinterpret_cast<uintptr_t>(packed->data);
        logits_guard_need(x<=y ? y-x>=ggml_nbytes(input) : x-y>=ggml_nbytes(packed),
                          "native PLE key quant: overlapping input/output");
    }
    if(g.status->buffer==packed->buffer) {
        const auto x=reinterpret_cast<uintptr_t>(g.status->data), y=reinterpret_cast<uintptr_t>(packed->data);
        logits_guard_need(x<=y ? y-x>=ggml_nbytes(g.status) : x-y>=ggml_nbytes(packed),
                          "native PLE key quant: output overlaps metadata");
    }
    params[6]=params[5]+STRATA_Q8_STAGE_COUNT_WORD;
    ggml_backend_vk_synchronize(g.backend);
    ggml_pipeline_request_descriptor_sets(ctx,g.pipeline,1);
    auto compute=ggml_vk_get_compute_ctx(ctx); ggml_vk_sync_buffers(ctx,compute);
    ggml_vk_dispatch_pipeline(ctx,compute,g.pipeline,
        {ggml_vk_tensor_subbuffer(ctx,input,true),ggml_vk_tensor_subbuffer(ctx,input,true),
         ggml_vk_tensor_subbuffer(ctx,g.status,true),ggml_vk_tensor_subbuffer(ctx,g.status,true),
         ggml_vk_tensor_subbuffer(ctx,input,true),ggml_vk_tensor_subbuffer(ctx,packed,true)},
        params,{uint32_t(blocks),uint32_t(tokens),1});
    ggml_vk_sync_buffers(ctx,compute); compute.reset();
    try { ggml_backend_vk_synchronize(g.backend); }
    catch(...) { ++s->quant_submission_uncertain; throw; }
    ++s->quant_submitted; ++s->quant_completed;
}
void strata_vk_ple_key_quantize_bounded(StrataVkNativeStages* s,const ggml_tensor* input,ggml_tensor* packed,size_t tokens) {
    strata_vk_quantize_rows(s,input,packed,tokens,2560);
}
void strata_vk_gdn_input_quantize_bounded(StrataVkNativeStages* s,const ggml_tensor* input,ggml_tensor* packed,size_t tokens) {
    strata_vk_quantize_rows(s,input,packed,tokens,2560);
}
void strata_vk_gdn_output_quantize_bounded(StrataVkNativeStages* s,const ggml_tensor* input,ggml_tensor* packed,size_t tokens) {
    strata_vk_quantize_rows(s,input,packed,tokens,6144);
}
