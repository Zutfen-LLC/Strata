#pragma once
#include "vulkan_native_q8_layout.hpp"
#include "../core/strata_vk_native_stages.hpp"
namespace strata::worker::vulkan_detail {
// Metadata-only projection. No backend buffer/pipeline creation or weight reads.
// Cache construction consumes these exact contexts/tensors after admission.
struct NativeQ8Storage {
    struct Block {Context ctx{nullptr,ggml_free};Buffer buffer{nullptr,ggml_backend_buffer_free};
        ggml_tensor* tensor=nullptr;size_t bytes=0;};
    std::vector<std::unique_ptr<Block>> blocks_;
    Context ctx_{nullptr,ggml_free};
    ggml_tensor *i_,*x_,*g_,*u_,*h_,*q_,*o_;
    size_t per_block_=0,weights_=0,scratch_=0,bytes_=0;
};
inline NativeQ8Storage make_native_q8_storage(ggml_backend_t backend,size_t slots,size_t budget,size_t limit) {
    using namespace strata::core::portable_detail;
    constexpr size_t blob_bytes=5222400;
    const auto raw_weights=native_q8_weight_bytes(slots,budget);
    NativeQ8Storage s;
    auto buft=ggml_backend_get_default_buffer_type(backend);
    const size_t alignment=ggml_backend_buft_get_alignment(buft);
    const auto layout=expert_block_layout(slots,blob_bytes,limit);
    s.per_block_=layout.slots_per_block;
    // A whole blob already has native 256-byte alignment. Query real GGML
    // allocation sizes, including end padding, before creating any buffer.
    s.weights_=0;
    for(size_t first=0;first<slots;) {
        const size_t count=std::min(slots-first,s.per_block_);
        auto block=std::make_unique<NativeQ8Storage::Block>();
        block->ctx.reset(ggml_init({4096,nullptr,true}));require(bool(block->ctx),"native Q8 block metadata allocation failed");
        block->tensor=ggml_new_tensor_1d(block->ctx.get(),GGML_TYPE_I32,count*blob_bytes/4);
        block->bytes=ggml_backend_alloc_ctx_tensors_from_buft_size(block->ctx.get(),buft);
        require(alignment && !(alignment&(alignment-1)) && block->bytes>=count*blob_bytes && block->bytes<=limit && blob_bytes%alignment==0,
            "native Q8 backend block layout exceeded");
        s.weights_=expert_checked_add(s.weights_,block->bytes);
        s.blocks_.push_back(std::move(block));first+=count;
    }
    require(s.weights_>=raw_weights && s.weights_<=budget,"native Q8 aligned weight budget exceeded");
    s.ctx_.reset(ggml_init({64*1024,nullptr,true}));require(bool(s.ctx_),"native Q8 metadata allocation failed");
    s.i_=ggml_new_tensor_1d(s.ctx_.get(),GGML_TYPE_F32,8*H);
    s.x_=ggml_new_tensor_1d(s.ctx_.get(),GGML_TYPE_I32,8*(H/32)*9);
    s.g_=ggml_new_tensor_1d(s.ctx_.get(),GGML_TYPE_F32,q8_entry_capacity*F);
    s.u_=ggml_new_tensor_1d(s.ctx_.get(),GGML_TYPE_F32,q8_entry_capacity*F);
    s.h_=ggml_new_tensor_1d(s.ctx_.get(),GGML_TYPE_F32,q8_entry_capacity*F);
    s.q_=ggml_new_tensor_1d(s.ctx_.get(),GGML_TYPE_I32,q8_entry_capacity*(F/32)*9);
    s.o_=ggml_new_tensor_1d(s.ctx_.get(),GGML_TYPE_F32,q8_entry_capacity*H);
    const size_t tensor_scratch=ggml_backend_alloc_ctx_tensors_from_buft_size(s.ctx_.get(),buft);
    require(tensor_scratch<=limit,"native Q8 scratch exceeds backend block limit");
    auto metadata_size=[&](size_t words) {
        Context probe(ggml_init({4096,nullptr,true}),ggml_free);require(bool(probe),"native Q8 component probe failed");
        ggml_new_tensor_1d(probe.get(),GGML_TYPE_I32,words);
        return ggml_backend_alloc_ctx_tensors_from_buft_size(probe.get(),buft);
    };
    const size_t stage_bytes=metadata_size(STRATA_Q8_STAGE_WORDS),group_bytes=metadata_size(q8_metadata_words);
    require(stage_bytes<=limit && group_bytes<=limit,"native Q8 component exceeds backend block limit");
    const size_t projected_components=expert_checked_add(stage_bytes,expert_checked_add(group_bytes,group_bytes));
    s.scratch_=expert_checked_add(tensor_scratch,projected_components);
    s.bytes_=expert_checked_add(s.weights_,s.scratch_);
    require(s.bytes_<=budget,"native Q8 tensor/component budget exceeded");
    return s;
}
}
