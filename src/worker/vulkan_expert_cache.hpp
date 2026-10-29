#pragma once
#include "vulkan_expert_graph.hpp"
#include <algorithm>

namespace strata::worker::vulkan_detail {
inline size_t expert_checked_add(size_t a, size_t b) {
    require(b <= SIZE_MAX-a,"expert allocation size overflow"); return a+b;
}
inline size_t expert_aligned_size(size_t bytes, size_t alignment) {
    require(alignment && !(alignment & (alignment-1)),"invalid expert backend alignment");
    return expert_checked_add(bytes,alignment-1) & ~(alignment-1);
}
struct ExpertBlockLayout {
    size_t slot_bytes, slots_per_block, bytes, blocks;
};
inline ExpertBlockLayout expert_block_layout(size_t slots, size_t slot_bytes, size_t max_bytes) {
    require(slots && slot_bytes && slot_bytes <= max_bytes,"expert slot exceeds backend block limit");
    require(slots <= SIZE_MAX/slot_bytes,"expert allocation size overflow");
    const size_t per_block=max_bytes/slot_bytes;
    return {slot_bytes,per_block,slots*slot_bytes,1+(slots-1)/per_block};
}

// Native ExpertCache::open_sized owns one logical packed slot arena. GGML limits
// individual backend buffers, so each block holds whole slots in frozen plan order.
// Only weight descriptors scale with slots; execution metadata and scratch do not.
class ExpertCache {
    struct Block {
        Context ctx{nullptr,ggml_free};
        Buffer buffer{nullptr,ggml_backend_buffer_free};
        size_t bytes=0;
    };
    std::vector<std::unique_ptr<Block>> blocks_;
    std::vector<ExpertWeights> slots_;
    std::vector<bool> uploaded_;
    std::unique_ptr<ExpertGraph> execution_;
    ggml_backend_t backend_;
    size_t weight_bytes_=0;
    bool poisoned_=false;
public:
    ExpertCache(ggml_backend_t backend, const cpu::NativeFmt& fmt, size_t slots, size_t budget):backend_(backend) {
        auto buft=ggml_backend_get_default_buffer_type(backend);
        // Query the actual backend projection sizes, then let the existing GGML
        // helper pack and allocate each block. No independent device allocator.
        Context probe(ggml_init({4096,nullptr,true}),ggml_free);
        require(bool(probe),"expert layout metadata allocation failed");
        auto* gu=ggml_new_tensor_2d(probe.get(),GGML_TYPE_Q8_0,H,F);
        auto* down=ggml_new_tensor_2d(probe.get(),GGML_TYPE_Q8_0,F,H);
        const size_t alignment=ggml_backend_buft_get_alignment(buft);
        const size_t gu_bytes=expert_aligned_size(ggml_backend_buft_get_alloc_size(buft,gu),alignment);
        const size_t down_bytes=expert_aligned_size(ggml_backend_buft_get_alloc_size(buft,down),alignment);
        const size_t stride=expert_checked_add(expert_checked_add(gu_bytes,gu_bytes),down_bytes);
        // The fixed native layout is 256 aligned already. Reject an incompatible
        // backend layout rather than silently changing slot alignment.
        require(stride%256 == 0,"expert backend slot is not native aligned");
        const auto layout=expert_block_layout(slots,stride,ggml_backend_buft_get_max_size(buft));
        require(layout.bytes <= budget,"Vulkan resident tensor budget exceeded");
        slots_.reserve(slots); uploaded_.resize(slots,false);
        for(size_t first=0;first<slots;) {
            const size_t count=std::min(slots-first,layout.slots_per_block);
            require(count <= (SIZE_MAX-1024)/(3*ggml_tensor_overhead()),"expert metadata size overflow");
            auto block=std::make_unique<Block>();
            block->ctx.reset(ggml_init({1024+count*3*ggml_tensor_overhead(),nullptr,true}));
            require(bool(block->ctx),"expert weight metadata allocation failed");
            for(size_t i=0;i<count;++i) slots_.push_back({
                ggml_new_tensor_2d(block->ctx.get(),GGML_TYPE_Q8_0,H,F),
                ggml_new_tensor_2d(block->ctx.get(),GGML_TYPE_Q8_0,H,F),
                ggml_new_tensor_2d(block->ctx.get(),GGML_TYPE_Q8_0,F,H)});
            block->bytes=ggml_backend_alloc_ctx_tensors_from_buft_size(block->ctx.get(),buft);
            require(block->bytes == count*stride,"expert backend block layout changed");
            blocks_.push_back(std::move(block)); first+=count;
        }
        execution_=std::make_unique<ExpertGraph>(backend,fmt,std::vector<uint8_t>{},true,budget,&slots_[0],false);
        const size_t scratch=execution_->required_bytes();
        require(expert_checked_add(layout.bytes,scratch) <= budget,"Vulkan resident tensor budget exceeded");
        // All size/budget checks finish before the first backend allocation.
        for(auto& block:blocks_) {
            block->buffer.reset(ggml_backend_alloc_ctx_tensors(block->ctx.get(),backend));
            require(bool(block->buffer),"backend expert cache allocation failed");
            require(ggml_backend_buffer_get_size(block->buffer.get()) == block->bytes,"expert block allocation size mismatch");
            ggml_backend_buffer_set_usage(block->buffer.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            weight_bytes_=expert_checked_add(weight_bytes_,block->bytes);
        }
        execution_->allocate_buffer(scratch);
    }
    size_t size() const { return slots_.size(); }
    size_t weight_storage_bytes() const { return weight_bytes_; }
    size_t scratch_bytes() const { return execution_->buffer_bytes(); }
    size_t buffer_bytes() const { return weight_storage_bytes()+scratch_bytes(); }
    size_t allocation_count() const { return blocks_.size()+1; }
    void upload(size_t slot, const cpu::NativeFmt& fmt, const std::vector<uint8_t>& blob) {
        require(!poisoned_ && slot < slots_.size() && !uploaded_[slot] && blob.size() == fmt.bytes &&
                fmt.bytes == 5222400 && fmt.up_off == 1740800 && fmt.down_off == 3481600,
                "invalid expert cache upload");
        try {
            const auto& w=slots_[slot];
            ggml_backend_tensor_set(w.gate,blob.data(),0,fmt.up_off);
            ggml_backend_tensor_set(w.up,blob.data()+fmt.up_off,0,fmt.down_off-fmt.up_off);
            ggml_backend_tensor_set(w.down,blob.data()+fmt.down_off,0,fmt.bytes-fmt.down_off);
            uploaded_[slot]=true;
        } catch(...) { poisoned_=true; ggml_backend_synchronize(backend_); throw; }
    }
    Rows run(size_t slot, const std::vector<float>& input) {
        require(!poisoned_ && slot < slots_.size() && uploaded_[slot],"expert cache unavailable or invalid slot");
        try { execution_->select(slots_[slot]); return execution_->run(input); }
        // Execution scratch has no recurrent state. The model owns poisoning
        // and explicit reset/replay; preserve the existing executor retry contract.
        catch(...) { ggml_backend_synchronize(backend_); throw; }
    }
};
} // namespace strata::worker::vulkan_detail
