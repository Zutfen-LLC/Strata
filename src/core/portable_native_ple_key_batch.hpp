#pragma once
#include "portable_native_head_vocab.hpp"
#include "portable_native_ple_host_bound.hpp"
#include <cstring>

namespace strata::core::portable_detail {
// Standalone bounded projection. The resident weight is borrowed, never copied.
class PortableNativePleKeyBatch {
    using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
    static constexpr size_t width = 2560, rows = 10240, packed_bytes = 2880;
    Context ctx_{nullptr, ggml_free};
    Buffer buffer_{nullptr, ggml_backend_buffer_free};
    std::unique_ptr<StrataVkNativeStages, decltype(&strata_vk_native_stages_destroy)> quant_{nullptr, strata_vk_native_stages_destroy};
    std::unique_ptr<StrataVkQ8Wo, decltype(&strata_vk_q8_wo_destroy)> dot_{nullptr, strata_vk_q8_wo_destroy};
    ggml_tensor *input_[2]{}, *packed_[2]{}, *output_[2]{}, *weight_ = nullptr;
    size_t bytes_ = 0;
    uint64_t calls_ = 0;
    bool valid_ = false;
    static Context context() {
        Context c(ggml_init({16384, nullptr, true}), ggml_free);
        if (!c) throw std::runtime_error("native PLE key batch: metadata");
        return c;
    }
public:
    // Persistent 16 KiB ggml context; peak of constructor's 64 KiB role
    // comparison and run's packed rows. Weight and input are borrowed.
    static constexpr size_t host_bound(size_t tokens) {
        const size_t packed = ple_host_mul(tokens, packed_bytes);
        return ple_host_add(16384, packed > 65536 ? packed : 65536);
    }
    static size_t estimate(ggml_backend_t backend) {
        if (!backend) throw std::runtime_error("native PLE key batch: backend");
        auto c = context();
        ggml_new_tensor_1d(c.get(), GGML_TYPE_F32, 2 * width);
        ggml_new_tensor_1d(c.get(), GGML_TYPE_I32, 2 * packed_bytes / 4);
        ggml_new_tensor_1d(c.get(), GGML_TYPE_F32, 2 * rows);
        const size_t bank = ggml_backend_alloc_ctx_tensors_from_buft_size(c.get(), ggml_backend_get_default_buffer_type(backend));
        if (bank > SIZE_MAX - (3u << 20)) throw std::runtime_error("native PLE key batch: budget overflow");
        return bank + (3u << 20);
    }
    PortableNativePleKeyBatch(ggml_backend_t backend, ggml_tensor* weight, const void* role, size_t role_bytes,
                              const std::string& directory, size_t budget) {
        if (!backend || !weight || !role || weight->type != GGML_TYPE_Q8_0 || weight->ne[0] != width ||
            weight->ne[1] != rows || !weight->buffer || !ggml_is_contiguous(weight) ||
            ggml_nbytes(weight) != rows * head_vocab_row_bytes || role_bytes != ggml_nbytes(weight) || estimate(backend) > budget)
            throw std::runtime_error("native PLE key batch: role or budget");
        qsa_wo_require_scales(qsa_wo_scale_classes(static_cast<const uint8_t*>(role), rows * 80, 34), "PLE key weight", true);
        std::vector<uint8_t> check(65536);
        for (size_t at = 0; at < role_bytes; at += check.size()) {
            const size_t n = std::min(check.size(), role_bytes - at);
            ggml_backend_tensor_get(weight, check.data(), at, n);
            if (std::memcmp(check.data(), static_cast<const uint8_t*>(role) + at, n))
                throw std::runtime_error("native PLE key batch: resident role differs");
        }
        if (!std::filesystem::create_directory(directory)) throw std::runtime_error("native PLE key batch: fresh receipt required");
        ctx_ = context();
        input_[1] = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_F32, 2 * width);
        packed_[1] = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 2 * packed_bytes / 4);
        output_[1] = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_F32, 2 * rows);
        input_[0] = ggml_view_1d(ctx_.get(), input_[1], width, 0);
        packed_[0] = ggml_view_1d(ctx_.get(), packed_[1], packed_bytes / 4, 0);
        output_[0] = ggml_view_1d(ctx_.get(), output_[1], rows, 0);
        weight_ = ggml_view_1d(ctx_.get(), weight, width * rows, 0);
        weight_->type = GGML_TYPE_I32; weight_->ne[0] = role_bytes / 4;
        weight_->nb[0] = 4; weight_->nb[1] = weight_->nb[2] = weight_->nb[3] = role_bytes;
        if (ggml_backend_view_init(weight_) != GGML_STATUS_SUCCESS) throw std::runtime_error("native PLE key batch: weight view");
        const size_t bank = ggml_backend_alloc_ctx_tensors_from_buft_size(ctx_.get(), ggml_backend_get_default_buffer_type(backend));
        if (bank > budget) throw std::runtime_error("native PLE key batch: bank budget");
        quant_.reset(strata_vk_native_stages_create_quant_rcp_observation(backend, (std::filesystem::path(directory) / "quant-effective.spv").string().c_str(), budget - bank));
        const size_t qb = strata_vk_native_stages_bytes(quant_.get());
        const size_t borrowed = ggml_backend_buffer_get_size(weight->buffer);
        if (qb > budget - bank || borrowed > SIZE_MAX - (budget - bank - qb)) throw std::runtime_error("native PLE key batch: component budget");
        dot_.reset(strata_vk_head_vocab_create(backend, (std::filesystem::path(directory) / "dot-effective.spv").string().c_str(), budget - bank - qb + borrowed));
        buffer_.reset(ggml_backend_alloc_ctx_tensors(ctx_.get(), backend));
        if (!buffer_) throw std::runtime_error("native PLE key batch: allocation");
        bytes_ = ggml_backend_buffer_get_size(buffer_.get()) + qb + strata_vk_q8_wo_bytes(dot_.get());
        if (bytes_ > budget || bytes_ > estimate(backend)) throw std::runtime_error("native PLE key batch: actual budget");
    }
    size_t bytes() const { return bytes_; }
    uint64_t calls() const { return calls_; }
    ggml_tensor* output_bank(size_t tokens) const {
        if (tokens < 1 || tokens > 2) throw std::runtime_error("native PLE key batch: tokens");
        return output_[tokens - 1];
    }
    ggml_tensor* run(const std::vector<float>& input, size_t tokens) {
        valid_ = false;
        if (tokens < 1 || tokens > 2 || input.size() != tokens * width) throw std::runtime_error("native PLE key batch: shape");
        qsa_q8_input_domain(input, tokens * 80, "native PLE key batch");
        ggml_backend_tensor_set(input_[tokens - 1], input.data(), 0, input.size() * sizeof(float));
        strata_vk_ple_key_quantize_bounded(quant_.get(), input_[tokens - 1], packed_[tokens - 1], tokens);
        std::vector<uint8_t> packed(tokens * packed_bytes);
        ggml_backend_tensor_get(packed_[tokens - 1], packed.data(), 0, packed.size());
        qsa_wo_require_scales(qsa_wo_scale_classes(packed.data(), tokens * 80, 36), "PLE key producer", true);
        strata_vk_ple_key_run_bounded(dot_.get(), weight_, packed_[tokens - 1], output_[tokens - 1], rows, tokens);
        ++calls_; valid_ = true;
        return output_[tokens - 1];
    }
    bool valid() const { return valid_; }
};
}
