#pragma once
#include "portable_native_head_vocab.hpp"
#include "ggml-alloc.h"
#include <memory>
#include <vector>

namespace strata::core::portable_detail {
// Diagnostic owner for the Q8_0 PLE key projection. The gathered host vector
// is copied into one device bank; the existing native Q8_1 producer and MMVQ
// consumer use that bank and return their actual device output. The caller
// owns the surrounding PLE graph and its transactional history.
class PortableNativePleKey {
    using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
    static constexpr size_t width = 2560, rows = 10240;
    Context ctx_{nullptr, ggml_free};
    Buffer input_buffer_{nullptr, ggml_backend_buffer_free};
    ggml_tensor* input_ = nullptr;
    std::unique_ptr<PortableNativeHeadVocab> projection_;
    size_t bytes_ = 0;

    static Context context() {
        Context ctx(ggml_init({16384, nullptr, true}), ggml_free);
        if (!ctx) throw std::runtime_error("native PLE key: metadata");
        return ctx;
    }
    static ggml_tensor* input(ggml_context* c) {
        auto* value = ggml_new_tensor_1d(c, GGML_TYPE_F32, width);
        ggml_set_input(value);
        return value;
    }
public:
    static size_t estimate(ggml_backend_t backend) {
        if (!backend) throw std::runtime_error("native PLE key: backend");
        auto c = context(); input(c.get());
        size_t bank = ggml_backend_alloc_ctx_tensors_from_buft_size(
            c.get(), ggml_backend_get_default_buffer_type(backend));
        auto projection = PortableNativeHeadVocab::bound(rows);
        if (bank > SIZE_MAX - projection) throw std::runtime_error("native PLE key: budget overflow");
        return bank + projection;
    }
    PortableNativePleKey(ggml_backend_t backend, ggml_tensor* weight, const void* role, size_t role_bytes,
                         const std::string& directory, size_t budget) {
        if (!backend || !weight || !role || weight->type != GGML_TYPE_Q8_0 || weight->ne[0] != width ||
            weight->ne[1] != rows || ggml_nbytes(weight) != rows * 80 * 34 ||
            role_bytes != ggml_nbytes(weight) || estimate(backend) > budget)
            throw std::runtime_error("native PLE key: unsupported role or admission");
        ctx_ = context(); input_ = input(ctx_.get());
        input_buffer_.reset(ggml_backend_alloc_ctx_tensors(ctx_.get(), backend));
        if (!input_buffer_) throw std::runtime_error("native PLE key: input allocation");
        projection_ = std::make_unique<PortableNativeHeadVocab>(backend, weight, input_, role, role_bytes,
                                                                rows, directory, budget);
        const size_t bank = ggml_backend_buffer_get_size(input_buffer_.get());
        if (projection_->bytes() > SIZE_MAX - bank) throw std::runtime_error("native PLE key: allocation overflow");
        bytes_ = bank + projection_->bytes();
        if (bytes_ > budget || bytes_ > estimate(backend)) throw std::runtime_error("native PLE key: allocation budget");
    }
    size_t bytes() const { return bytes_; }
    uint64_t calls() const { return projection_->calls(); }
    const ggml_tensor* input_bank() const { return input_; }
    ggml_tensor* output_bank() const { return projection_->output_bank(); }
    ggml_tensor* run(const std::vector<float>& embedding) {
        if (embedding.size() != width) throw std::runtime_error("native PLE key: embedding shape");
        // Check before touching the device input or dispatching either component.
        qsa_q8_input_domain(embedding, 80, "native PLE key");
        ggml_backend_tensor_set(input_, embedding.data(), 0, width * sizeof(float));
        return projection_->run(input_);
    }
};
} // namespace strata::core::portable_detail
