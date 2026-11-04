#pragma once
#include "portable_native_head_vocab.hpp"
#include "ggml-alloc.h"
#include <fstream>
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
    std::vector<float> last_input_, last_output_;
    std::array<uint8_t, head_vocab_packed_bytes> last_packed_{};
    std::filesystem::path directory_;
    bool recorded_ = false, candidate_ready_ = false;
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
                         const std::string& directory, size_t budget) : directory_(directory) {
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
        candidate_ready_ = false;
        if (embedding.size() != width) throw std::runtime_error("native PLE key: embedding shape");
        // Check before touching the device input or dispatching either component.
        qsa_q8_input_domain(embedding, 80, "native PLE key");
        ggml_backend_tensor_set(input_, embedding.data(), 0, width * sizeof(float));
        auto* output = projection_->run(input_);
        last_input_ = embedding;
        last_packed_ = projection_->packed_for_candidate(input_, output);
        last_output_.resize(rows);
        ggml_backend_tensor_get(output, last_output_.data(), 0, rows * sizeof(float));
        candidate_ready_ = true;
        return output;
    }
    // The first graph-completed candidate is lifetime-scoped and not an FFN commit.
    void verify_and_record(const ggml_tensor* consumed, uint64_t position, int32_t token,
                           const std::array<int32_t,2>& previous, const std::array<uint32_t,16>& indices) {
        if (!candidate_ready_ || consumed != projection_->output_bank() || !consumed->buffer || !ggml_is_contiguous(consumed) ||
            ggml_nbytes(consumed) != rows * sizeof(float) || last_input_.size() != width || last_output_.size() != rows)
            throw std::runtime_error("native PLE key: actual consumer bank differs");
        std::vector<float> input(width), output(rows);
        ggml_backend_tensor_get(input_, input.data(), 0, width * sizeof(float));
        ggml_backend_tensor_get(consumed, output.data(), 0, rows * sizeof(float));
        if (std::memcmp(input.data(), last_input_.data(), width * sizeof(float)) ||
            std::memcmp(output.data(), last_output_.data(), rows * sizeof(float)) ||
            projection_->packed_for_candidate(input_, consumed) != last_packed_)
            throw std::runtime_error("native PLE key: graph changed candidate banks");
        if (recorded_) return;
        const auto path = directory_ / "first-candidate";
        if (!std::filesystem::create_directory(path)) throw std::runtime_error("native PLE key: candidate exists");
        auto save = [&](const char* name, const void* data, size_t size) {
            std::ofstream f(path / name, std::ios::binary);
            f.write(static_cast<const char*>(data), size); f.flush();
            if (!f) throw std::runtime_error("native PLE key: candidate write");
        };
        save("input.f32le", input.data(), width * sizeof(float));
        save("qx.bin", last_packed_.data(), last_packed_.size());
        save("key.f32le", output.data(), rows * sizeof(float));
        std::ofstream meta(path / "metadata.json");
        meta << "{\"position\":" << position << ",\"token\":" << token << ",\"previous\":[" << previous[0] << "," << previous[1] << "],\"rows\":[";
        for (size_t i = 0; i < indices.size(); ++i) meta << (i ? "," : "") << indices[i];
        meta << "],\"direct_producer_bank\":true,\"packed_consumer_bank\":true,\"graph_completed\":true,\"committed\":false,\"qualified\":false}\n";
        meta.flush();
        if (!meta) throw std::runtime_error("native PLE key: candidate metadata write");
        recorded_ = true;
    }
};
} // namespace strata::core::portable_detail
