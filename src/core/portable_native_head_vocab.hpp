#pragma once
#include "ggml-alloc.h"
#include "strata/core/portable_ffn.hpp"
#include "native_head_vocab_layout.hpp"
#include "native_qsa_wo_domain.hpp"
#include "strata_vk_native_stages.hpp"
#include "strata_vk_q8_wo.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <vector>

namespace strata::core::portable_detail {
// The mixer, resident matrix and final logits remain on their owning device.
class PortableNativeHeadVocab {
    using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
    Context ctx_{nullptr, ggml_free};
    Buffer buffer_{nullptr, ggml_backend_buffer_free};
    std::unique_ptr<StrataVkNativeStages, decltype(&strata_vk_native_stages_destroy)> quant_{
        nullptr, strata_vk_native_stages_destroy};
    std::unique_ptr<StrataVkQ8Wo, decltype(&strata_vk_q8_wo_destroy)> dot_{nullptr, strata_vk_q8_wo_destroy};
    ggml_tensor *packed_ = nullptr, *logits_ = nullptr, *mixed_ = nullptr;
    std::vector<ggml_tensor *> weights_, outputs_;
    std::vector<size_t> rows_;
    size_t bytes_ = 0;
    uint64_t completed_ = 0;
    std::filesystem::path receipts_;
    bool valid_ = false, recorded_ = false;
    std::array<uint8_t, head_vocab_packed_bytes> consumed_packed_{};
    static Context context() {
        Context c(ggml_init({256u << 10, nullptr, true}), ggml_free);
        if (!c)
            throw std::runtime_error("native head vocabulary: metadata");
        return c;
    }

public:
    // Includes private component resources plus padding for the output and views.
    static size_t bound(size_t rows) {
        if (!rows || rows > 1048576)
            throw std::runtime_error("native head vocabulary: rows");
        return (3u << 20) + rows * sizeof(float);
    }
    PortableNativeHeadVocab(ggml_backend_t backend, ggml_tensor *weight, ggml_tensor *mixed, const void *role,
                            size_t role_size, size_t rows, const std::string &directory, size_t budget) {
        if (!backend || !weight || !mixed || !role || weight->type != GGML_TYPE_Q8_0 ||
            weight->ne[0] != 2560 || size_t(weight->ne[1]) != rows ||
            ggml_nbytes(weight) != rows * head_vocab_row_bytes || role_size != ggml_nbytes(weight) ||
            !weight->buffer || !ggml_is_contiguous(weight) || !mixed->buffer || !ggml_is_contiguous(mixed) ||
            ggml_nbytes(mixed) != 2560 * sizeof(float) || bound(rows) > budget)
            throw std::runtime_error("native head vocabulary: role, producer or budget");
        qsa_wo_require_scales(qsa_wo_scale_classes(static_cast<const uint8_t *>(role), rows * 80, 34),
                              "head vocabulary weight", true);
        std::vector<uint8_t> check(65536);
        const auto *expected = static_cast<const uint8_t *>(role);
        for (size_t offset = 0; offset < role_size; offset += check.size()) {
            size_t n = std::min(check.size(), role_size - offset);
            ggml_backend_tensor_get(weight, check.data(), offset, n);
            if (std::memcmp(check.data(), expected + offset, n))
                throw std::runtime_error("native head vocabulary: resident role differs");
        }
        if (!std::filesystem::create_directory(directory))
            throw std::runtime_error("native head vocabulary: fresh receipt required");
        receipts_ = directory;
        mixed_ = mixed;
        ctx_ = context();
        packed_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, head_vocab_packed_bytes / 4);
        logits_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_F32, rows);
        ggml_set_output(logits_);
        for (size_t start = 0; start < rows; start += head_vocab_slice_rows) {
            size_t n = std::min(head_vocab_slice_rows, rows - start);
            auto *w = ggml_view_1d(ctx_.get(), weight, 2560 * n, start * head_vocab_row_bytes);
            w->type = GGML_TYPE_I32;
            w->ne[0] = n * head_vocab_row_bytes / 4;
            w->nb[0] = 4;
            w->nb[1] = w->nb[2] = w->nb[3] = n * head_vocab_row_bytes;
            if (ggml_backend_view_init(w) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("native head vocabulary: weight view");
            weights_.push_back(w);
            outputs_.push_back(ggml_view_1d(ctx_.get(), logits_, n, start * sizeof(float)));
            rows_.push_back(n);
        }
        const auto bank = ggml_backend_alloc_ctx_tensors_from_buft_size(
            ctx_.get(), ggml_backend_get_default_buffer_type(backend));
        if (bank >= budget)
            throw std::runtime_error("native head vocabulary: bank budget");
        quant_.reset(strata_vk_native_stages_create_quant_rcp_observation(
            backend, (std::filesystem::path(directory) / "quant-effective.spv").string().c_str(),
            budget - bank));
        const size_t qb = strata_vk_native_stages_bytes(quant_.get());
        const auto borrowed = ggml_backend_buffer_get_size(weight->buffer);
        if (qb > budget - bank || borrowed > SIZE_MAX - (budget - qb))
            throw std::runtime_error("native head vocabulary: owner budget");
        dot_.reset(strata_vk_head_vocab_create(
            backend, (std::filesystem::path(directory) / "dot-effective.spv").string().c_str(),
            budget - qb + borrowed));
        buffer_.reset(ggml_backend_alloc_ctx_tensors(ctx_.get(), backend));
        if (!buffer_)
            throw std::runtime_error("native head vocabulary: bank allocation");
        bytes_ = ggml_backend_buffer_get_size(buffer_.get()) + qb + strata_vk_q8_wo_bytes(dot_.get());
        if (bytes_ > bound(rows) || bytes_ > budget)
            throw std::runtime_error("native head vocabulary: actual budget");
    }
    size_t bytes() const { return bytes_; }
    uint64_t calls() const { return completed_; }
    ggml_tensor *run(ggml_tensor *mixed) {
        valid_ = false;
        if (mixed != mixed_)
            throw std::runtime_error("native head vocabulary: actual mixer bank required");
        std::vector<float> input(2560);
        ggml_backend_tensor_get(mixed, input.data(), 0, input.size() * sizeof(float));
        qsa_q8_input_domain(input, 80, "native head vocabulary");
        const int32_t starts[] = {0, 1};
        strata_vk_native_stages_run(quant_.get(), 0, mixed, mixed, mixed, packed_, 2560, starts, 2);
        std::array<uint8_t, head_vocab_packed_bytes> packed{};
        ggml_backend_tensor_get(packed_, packed.data(), 0, packed.size());
        qsa_wo_require_scales(qsa_wo_scale_classes(packed.data(), 80, 36), "head vocabulary producer Q8_1", true);
        for (size_t i = 0; i < rows_.size(); ++i)
            strata_vk_q8_wo_run(dot_.get(), weights_[i], packed_, outputs_[i], rows_[i]);
        if (!recorded_) {
            ggml_backend_tensor_get(packed_, consumed_packed_.data(), 0, consumed_packed_.size());
            if (packed != consumed_packed_)
                throw std::runtime_error("native head vocabulary: consumed packed bank changed");
        }
        ++completed_;
        valid_ = true;
        return logits_;
    }
    // Called only after the caller has validated the full projected logits.
    // The mixed and output identities tie this receipt to the actual dispatch.
    bool record_first(const PortableFinalHcResidual &producer, const ggml_tensor *mixed,
                      const ggml_tensor *logits) {
        if (!valid_ || mixed != mixed_ || logits != logits_ || !producer.tensor ||
            producer.position == UINT64_MAX || producer.owner_id == UINT64_MAX ||
            producer.write_sequence == 0 || producer.layer != 47 || !producer.owner ||
            !producer.is_current || !producer.is_current(producer.owner, producer))
            throw std::runtime_error("native head vocabulary: current producer and output required");
        if (recorded_)
            return false;
        std::array<uint8_t, head_vocab_packed_bytes> bytes{};
        ggml_backend_tensor_get(packed_, bytes.data(), 0, bytes.size());
        if (bytes != consumed_packed_)
            throw std::runtime_error("native head vocabulary: current packed bank changed");
        const auto packed_path = receipts_ / "first-q8_1.bin";
        const auto metadata_path = receipts_ / "first.json";
        bool packed_created = false, metadata_created = false;
        try {
            auto *f = std::fopen(packed_path.string().c_str(), "wbx");
            if (!f) throw std::runtime_error("native head vocabulary: fresh packed receipt required");
            packed_created = true;
            const bool written = std::fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
            const bool closed = std::fclose(f) == 0;
            if (!written || !closed) throw std::runtime_error("native head vocabulary: packed receipt write failed");
            f = std::fopen(metadata_path.string().c_str(), "wx");
            if (!f) throw std::runtime_error("native head vocabulary: fresh metadata required");
            metadata_created = true;
            const bool metadata_written = std::fprintf(f,
                "{\"position\":%llu,\"producer_owner_id\":%llu,\"producer_write_sequence\":%llu,"
                "\"producer_layer\":47,\"mixed_tensor_identity\":true,\"packed_consumer_identity\":true,"
                "\"logits_tensor_identity\":true,\"packed_bytes\":%zu,\"slice_count\":%zu,\"qualified\":false}\n",
                (unsigned long long)producer.position, (unsigned long long)producer.owner_id,
                (unsigned long long)producer.write_sequence, bytes.size(), rows_.size()) > 0;
            const bool metadata_closed = std::fclose(f) == 0;
            if (!metadata_written || !metadata_closed) throw std::runtime_error("native head vocabulary: metadata write failed");
            recorded_ = true;
            return true;
        } catch (...) {
            std::error_code ignored;
            if (metadata_created) std::filesystem::remove(metadata_path, ignored);
            if (packed_created) std::filesystem::remove(packed_path, ignored);
            throw;
        }
    }
    // Only remove files created by the immediately preceding record_first call.
    void rollback_first() noexcept {
        if (!recorded_) return;
        std::error_code ignored;
        std::filesystem::remove(receipts_ / "first.json", ignored);
        std::filesystem::remove(receipts_ / "first-q8_1.bin", ignored);
        recorded_ = false;
    }
};
} // namespace strata::core::portable_detail
