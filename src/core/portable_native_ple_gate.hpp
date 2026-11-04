#pragma once
#include "strata_vk_native_ple_gate.hpp"
#include "ggml-alloc.h"
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace strata::core::portable_detail {
// The prefix graph writes normalized PLE key/query into these device-owned
// banks. The existing PLE value-broadcast graph reads the actual gate bank.
class PortableNativePleGate {
    using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
    Context ctx_{nullptr, ggml_free};
    Buffer buffer_{nullptr, ggml_backend_buffer_free};
    std::unique_ptr<StrataVkNativePleGate, decltype(&strata_vk_native_ple_gate_destroy)>
        component_{nullptr, strata_vk_native_ple_gate_destroy};
    ggml_tensor *key_ = nullptr, *query_ = nullptr, *gate_ = nullptr, *diagnostic_ = nullptr;
    ggml_backend_t backend_ = nullptr;
    std::filesystem::path directory_;
    size_t bytes_ = 0;
    bool candidate_ = false, recorded_ = false;
    const ggml_tensor *key_copy_ = nullptr, *query_copy_ = nullptr, *raw_key_ = nullptr, *residual_ = nullptr;
    std::array<uint8_t, 40960> last_key_{}, last_query_{};
    std::array<uint8_t, 16> last_gate_{};

    static void need(bool condition, const char* message) {
        if (!condition) throw std::runtime_error(message);
    }
    static Context context() {
        Context c(ggml_init({16384, nullptr, true}), ggml_free);
        need(bool(c), "PLE gate: metadata");
        return c;
    }
    static void leaves(ggml_context* c, ggml_tensor*& key, ggml_tensor*& query,
                       ggml_tensor*& gate, ggml_tensor*& diagnostic) {
        key = ggml_new_tensor_2d(c, GGML_TYPE_F32, 2560, 4);
        query = ggml_new_tensor_2d(c, GGML_TYPE_F32, 2560, 4);
        gate = ggml_new_tensor_2d(c, GGML_TYPE_F32, 1, 4);
        diagnostic = ggml_new_tensor_2d(c, GGML_TYPE_F32, 4, 4);
        for (auto* tensor : {key, query, gate, diagnostic}) ggml_set_input(tensor);
    }
    // GGML may rewrite graph edges to backend-copy tensors during scheduling.
    // Establish the logical producer chain before handing the graph to it.
    static void require_logical_source(const ggml_tensor* copy, const ggml_tensor* bank,
                                       const ggml_tensor* original) {
        need(copy && copy->op == GGML_OP_CPY && copy->src[1] == bank &&
             copy->view_src == bank && copy->src[0], "PLE gate: prefix destination differs");
        const auto* norm = copy->src[0];
        need(norm->op == GGML_OP_MUL && norm->src[0] && norm->src[0]->op == GGML_OP_RMS_NORM &&
             norm->src[0]->src[0] && norm->src[0]->src[0]->src[0] == original,
             "PLE gate: normalized producer differs");
    }
    void require_scheduled_source(const ggml_tensor* copy, const ggml_tensor* bank) const {
        need(copy && copy->op == GGML_OP_CPY && copy->src[1] == bank &&
             copy->view_src == bank && copy->src[0], "PLE gate: prefix destination differs");
        // The copy source can be the scheduler's backend copy, but the actual
        // source and the persistent destination must both be on this device.
        for (const ggml_tensor* tensor : std::array<const ggml_tensor*, 2>{copy->src[0], bank}) {
            need(tensor->type == GGML_TYPE_F32 && tensor->buffer && ggml_is_contiguous(tensor) && ggml_nbytes(tensor) == 40960,
                 "PLE gate: normalized device span");
            auto type = ggml_backend_buffer_get_type(tensor->buffer);
            auto owner = ggml_backend_get_device(backend_);
            auto device = ggml_backend_buft_get_device(type);
            need(type == ggml_backend_get_default_buffer_type(backend_) && owner &&
                 (device ? device == owner : (ggml_backend_buft_is_host(type) &&
                                             ggml_backend_dev_type(owner) == GGML_BACKEND_DEVICE_TYPE_CPU)),
                 "PLE gate: normalized producer device differs");
        }
    }
  public:
    static size_t estimate(ggml_backend_t backend) {
        need(backend, "PLE gate: backend");
        auto c = context();
        ggml_tensor *key, *query, *gate, *diagnostic;
        leaves(c.get(), key, query, gate, diagnostic);
        auto bank = ggml_backend_alloc_ctx_tensors_from_buft_size(c.get(), ggml_backend_get_default_buffer_type(backend));
        auto metadata = context();
        ggml_new_tensor_1d(metadata.get(), GGML_TYPE_I32, 1);
        auto status = ggml_backend_alloc_ctx_tensors_from_buft_size(metadata.get(), ggml_backend_get_default_buffer_type(backend));
        need(bank <= SIZE_MAX - status, "PLE gate: budget overflow");
        return bank + status;
    }
    PortableNativePleGate(ggml_backend_t backend, const std::string& directory, size_t budget)
        : backend_(backend), directory_(directory) {
        need(backend && !directory.empty() && estimate(backend) <= budget && std::filesystem::create_directory(directory),
             "PLE gate: backend, budget or fresh receipt");
        ctx_ = context();
        leaves(ctx_.get(), key_, query_, gate_, diagnostic_);
        component_.reset(strata_vk_native_ple_gate_create(backend, (directory_ / "effective.spv").c_str(), budget));
        need(bool(component_), "PLE gate: component");
        buffer_.reset(ggml_backend_alloc_ctx_tensors(ctx_.get(), backend));
        need(bool(buffer_), "PLE gate: bank allocation");
        const auto bank = ggml_backend_buffer_get_size(buffer_.get());
        const auto extra = strata_vk_native_ple_gate_bytes(component_.get());
        need(bank <= SIZE_MAX - extra, "PLE gate: budget overflow");
        bytes_ = bank + extra;
        need(bytes_ <= budget && bytes_ <= estimate(backend), "PLE gate: aligned budget");
    }
    size_t bytes() const { return bytes_; }
    uint64_t calls() const { return strata_vk_native_ple_gate_calls(component_.get()); }
    ggml_tensor* key() const { return key_; }
    ggml_tensor* query() const { return query_; }
    ggml_tensor* gate() const { return gate_; }
    void bind(const ggml_tensor* key_copy, const ggml_tensor* query_copy,
              const ggml_tensor* raw_key, const ggml_tensor* residual) {
        need(key_copy && query_copy && key_copy != query_copy && raw_key && residual,
             "PLE gate: missing logical producers");
        require_logical_source(key_copy, key_, raw_key);
        require_logical_source(query_copy, query_, residual);
        key_copy_ = key_copy; query_copy_ = query_copy; raw_key_ = raw_key; residual_ = residual;
    }
    void begin() { candidate_ = false; }
    void run(const ggml_tensor* key_copy, const ggml_tensor* query_copy,
             const ggml_tensor* raw_key, const ggml_tensor* residual) {
        candidate_ = false;
        need(key_copy == key_copy_ && query_copy == query_copy_ && raw_key == raw_key_ && residual == residual_,
             "PLE gate: logical producer binding differs");
        require_scheduled_source(key_copy, key_);
        require_scheduled_source(query_copy, query_);
        need(strata_vk_native_ple_gate_run(component_.get(), key_, query_, gate_, diagnostic_) == 0,
             "PLE gate: finite domain refused");
        ggml_backend_tensor_get(key_, last_key_.data(), 0, last_key_.size());
        ggml_backend_tensor_get(query_, last_query_.data(), 0, last_query_.size());
        ggml_backend_tensor_get(gate_, last_gate_.data(), 0, last_gate_.size());
        candidate_ = true;
    }
    // This records a graph-completed candidate; the FFN and token are not yet committed.
    void verify_and_record(ggml_tensor* consumed, ggml_tensor* value, ggml_tensor* broadcast,
                           uint64_t position, int32_t token, const std::array<int32_t, 2>& previous,
                           const std::array<uint32_t, 16>& rows) {
        need(candidate_ && consumed == gate_ && broadcast && broadcast->op == GGML_OP_MUL &&
             broadcast->src[1] == gate_ && broadcast->src[0] && broadcast->src[0]->src[0] &&
             broadcast->src[0]->src[0]->src[0] == value,
             "PLE gate: actual value broadcast or gate bank differs");
        auto unchanged = [&](ggml_tensor* tensor, const auto& expected) {
            std::vector<uint8_t> bytes(expected.size());
            ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
            need(std::equal(bytes.begin(), bytes.end(), expected.begin()), "PLE gate: producer or gate bank changed");
        };
        unchanged(key_, last_key_); unchanged(query_, last_query_); unchanged(gate_, last_gate_);
        if (recorded_) return;
        auto path = directory_ / "first-candidate";
        need(std::filesystem::create_directory(path), "PLE gate: receipt exists");
        auto save = [&](const char* name, ggml_tensor* tensor) {
            std::vector<uint8_t> bytes(ggml_nbytes(tensor));
            ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
            std::ofstream file(path / name, std::ios::binary);
            file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            file.flush();
            need(bool(file), "PLE gate: receipt write");
        };
        save("key.f32le", key_); save("query.f32le", query_);
        save("gate.f32le", gate_); save("diagnostic.f32le", diagnostic_);
        std::ofstream meta(path / "metadata.json");
        meta << "{\"position\":" << position << ",\"token\":" << token
             << ",\"previous\":[" << previous[0] << "," << previous[1] << "],\"rows\":[";
        for (size_t i = 0; i < rows.size(); ++i) meta << (i ? "," : "") << rows[i];
        meta << "],\"direct_normalized_banks\":true,\"direct_gate_broadcast\":true,"
                "\"input_and_gate_unchanged\":true,\"graph_completed\":true,\"committed\":false,\"qualified\":false}\n";
        meta.flush();
        need(bool(meta), "PLE gate: metadata write");
        recorded_ = true;
    }
};
}
