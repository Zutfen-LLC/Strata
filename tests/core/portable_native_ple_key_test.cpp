#include "portable_native_ple_key.hpp"
#include "ggml-cpu.h"
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <vector>
using namespace strata::core::portable_detail;
namespace {
ggml_backend_t backend = nullptr;
const ggml_tensor* input_bank = nullptr;
const ggml_tensor* packed_bank = nullptr;
const ggml_tensor* resident_weight = nullptr;
size_t quant_calls = 0, dot_calls = 0;
bool fail_dot = false;
std::vector<float> expected_embedding;
void need(bool v, const char* message) { if (!v) throw std::runtime_error(message); }
}
namespace strata::core::portable_detail {
struct StrataVkNativeStages {};
StrataVkNativeStages* strata_vk_native_stages_create_quant_rcp_observation(ggml_backend_t, const char* p, size_t budget) {
    need(budget >= 512, "quant budget"); std::ofstream(p) << "host provider"; return new StrataVkNativeStages;
}
size_t strata_vk_native_stages_bytes(const StrataVkNativeStages*) { return 512; }
void strata_vk_native_stages_destroy(StrataVkNativeStages* p) noexcept { delete p; }
void strata_vk_native_stages_run(StrataVkNativeStages*, uint32_t mode, const ggml_tensor* input,
                                 const ggml_tensor*, ggml_tensor*, ggml_tensor* packed,
                                 uint32_t count, const int32_t*, size_t) {
    need(mode == 0 && count == 2560 && input == input_bank && input->buffer &&
         ggml_backend_buffer_get_type(input->buffer) == ggml_backend_get_default_buffer_type(backend),
         "actual PLE embedding bank or device differs");
    std::vector<float> actual(count); ggml_backend_tensor_get(input, actual.data(), 0, actual.size() * 4);
    need(actual == expected_embedding, "gathered PLE embedding differs");
    std::vector<uint8_t> q(2880, 0); ggml_backend_tensor_set(packed, q.data(), 0, q.size());
    packed_bank = packed; ++quant_calls;
}
struct StrataVkQ8Wo {};
StrataVkQ8Wo* strata_vk_head_vocab_create(ggml_backend_t, const char* p, size_t budget) {
    need(budget >= 512, "dot budget"); std::ofstream(p) << "host provider"; return new StrataVkQ8Wo;
}
size_t strata_vk_q8_wo_bytes(const StrataVkQ8Wo*) { return 512; }
void strata_vk_q8_wo_destroy(StrataVkQ8Wo* p) noexcept { delete p; }
void strata_vk_q8_wo_run(StrataVkQ8Wo*, const ggml_tensor* weight, const ggml_tensor* packed,
                         ggml_tensor* output, size_t rows) {
    need(!fail_dot && rows == 10240 && weight->view_src == resident_weight && weight->buffer &&
         packed == packed_bank && output->buffer &&
         ggml_backend_buffer_get_type(output->buffer) == ggml_backend_get_default_buffer_type(backend),
         "resident PLE role, packed bank or output device differs");
    std::vector<float> value(rows); for (size_t i = 0; i < rows; ++i) value[i] = float(i);
    ggml_backend_tensor_set(output, value.data(), 0, value.size() * 4); ++dot_calls;
}
}
int main(int argc, char** argv) { try {
    need(argc == 2, "scratch root");
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> b(ggml_backend_cpu_init(), ggml_backend_free);
    need(bool(b), "backend"); backend = b.get();
    std::unique_ptr<ggml_context, decltype(&ggml_free)> c(ggml_init({16384, nullptr, true}), ggml_free);
    auto* weight = ggml_new_tensor_2d(c.get(), GGML_TYPE_Q8_0, 2560, 10240); resident_weight = weight;
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> wb(ggml_backend_alloc_ctx_tensors(c.get(), b.get()), ggml_backend_buffer_free);
    need(bool(wb), "weight buffer"); std::vector<uint8_t> role(ggml_nbytes(weight), 0);
    ggml_backend_tensor_set(weight, role.data(), 0, role.size());
    auto path = std::filesystem::path(argv[1]) / ("ple-key-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const size_t budget = PortableNativePleKey::estimate(b.get());
    bool refused = false; try { PortableNativePleKey bad(b.get(), weight, role.data(), role.size(), path.string(), budget - 1); } catch (...) { refused = true; }
    need(refused && !std::filesystem::exists(path), "one byte short must refuse before allocation");
    role[1] = 0x7c; refused = false;
    try { PortableNativePleKey bad(b.get(), weight, role.data(), role.size(), path.string(), budget); } catch (...) { refused = true; }
    need(refused && !std::filesystem::exists(path), "nonfinite weight scale accepted"); role[1] = 0;
    role[2] = 1; refused = false;
    try { PortableNativePleKey bad(b.get(), weight, role.data(), role.size(), path.string(), budget); } catch (...) { refused = true; }
    need(refused && !std::filesystem::exists(path), "different resident role accepted"); role[2] = 0;
    PortableNativePleKey key(b.get(), weight, role.data(), role.size(), path.string(), budget);
    input_bank = key.input_bank(); std::vector<float> embedding(2560, 1.f); expected_embedding = embedding;
    auto* graph_bank = key.output_bank();
    need(graph_bank && graph_bank->buffer && ggml_nbytes(graph_bank) == 10240 * 4, "prebound output bank");
    auto* output = key.run(embedding); need(output == graph_bank && key.calls() == 1, "actual output bank");
    std::vector<float> values(10240); ggml_backend_tensor_get(output, values.data(), 0, values.size() * 4);
    need(values.front() == 0.f && values.back() == 10239.f && quant_calls == 1 && dot_calls == 1, "one quantizer and MMVQ");
    embedding[0] = std::numeric_limits<float>::infinity(); refused = false;
    try { key.run(embedding); } catch (...) { refused = true; }
    need(refused && key.calls() == 1 && quant_calls == 1 && dot_calls == 1, "invalid embedding must not dispatch");
    embedding[0] = 1.f; embedding[1] = -1.f; expected_embedding = embedding; fail_dot = true; refused = false;
    try { key.run(embedding); } catch (...) { refused = true; }
    need(refused && key.calls() == 1, "failed dot must not publish a completed call");
    fail_dot = false; output = key.run(embedding);
    need(output && key.calls() == 2 && quant_calls == 3 && dot_calls == 2, "retry after failed dot");
    std::puts("PLE key owner, admitted budget, actual banks, refusal and retry passed with host providers"); return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; } }
