// Test-only first real HC projection discriminator: no experts, session or
// inference loop.
#include "../../src/core/portable_graph.hpp"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-vulkan.h"
#include "strata/artifact/dequant.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
using namespace strata::core;
using namespace strata::core::portable_detail;
namespace {
constexpr int H = 2560, N = 4 * H, R = 320;
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
std::vector<float> run(const void *data, bool widened, const std::vector<float> &input, ggml_backend_t accelerator) {
    Backend cpu(ggml_backend_cpu_init(), ggml_backend_free);
    require(bool(cpu), "CPU init failed");
    ggml_backend_cpu_set_n_threads(cpu.get(), 4);
    Context weights(ggml_init({2u << 20, nullptr, true}), ggml_free),
        graphctx(ggml_init({2u << 20, nullptr, true}), ggml_free);
    require(bool(weights) && bool(graphctx), "metadata allocation failed");
    auto *w = ggml_new_tensor_2d(weights.get(), widened ? GGML_TYPE_F32 : GGML_TYPE_Q8_0, N, R);
    Buffer buffer(ggml_backend_alloc_ctx_tensors(weights.get(), cpu.get()), ggml_backend_buffer_free);
    require(bool(buffer), "weight allocation failed");
    ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(w, data, 0, ggml_nbytes(w));
    auto *x = ggml_new_tensor_2d(graphctx.get(), GGML_TYPE_F32, N, 1);
    ggml_set_input(x);
    auto *y = ggml_mul_mat(graphctx.get(), w, x);
    // F32 control requests F32 accumulation. Q8 arm retains the product's default
    // policy.
    if (widened)
        require(ggml_prec_set_acc(y, GGML_PREC_F32) && ggml_prec_set_src(y, GGML_PREC_F32, 1),
                "F32 precision request declined");
    ggml_set_name(y, "blk.0.hc_attn_down.first_boundary");
    ggml_set_output(y);
    auto *graph = ggml_new_graph(graphctx.get());
    ggml_build_forward_expand(graph, y);
    PortableFfnOptions options;
    options.accelerator = accelerator;
    options.request_accelerator = bool(accelerator);
    std::vector<PortablePlacement> placement;
    size_t size = 0;
    auto scheduler = schedule(graph, "first-hc-down", options, cpu, placement, size);
    require(placement.size() == 1 && placement[0].operation == "MUL_MAT" && placement[0].fallback_reason.empty(),
            "unexpected computation/fallback");
    require(placement[0].backend == ggml_backend_name(accelerator ? accelerator : cpu.get()), "wrong actual backend");
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * 4);
    require(ggml_backend_sched_graph_compute(scheduler.get(), graph) == GGML_STATUS_SUCCESS, "compute failed");
    std::vector<float> output(R);
    ggml_backend_tensor_get(y, output.data(), 0, output.size() * 4);
    for (float f : output)
        require(std::isfinite(f), "nonfinite output");
    std::cerr << "boundary arm=" << (widened ? "exact-F32" : "Q8-default") << " backend=" << placement[0].backend
              << " actual_device_graph_bytes="
              << (accelerator ? ggml_backend_sched_get_buffer_size(scheduler.get(), accelerator) : 0) << "\n";
    return output;
}
void metrics(const char *name, const std::vector<float> &a, const std::vector<float> &b, bool &comma) {
    require(a.size() == b.size(), "metric shape mismatch");
    double sum = 0, base = 0, maximum = 0;
    size_t at = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = double(b[i]) - a[i];
        sum += d * d;
        base += double(a[i]) * a[i];
        if (std::abs(d) > maximum) {
            maximum = std::abs(d);
            at = i;
        }
    }
    if (comma)
        std::cout << ",";
    comma = true;
    std::cout << "\n\"" << name << "\":{\"max_abs\":" << maximum << ",\"rms\":" << std::sqrt(sum / a.size())
              << ",\"relative_l2\":" << std::sqrt(sum) / std::max(1., std::sqrt(base)) << ",\"max_channel\":" << at
              << "}";
}
} // namespace
int main(int argc, char **argv) {
    try {
        require(argc == 4 || argc == 3, "usage: portable_hc_boundary (--model "
                                        "SHARD1 | --synthetic) DEVICE_INDEX");
        const bool synthetic = std::string(argv[1]) == "--synthetic";
        require((synthetic && argc == 3) || (!synthetic && std::string(argv[1]) == "--model" && argc == 4),
                "invalid arguments");
        std::string index = argv[argc - 1];
        size_t consumed = 0;
        auto device = std::stoul(index, &consumed);
        require(consumed == index.size() && device < size_t(ggml_backend_vk_get_device_count()),
                "Vulkan device unavailable");
        Backend accelerator(ggml_backend_vk_init(device), ggml_backend_free);
        require(bool(accelerator), "Vulkan init failed");
        std::vector<uint8_t> q8(size_t(N) * R / 32 * 34);
        std::vector<float> input(N), decoded(size_t(N) * R);
        std::unique_ptr<strata::GgufModel> model;
        if (synthetic) {
            for (size_t b = 0; b < q8.size() / 34; ++b) {
                uint16_t scale = 0x1800;
                std::memcpy(q8.data() + b * 34, &scale, 2);
                for (size_t j = 0; j < 32; ++j) {
                    int8_t v = int8_t(int((b * 7 + j * 3) % 23) - 11);
                    std::memcpy(q8.data() + b * 34 + 2 + j, &v, 1);
                }
            }
            for (int i = 0; i < N; ++i)
                input[i] = std::sin(float(i) * .017f) + .001f * float(i % 13);
        } else {
            model = std::make_unique<strata::GgufModel>(strata::gguf_split_paths(argv[2]));
            auto tensor = [&](const char *name, uint32_t type, std::vector<uint64_t> shape) {
                size_t s = 0;
                auto *t = model->find(name, &s);
                require(t && t->type == type && t->shape == shape && model->in_bounds(*t, s),
                        std::string("invalid ") + name);
                return static_cast<const uint8_t *>(model->shard(s).tensor_data(*t));
            };
            auto *w = tensor("blk.0.hc_attn_down.weight", 8, {N, R});
            std::memcpy(q8.data(), w, q8.size());
            auto *norm = tensor("blk.0.hc_attn_norm.weight", 0, {N});
            auto *embedding = tensor("token_embd.weight", 8, {H, 248320});
            constexpr size_t token = 248045;
            std::vector<float> row(H);
            for (int b = 0; b < H / 32; ++b)
                strata::dequantize_q8_0(embedding + token * (H / 32 * 34) + b * 34, row.data() + b * 32);
            double square = 0;
            for (float f : row) {
                require(std::isfinite(f), "nonfinite embedding");
                square += double(f) * f;
            }
            // One common independently computed input eliminates RMS/backend
            // scheduling as a confounder.
            float inverse = 1.f / std::sqrt(float(square / H) + 1e-6f);
            for (int i = 0; i < N; ++i) {
                float gain;
                std::memcpy(&gain, norm + i * 4, 4);
                input[i] = row[i % H] * inverse * gain;
            }
        }
        for (size_t b = 0; b < q8.size() / 34; ++b)
            strata::dequantize_q8_0(q8.data() + b * 34, decoded.data() + b * 32);
        for (float f : input)
            require(std::isfinite(f), "nonfinite common input");
        for (float f : decoded)
            require(std::isfinite(f), "nonfinite decoded weight");
        std::vector<float> oracle(R);
        for (int r = 0; r < R; ++r) {
            double value = 0;
            for (int c = 0; c < N; ++c)
                value += double(decoded[size_t(r) * N + c]) * input[c];
            oracle[r] = float(value);
        }
        const auto *weight_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
        require(weight_traits && weight_traits->vec_dot_type == GGML_TYPE_Q8_0,
                "unexpected CPU Q8 dot activation type");
        const auto *activation_traits = ggml_get_type_traits_cpu(weight_traits->vec_dot_type);
        require(activation_traits && activation_traits->from_float, "CPU activation quantizer missing");
        std::vector<uint8_t> packed_input(N / 32 * 34);
        std::vector<float> quantized_input(N), quantized_oracle(R);
        activation_traits->from_float(input.data(), packed_input.data(), N);
        for (int b = 0; b < N / 32; ++b)
            strata::dequantize_q8_0(packed_input.data() + b * 34, quantized_input.data() + b * 32);
        for (int r = 0; r < R; ++r) {
            double value = 0;
            for (int c = 0; c < N; ++c)
                value += double(decoded[size_t(r) * N + c]) * quantized_input[c];
            quantized_oracle[r] = float(value);
        }
        auto cpuq = run(q8.data(), false, input, nullptr), gpuq = run(q8.data(), false, input, accelerator.get());
        auto cpuf = run(decoded.data(), true, input, nullptr),
             gpuf = run(decoded.data(), true, input, accelerator.get());
        std::cout.precision(12);
        std::cout << "{\"schema\":\"portable-hc-boundary-v1\",\"token\":248045,"
                     "\"layer\":0,\"input_width\":"
                  << N << ",\"rows\":" << R << ",\"synthetic\":" << (synthetic ? "true" : "false")
                  << ",\"diagnostics_only\":true,\"metrics\":{";
        bool comma = false;
        metrics("Q8_CPU_vs_Vulkan", cpuq, gpuq, comma);
        metrics("F32_CPU_vs_Vulkan", cpuf, gpuf, comma);
        metrics("scalar_vs_F32_CPU", oracle, cpuf, comma);
        metrics("scalar_vs_F32_Vulkan", oracle, gpuf, comma);
        metrics("CPU_quantized_input_oracle_vs_Q8_CPU", quantized_oracle, cpuq, comma);
        metrics("scalar_vs_CPU_quantized_input_oracle", oracle, quantized_oracle, comma);
        metrics("scalar_vs_Q8_CPU", oracle, cpuq, comma);
        metrics("scalar_vs_Q8_Vulkan", oracle, gpuq, comma);
        std::cout << "\n}}\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
