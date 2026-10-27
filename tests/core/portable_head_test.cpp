#include "ggml-backend.h"
#include "strata/artifact/dequant.hpp"
#include "strata/core/portable_model.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#ifdef STRATA_PORTABLE_MODEL_VULKAN
#include "ggml-vulkan.h"
#endif
using namespace strata::core;
namespace {
void check(bool ok, const std::string &e) {
    if (!ok)
        throw std::runtime_error(e);
}
} // namespace
int main(int argc, char **argv) {
    try {
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(nullptr, ggml_backend_free);
        PortableFfnOptions options;
        if (argc != 1) {
            check(argc == 3 && std::string(argv[1]) == "--vulkan", "usage: portable_head_test [--vulkan INDEX]");
#ifdef STRATA_PORTABLE_MODEL_VULKAN
            size_t used = 0;
            const auto index = std::stoul(argv[2], &used);
            check(used == std::strlen(argv[2]) && index < size_t(ggml_backend_vk_get_device_count()),
                  "invalid Vulkan device");
            backend.reset(ggml_backend_vk_init(index));
            check(bool(backend), "requested Vulkan unavailable");
            options.accelerator = backend.get();
            options.request_accelerator = true;
#else
            throw std::runtime_error("this build has no Vulkan backend");
#endif
        }
        constexpr size_t H = 2560, HC = 4, R = 320, V = 64;
        std::vector<float> norm(H * HC);
        for (size_t i = 0; i < norm.size(); ++i)
            norm[i] = 0.7f + float(i % 7) * 0.1f;
        std::vector<std::vector<uint8_t>> data(3);
        std::vector<std::vector<float>> decoded(3);
        const size_t inputs[]{H * HC, R, H}, outputs[]{R, H * HC, V};
        for (size_t w = 0; w < 3; ++w) {
            data[w].resize(inputs[w] * outputs[w] / 32 * 34);
            decoded[w].resize(inputs[w] * outputs[w]);
            for (size_t b = 0; b < data[w].size() / 34; ++b) {
                const uint16_t scale = 0x2800; // exactly 1/32
                std::memcpy(data[w].data() + b * 34, &scale, 2);
                for (size_t i = 0; i < 32; ++i) {
                    const int8_t code = int8_t(int((i * 3 + b * 7 + w) % 11) - 5);
                    std::memcpy(data[w].data() + b * 34 + 2 + i, &code, 1);
                }
                strata::dequantize_q8_0(data[w].data() + b * 34, decoded[w].data() + b * 32);
            }
        }
        std::vector<PortableTensorView> weights{
            {"output_hc_norm.weight", {H * HC}, 0, norm.data(), norm.size() * 4},
            {"output_hc_down.weight", {H * HC, R}, 8, data[0].data(), data[0].size()},
            {"output_hc_up.weight", {R, H * HC}, 8, data[1].data(), data[1].size()},
            {"output.weight", {H, V}, 8, data[2].data(), data[2].size()}};
        std::string error;
        auto head = PortableHead::create({H, HC, R, V, 1e-6f}, weights, options, error);
        check(bool(head), error);
        auto mm = [&](size_t w, const std::vector<float> &x) {
            std::vector<float> result(outputs[w]);
            for (size_t r = 0; r < result.size(); ++r) {
                double sum = 0;
                for (size_t i = 0; i < x.size(); ++i)
                    sum += double(decoded[w][r * x.size() + i]) * x[i];
                result[r] = float(sum);
            }
            return result;
        };
        for (size_t token = 0; token < 3; ++token) {
            std::vector<float> input(H * HC), normalized(H * HC), logits;
            for (size_t i = 0; i < input.size(); ++i)
                input[i] = float(int((i * 13 + token * 7) % 29) - 14) * 0.1f;
            for (size_t s = 0; s < HC; ++s) {
                double sq = 0;
                for (size_t i = 0; i < H; ++i)
                    sq += double(input[s * H + i]) * input[s * H + i];
                const float inv = 1 / std::sqrt(float(sq / H) + 1e-6f);
                for (size_t i = 0; i < H; ++i)
                    normalized[s * H + i] = input[s * H + i] * inv * norm[s * H + i];
            }
            auto low = mm(0, normalized);
            for (float &v : low) {
                v *= 0.25f;
                v = v / (1 + std::exp(-v));
            }
            const auto gates = mm(1, low);
            std::vector<float> mixed(H, 0);
            for (size_t s = 0; s < HC; ++s)
                for (size_t i = 0; i < H; ++i)
                    mixed[i] += normalized[s * H + i] / (1 + std::exp(-gates[s * H + i])) * 0.25f;
            const auto expected = mm(2, mixed);
            check(head->run(input, logits, error), error);
            for (size_t i = 0; i < V; ++i)
                check(std::fabs(logits[i] - expected[i]) <= 0.012f * (1 + std::fabs(expected[i])),
                      "independent HC/head scalar reference");
            // ggml Q8 matvec quantizes activation; independent F32 dots above
            // use an a priori screening tolerance, not an exact-bit claim.
        }
        std::vector<float> out;
        check(!head->run({}, out, error) && out.empty(), "head input rejection");
        auto small = PortableFfnOptions{};
        small.weight_budget = 1;
        check(!PortableHead::create({H, HC, R, V, 1e-6f}, weights, small, error), "head weight cap");
        small = {};
        small.graph_budget = 1;
        check(!PortableHead::create({H, HC, R, V, 1e-6f}, weights, small, error), "head graph cap");
        const auto memory = head->memory_usage();
        check(memory.host_weights == head->weight_bytes() &&
                  (backend ? memory.device_graph > 0 : memory.device_graph == 0 && memory.host_graph > 0),
              "head allocation report");
        if (backend) {
            for (const auto &p : head->placements())
                check(p.metadata_only || (p.backend == ggml_backend_name(backend.get()) && p.fallback_reason.empty()),
                      "head tensor work must execute on requested Vulkan device");
            std::printf("PASS strict Vulkan portable head backend=%s actual_device_scheduler=%zu workspace_bound=%zu\n",
                        ggml_backend_name(backend.get()), memory.device_graph, memory.device_extra_bound);
        }
        std::puts("portable_head_test: independent Q8 HC/global head scalar reference, input/cap checks passed");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
