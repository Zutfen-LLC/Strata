#include "ggml-backend.h"
#include "strata/artifact/dequant.hpp"
#include "strata/core/portable_model.hpp"
#include "portable_logits_guard_test.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <limits>
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
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> guard_cpu(ggml_backend_cpu_init(), ggml_backend_free);
        logits_guard_test(backend ? backend.get() : guard_cpu.get());
        for (bool resident : {false, true}) {
            std::printf("dense weight fixture: resident=%d\n", int(resident));
            options.resident_dense_weights = resident;
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
            const auto envelope=PortableHead::admission({H,HC,R,V,1e-6f},weights,options,true);
            auto head = PortableHead::create({H, HC, R, V, 1e-6f}, weights, options, error);
            check(bool(head), error);
            auto guarded_head = PortableHead::create({H, HC, R, V, 1e-6f}, weights, options, error, true);
            check(bool(guarded_head), error);
            auto bounded_options=options;bounded_options.graph_budget=envelope.graph_bound;
            auto bounded_head=PortableHead::create({H,HC,R,V,1e-6f},weights,bounded_options,error,true);
            check(bool(bounded_head),"head must allocate inside metadata-sized budget: "+error);
            const auto estimated_memory=guarded_head->memory_usage();
            check(estimated_memory.host_graph+estimated_memory.device_graph<=envelope.scheduler_bound,
                  "actual scheduler allocation exceeds metadata node/copy envelope");
            check(estimated_memory.device_graph+estimated_memory.device_state+estimated_memory.device_extra_bound+
                  estimated_memory.device_shared_conversion_bound<=envelope.graph_bound,
                  "actual device graph/status/private bookkeeping exceeds head estimate");
            check(!envelope.verified_vulkan,"small/unproven head cannot earn full-vocabulary direct credit");
            auto invalid=weights;invalid[3].bytes--;
            bool refused=false;
            try{PortableHead::admission({H,HC,R,V,1e-6f},invalid,options,true);}
            catch(const std::exception&){refused=true;}
            check(refused,"metadata estimator must reject payload mismatch before allocations");
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
                std::vector<float> guarded_logits;
                std::vector<float> bounded_logits;
                check(bounded_head->run(input,bounded_logits,error) && bounded_logits==logits,
                      "head metadata admission must preserve exact same-backend outputs");
                check(guarded_head->run(input, guarded_logits, error) && guarded_logits == logits,
                      "guard must not change successful logits bytes");
                for (size_t i = 0; i < V; ++i)
                    check(std::fabs(logits[i] - expected[i]) <= 0.012f * (1 + std::fabs(expected[i])),
                          "independent HC/head scalar reference");
                // ggml Q8 matvec quantizes activation; independent F32 dots above
                // use an a priori screening tolerance, not an exact-bit claim.
            }
            std::vector<float> out;
            check(!head->run({}, out, error) && out.empty(), "head input rejection");
            std::vector<float> finite_edges(H * HC, 0.125f);
            finite_edges[0] = 0.0f;
            finite_edges[1] = -0.0f;
            finite_edges[finite_edges.size() / 2] = std::numeric_limits<float>::min();
            finite_edges.back() = std::numeric_limits<float>::denorm_min();
            for (float invalid : {std::numeric_limits<float>::quiet_NaN(),
                                  std::numeric_limits<float>::infinity(),
                                  -std::numeric_limits<float>::infinity()})
                for (size_t at : {size_t(0), finite_edges.size() / 2, finite_edges.size() - 1}) {
                    auto bad = finite_edges;
                    bad[at] = invalid;
                    out = {1};
                    check(!head->run(bad, out, error) && out.empty() &&
                          error == "portable head: nonfinite residual", "head input finite diagnostic/output clear");
                }
            check(head->run(finite_edges, out, error), error);
            const auto first_edges = out;
            check(head->run(finite_edges, out, error) && out.size() == first_edges.size() &&
                  std::memcmp(out.data(), first_edges.data(), out.size() * sizeof(float)) == 0,
                  "head finite edge replay bytes");
            for (bool guarded : {false, true}) {
                for (bool quantized : {false, true})
                for (uint32_t invalid_bits : {0x7f800000u, 0xff800000u, 0x7fc00001u, 0xffc00001u, 0x7f800001u, 0xff800001u}) {
                    auto invalid_weights = weights;
                    std::vector<float> invalid_output(H * V, 0.01f);
                    invalid_output.back() = std::bit_cast<float>(invalid_bits);
                    invalid_weights.back().type = 0;
                    invalid_weights.back().data = invalid_output.data();
                    invalid_weights.back().bytes = invalid_output.size() * sizeof(float);
                    auto invalid_q8 = data[2];
                    if (quantized) {
                        const uint16_t half_bits = uint16_t((invalid_bits >> 16) & 0x8000u) | 0x7c00u |
                            ((invalid_bits & 0x7fffffu) ? ((invalid_bits & 0x400000u) ? 0x201u : 1u) : 0u);
                        std::memcpy(invalid_q8.data() + invalid_q8.size() - 34, &half_bits, sizeof(half_bits));
                        invalid_weights.back().type = 8;
                        invalid_weights.back().data = invalid_q8.data();
                        invalid_weights.back().bytes = invalid_q8.size();
                    }
                    auto bad = PortableHead::create({H, HC, R, V, 1e-6f}, invalid_weights, options, error, guarded);
                    check(bool(bad), error);
                    if (guarded && backend) {
                        bool status_on_device = false;
                        for (const auto &p : bad->placements()) {
                            check(p.metadata_only || (p.backend == ggml_backend_name(backend.get()) &&
                                  p.fallback_reason.empty()), "injected head must stay on requested Vulkan backend");
                            status_on_device |= p.operation == "BIT_NONFINITE" && p.tensor == "portable_logits_status" &&
                                                p.backend == ggml_backend_name(backend.get());
                        }
                        check(status_on_device, "scheduled injected head must include device status guard");
                    }
                    out = {1};
                    check(!bad->run(finite_edges, out, error) && out.empty() &&
                          error == (guarded ? "portable head: nonfinite logits (integer status guard)" :
                                             "portable head: nonfinite logits"),
                          "scheduled head must reject at requested guard before logits download");
                    check(head->run(finite_edges, out, error) && out.size() == first_edges.size() &&
                          std::memcmp(out.data(), first_edges.data(), out.size() * sizeof(float)) == 0,
                          "head recovery after invalid calls");
                }
            }
            // Same scheduled head/guard must recover from a bad status without
            // retaining stale flags. Finite HC norm weights deliberately overflow
            // on nonzero input; zero input remains finite for either head encoding.
            for (bool quantized : {false, true}) {
                auto overflow_weights = weights;
                std::vector<float> huge_norm(H * HC, std::numeric_limits<float>::max());
                std::vector<float> output_f32(H * V, 0.01f);
                overflow_weights[0].data = huge_norm.data();
                if (!quantized) {
                    overflow_weights.back().type = 0;
                    overflow_weights.back().data = output_f32.data();
                    overflow_weights.back().bytes = output_f32.size() * sizeof(float);
                }
                auto repeat = PortableHead::create({H, HC, R, V, 1e-6f}, overflow_weights, options, error, true);
                check(bool(repeat), error);
                for (int trial = 0; trial < 2; ++trial) {
                    check(!repeat->run(finite_edges, out, error) && out.empty() &&
                          error == "portable head: nonfinite logits (integer status guard)", "scheduled bad status");
                    check(repeat->run(std::vector<float>(H * HC, 0), out, error) && out.size() == V,
                          "scheduled guard finite after failure");
                }
                const auto guarded_memory = repeat->memory_usage();
                check(backend ? guarded_memory.device_state >= 16 * sizeof(uint32_t) &&
                                guarded_memory.device_extra_bound >= (1u << 20) : guarded_memory.device_state == 0,
                      "aligned status and driver resource allowance ledger");
            }
            auto small = PortableFfnOptions{};
            small.weight_budget = 1;
            check(!PortableHead::create({H, HC, R, V, 1e-6f}, weights, small, error), "head weight cap");
            small = {};
            small.graph_budget = 1;
            check(!PortableHead::create({H, HC, R, V, 1e-6f}, weights, small, error), "head graph cap");
            const auto aligned_weights=PortableHead::weight_allocation_size(
                resident && backend ? backend.get() : guard_cpu.get(),{H,HC,R,V,1e-6f},weights);
            check(head->weight_bytes()<=aligned_weights && guarded_head->weight_bytes()<=aligned_weights,
                  "real head allocations must fit metadata-only selected-backend weight sizing");
            const auto memory = head->memory_usage();
            check(memory.host_weights + memory.device_weights == head->weight_bytes() &&
                      (memory.device_weights > 0) == (resident && bool(backend)) &&
                      (backend ? memory.device_graph > 0 : memory.device_graph == 0 && memory.host_graph > 0),
                  "head allocation report");
            if (backend) {
                for (const auto &p : guarded_head->placements())
                    check(p.metadata_only || (p.backend == ggml_backend_name(backend.get()) && p.fallback_reason.empty()),
                          "head tensor work must execute on requested Vulkan device");
                std::printf("PASS strict Vulkan portable head backend=%s actual_device_scheduler=%zu workspace_bound=%zu\n",
                            ggml_backend_name(backend.get()), memory.device_graph, memory.device_extra_bound);
            }
            std::puts("portable_head_test: independent Q8 HC/global head scalar reference, input/cap checks passed");
        } // validate streamed and resident dense weights
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
