#include "ggml-backend.h"
#include "portable_qsa_bf16_fixture.hpp"
#ifdef STRATA_PORTABLE_QSA_BF16_VULKAN
#include "ggml-vulkan.h"
#endif
using namespace qsa_fixture;
int main(int argc, char **argv) {
    try {
        // Exhaustive bit-exact widening, including signed zeros, subnormals,
        // infinities and NaN payload bits. This is the production row decoder.
        std::vector<uint8_t> codes(65536 * 2);
        for (uint32_t value = 0; value < 65536; ++value) {
            codes[2 * value] = uint8_t(value);
            codes[2 * value + 1] = uint8_t(value >> 8);
        }
        std::vector<float> decoded(65536);
        strata::dequantize_bf16(codes.data(), decoded.data(), int(decoded.size()));
        for (uint32_t value = 0; value < 65536; ++value) {
            uint32_t bits;
            std::memcpy(&bits, &decoded[value], 4);
            check(bits == value << 16, "BF16 widening must preserve all bits exactly");
        }
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(nullptr, ggml_backend_free);
        PortableQsaOptions options;
        options.capacity = 5;
        if (argc != 1) {
            check(argc == 3 && std::string(argv[1]) == "--vulkan", "usage: portable_qsa_bf16_test [--vulkan INDEX]");
#ifdef STRATA_PORTABLE_QSA_BF16_VULKAN
            size_t used = 0;
            const auto index = std::stoul(argv[2], &used);
            check(used == std::strlen(argv[2]) && index < size_t(ggml_backend_vk_get_device_count()),
                  "invalid Vulkan index");
            backend.reset(ggml_backend_vk_init(index));
            check(bool(backend), "Vulkan unavailable");
            options.execution.accelerator = backend.get();
            options.execution.request_accelerator = true;
#else
            throw std::runtime_error("this build has no Vulkan backend");
#endif
        }
        for (bool reuse : {false, true})
        for (bool resident : {false, true}) {
            options.reuse_attention = reuse;
            std::printf("dense weight fixture: resident=%d\n", int(resident));
            options.execution.resident_dense_weights = resident;
            Bf16Fixture fixture;
            std::string error;
            auto layer = PortableQsaLayer::create(3, fixture.views, options, error);
            check(bool(layer), error);
            auto reference = PortableQsaLayer::create(3, fixture.f32_views, options, error);
            check(bool(reference), error);
            const auto memory = layer->memory_usage(), expected_memory = reference->memory_usage();
            check((memory.device_weights > 0) == (resident && bool(backend)), "BF16 resident ledger");
            check(memory.host_weights + memory.device_weights == layer->weight_bytes(), "BF16 actual weight reconciliation");
            check(memory.host_weights == expected_memory.host_weights && layer->weight_bytes() == reference->weight_bytes(),
                  "BF16 is priced as persistent F32 storage");
            check(memory.host_extra_bound == expected_memory.host_extra_bound + 2560 * sizeof(float),
                  "one-row conversion scratch is admitted");
            check(memory.device_graph == expected_memory.device_graph &&
                      memory.device_extra_bound == expected_memory.device_extra_bound,
                  "F32 scheduler/device reservation parity");
            for (int bad = 0; bad < 4; ++bad) {
                auto views = fixture.views;
                for (auto &v : views) {
                    if (bad == 0 && v.name == "blk.3.attn_q.weight")
                        v.type = 30;
                    if (v.name != "blk.3.indexer.q_proj.weight")
                        continue;
                    if (bad == 1)
                        v.bytes -= 2;
                    if (bad == 2)
                        v.shape[1] += 1;
                    if (bad == 3)
                        v.type = 1;
                }
                check(!PortableQsaLayer::create(3, views, options, error),
                      "role/shape/truncation validation must remain strict");
            }
            auto limited = options;
            limited.execution.weight_budget = layer->weight_bytes() - 1;
            check(!PortableQsaLayer::create(3, fixture.views, limited, error), "widened aligned storage exceeds budget");
            for (auto &source : fixture.bf16) {
                source.clear();
                source.shrink_to_fit();
            }
            PortableRoute route, reference_route;
            for (size_t token = 0; token < options.capacity; ++token) {
                if (token == 2) {
                    const auto keys = layer->keys();
                    const auto residual = layer->last_residual();
                    check(!layer->step(
                              token, fixture.base.input(int(token)),
                              [](int64_t, const PortableRoute &, std::vector<float> &, std::string &) { return false; },
                              route, error),
                          "injected failure");
                    check(layer->tokens() == token && layer->keys() == keys && layer->last_residual() == residual,
                          "BF16 layer failure rollback");
                }
                PortableQsaTrace actual, expected;
                check(layer->step(token, fixture.base.input(int(token)), experts(), route, error, &actual), error);
                check(reference->step(token, fixture.base.input(int(token)), experts(), reference_route, error, &expected),
                      error);
                close(actual.index_raw_key, expected.index_raw_key, 0, "exact widened key parity");
                close(actual.index_query, expected.index_query, 0, "exact widened query parity");
                close(layer->last_residual(), reference->last_residual(), 0, "exact full layer F32 reference parity");
                check(layer->pooled_keys() == reference->pooled_keys() && route.ids == reference_route.ids,
                      "pool/route parity");
            }
            check(layer->reset(error) && reference->reset(error), error);
            check(layer->tokens() == 0 && layer->keys().empty(), "reset history");
            check(layer->step(0, fixture.base.input(0), experts(), route, error), error);
            check(reference->step(0, fixture.base.input(0), experts(), reference_route, error), error);
            close(layer->last_residual(), reference->last_residual(), 0, "reset exact parity");
            size_t conversions = 0;
            for (const auto &p : layer->placements()) {
                if (p.phase == "load") {
                    check(p.operation == "BF16_TO_F32_EXACT" && p.backend == "CPU", "explicit host load conversion");
                    ++conversions;
                } else if (backend && p.phase != "host")
                    check(p.metadata_only || (p.backend == ggml_backend_name(backend.get()) && p.fallback_reason.empty()),
                          "strict Vulkan tensor execution");
            }
            check(conversions == 2, "only indexer Q/K widened");
            std::printf("portable_qsa_bf16_test: 65536 exact decoder patterns; mixed BF16/Q8/F32 H2560, source lifetime, "
                        "bit-exact F32 parity, >FP16-range weights, role/range/budget/reset/rollback passed; backend=%s "
                        "host_weights=%zu device_scheduler=%zu conversion_scratch=10240\n",
                        backend ? ggml_backend_name(backend.get()) : "CPU", memory.host_weights, memory.device_graph);
        } // validate streamed and resident dense weights
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
