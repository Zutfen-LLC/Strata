#include "ggml-vulkan.h"
#include "portable_qsa_selection.hpp"
#include <cstdlib>
using namespace qsa_fixture;
int main(int argc, char **argv) {
    try {
        int index = 0;
        std::string expected;
        for (int i = 1; i < argc; i += 2) {
            check(i + 1 < argc, "usage: --device INDEX --expect-device SUBSTRING");
            if (std::string(argv[i]) == "--device") {
                char *end = nullptr;
                long n = std::strtol(argv[i + 1], &end, 10);
                check(end != argv[i + 1] && *end == 0 && n >= 0 && n < GGML_VK_MAX_DEVICES, "bad index");
                index = int(n);
            } else if (std::string(argv[i]) == "--expect-device")
                expected = argv[i + 1];
            else
                check(false, "unknown option");
        }
        check(ggml_backend_vk_get_device_count() > index, "Vulkan device required; no skipped pass");
        char description[256] = {};
        ggml_backend_vk_get_device_description(index, description, sizeof(description));
        check(expected.empty() || std::string(description).find(expected) != std::string::npos,
              "unexpected device");
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> device(ggml_backend_vk_init(size_t(index)),
                                                                           ggml_backend_free);
        check(bool(device) && !ggml_backend_supports_buft(device.get(), ggml_backend_cpu_buffer_type()),
              "distinct Vulkan buffers required");
        for (bool reuse : {false, true})
        for (bool resident : {false, true}) {
            std::printf("dense weight fixture: resident=%d\n", int(resident));
            PortableQsaOptions options;
            options.execution.resident_dense_weights = resident;
            options.capacity = 9;
            options.reuse_attention = reuse;
            options.execution.accelerator = device.get();
            options.execution.request_accelerator = true;
            // Strict policy: selection failures are failures, never replaced by CPU results.

            Fixture f;
            std::string error;
            auto layer = PortableQsaLayer::create(3, f.views, options, error);
            check(bool(layer), error);
            check((layer->memory_usage().device_weights > 0) == resident, "QSA resident ledger");
            check(layer->memory_usage().host_weights + layer->memory_usage().device_weights == layer->weight_bytes(),
                  "QSA actual weight reconciliation");
            auto cpu_options = options;
            cpu_options.execution = {};
            cpu_options.reuse_attention = false;
            auto cpu = PortableQsaLayer::create(3, f.views, cpu_options, error);
            check(bool(cpu), error);
            PortableRoute route, control_route;
            PortableQsaTrace trace;
            Oracle oracle;
            // Same-backend legacy control separates reuse from cross-backend numerics.
            auto legacy_options = options;
            legacy_options.reuse_attention = false;
            auto legacy = PortableQsaLayer::create(3, f.views, legacy_options, error);
            check(bool(legacy), error);
            const auto initial_memory = layer->memory_usage();
            size_t peak_graph_bytes = 0;
            for (size_t t = 0; t < 9; ++t) {
                if (t == 0 || t == 1 || t == 3 || t == 4 || t == 7 || t == 8) {
                    const auto keys = layer->keys(), values = layer->values();
                    const auto pooled = layer->pooled_keys(), tail = layer->indexer_tail(),
                               last = layer->last_residual();
                    for (int failure = 0; failure < 4; ++failure) {
                        check(!layer->step(
                                  t, f.input(int(t)),
                                  [&](int64_t, const PortableRoute &, std::vector<float> &rows, std::string &) {
                                      if (failure == 0)
                                          return false;
                                      if (failure == 1)
                                          throw std::runtime_error("callback throw");
                                      rows.assign(failure == 2 ? 1 : 25600, failure == 3 ? INFINITY : 0);
                                      return true;
                                  },
                                  route, error),
                              "Vulkan rejected callback");
                        check(layer->tokens() == t && layer->keys() == keys && layer->values() == values &&
                                  layer->pooled_keys() == pooled && layer->indexer_tail() == tail &&
                                  layer->last_residual() == last,
                              "Vulkan rollback");
                    }
                }
                check(layer->step(t, f.input(int(t)), experts(), route, error, &trace), error);
                try {
                    oracle.check_step(f, t, f.input(int(t)), trace, *layer);
                } catch (const std::exception &e) {
                    throw std::runtime_error("token=" + std::to_string(t) + " " + e.what());
                }
                check(legacy->step(t, f.input(int(t)), experts(), control_route, error), error);
                check(legacy->keys() == layer->keys() && legacy->values() == layer->values() &&
                      legacy->pooled_keys() == layer->pooled_keys() && legacy->indexer_tail() == layer->indexer_tail() &&
                      legacy->last_residual() == layer->last_residual() && control_route.ids == route.ids &&
                      control_route.weights == route.weights, "same Vulkan exact reuse parity");
                if (reuse) {
                    auto memory = layer->memory_usage();
                    check(memory.host_graph == initial_memory.host_graph && memory.device_graph == initial_memory.device_graph,
                          "Vulkan capacity-primed buffers do not grow");
                }
                check(cpu->step(t, f.input(int(t)), experts(), control_route, error), error);
                close(layer->last_residual(), cpu->last_residual(), 0.01f, "CPU/Vulkan complete layer residual");
                close(route.weights, control_route.weights, 0.003f, "CPU/Vulkan route weights");
                check(route.ids == control_route.ids, "CPU/Vulkan route IDs");
                peak_graph_bytes = std::max(peak_graph_bytes, layer->graph_bytes());
                for (const auto &p : layer->placements())
                    if (!p.metadata_only && p.phase != "host")
                        check(p.backend == ggml_backend_name(device.get()) && p.fallback_reason.empty(),
                              "strict actual Vulkan layer placement: " + p.phase + " " + p.operation + " " + p.tensor +
                                  " backend=" + p.backend + " fallback=" + p.fallback_reason);
            }
            size_t mat = 0, rope = 0;
            for (const auto &p : layer->placements())
                if (p.phase == "qsa-project" && !p.metadata_only) {
                    if (p.operation == "MUL_MAT")
                        ++mat;
                    if (p.operation == "ROPE")
                        ++rope;
                }
            check(mat == 8 && rope == 3, "QSA projection coverage");
            check(!layer->step(9, f.input(9), experts(), route, error) && layer->tokens() == 9,
                  "Vulkan capacity rejection retains committed state");
            check(layer->reset(error), error);
            check(layer->step(0, f.input(0), experts(), route, error, &trace), error);
            auto first = layer->last_residual();
            check(layer->reset(error), error);
            for (size_t t = 0; t < 9; ++t)
                check(layer->step(t, f.input(int(t + 9)), experts(), route, error), error);
            check(layer->reset(error) && layer->step(0, f.input(0), experts(), route, error, &trace), error);
            check(layer->last_residual() == first, "Vulkan A/B/A replay");
            check(!layer->step(9, f.input(9), experts(), route, error), "Vulkan noncontiguous rejection");
            Oracle reset;
            reset.check_step(f, 0, f.input(0), trace, *layer);
            spare_high_score_test(options.execution);
            selection_tests(options.execution, reuse);
            std::printf("PASS strict Vulkan selection width2051 boundaries/ties device=%s\n", description);
            std::printf("PASS strict Vulkan QSA H2560 layer3 positions0-8 projection_MUL_MAT=%zu ROPE=%zu, scalar "
                        "checks/CPU parity/rollback/reset device=%s weights=%zu graphs_with_workspace=%zu state=%zu reuse=%d\n",
                        mat, rope, description, layer->weight_bytes(), peak_graph_bytes, layer->state_bytes(), int(reuse));
        } // validate streamed and resident dense weights
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL strict Vulkan QSA: %s\n", e.what());
        return 1;
    }
}
