#include "ggml-cpu.h"
#include "ggml-vulkan.h"
#include "portable_gdn_fixture.hpp"
#include <cstdlib>
using namespace strata::core;
using namespace gdn_fixture;
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
        check(expected.empty() || std::string(description).find(expected) != std::string::npos, "unexpected device");
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> device(ggml_backend_vk_init(size_t(index)),
                                                                           ggml_backend_free);
        check(bool(device) && !ggml_backend_supports_buft(device.get(), ggml_backend_cpu_buffer_type()),
              "distinct Vulkan buffers required");
        Fixture f;
        PortableGdnOptions options;
        options.execution.accelerator = device.get();
        options.execution.request_accelerator = true;
        std::string error;
        auto layer = PortableGdnLayer::create(0, f.views, options, error);
        check(bool(layer), error);
        auto cpu = PortableGdnLayer::create(0, f.views, {}, error);
        check(bool(cpu), error);
        size_t gdn = 0, conv = 0, mat = 0;
        for (const auto &p : layer->placements())
            if (p.phase == "gdn-attention" && !p.metadata_only) {
                check(p.backend == ggml_backend_name(device.get()) && p.fallback_reason.empty(),
                      "strict attention placement");
                if (p.operation == "GATED_DELTA_NET")
                    ++gdn;
                if (p.operation == "SSM_CONV")
                    ++conv;
                if (p.operation == "MUL_MAT")
                    ++mat;
            }
        check(gdn == 1 && conv == 1 && mat == 8, "actual attention operation coverage");
        PortableRoute route;
        PortableGdnTrace trace;
        for (int t = 0; t < 2; ++t) {
            auto state = layer->recurrent_state(), history = layer->convolution_history();
            PortableRoute cpu_route;
            check(cpu->step(f.input(t), experts(), cpu_route, error), error);
            check(layer->step(f.input(t), experts(), route, error, &trace), error);
            reference(f, state, history, trace, *layer);
            close(layer->last_residual(), cpu->last_residual(), 0.003f, "full layer CPU/Vulkan output");
            close(route.weights, cpu_route.weights, 0.003f, "full layer route weights");
            check(route.ids == cpu_route.ids, "full layer route IDs");
        }
        auto state = layer->recurrent_state(), history = layer->convolution_history(), last = layer->last_residual();
        check(!layer->step(
                  f.input(2), [](int64_t, const PortableRoute &, std::vector<float> &, std::string &) { return false; },
                  route, error),
              "Vulkan callback rejection");
        close(layer->recurrent_state(), state, 0, "Vulkan state rollback");
        close(layer->convolution_history(), history, 0, "Vulkan history rollback");
        close(layer->last_residual(), last, 0, "Vulkan residual rollback");
        check(layer->tokens() == 2 && layer->reset(error), "Vulkan reset");
        state = layer->recurrent_state();
        history = layer->convolution_history();
        check(layer->step(f.input(0), experts(), route, error, &trace), error);
        reference(f, state, history, trace, *layer);
        std::printf("portable_gdn_vulkan_test: device=%s actual H2560/S128/Hk16/Hv48; backend=%s GATED_DELTA_NET=%zu "
                    "SSM_CONV=%zu attention_MUL_MAT=%zu strict execution, independent state, full layer CPU parity, "
                    "reset and rollback passed\n",
                    description, ggml_backend_name(device.get()), gdn, conv, mat);
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
