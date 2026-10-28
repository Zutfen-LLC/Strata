#include "portable_ffn_fixture.hpp"
#include "ggml-vulkan.h"
#include "ggml-cpu.h"
#include <cstdlib>

int main(int argc, char** argv) {
    try {
        int index = 0;
        std::string expected_device;
        for (int i = 1; i < argc; i += 2) {
            check(i + 1 < argc, "usage: --device INDEX --expect-device SUBSTRING");
            if (std::string(argv[i]) == "--device") {
                char* end = nullptr;
                const long value = std::strtol(argv[i+1], &end, 10);
                check(end != argv[i+1] && *end == 0 && value >= 0 && value < GGML_VK_MAX_DEVICES, "invalid device index");
                index = int(value);
            } else if (std::string(argv[i]) == "--expect-device") expected_device = argv[i+1];
            else check(false, "unknown argument");
        }
        check(ggml_backend_vk_get_device_count() > index, "Vulkan device required; not a skipped pass");
        char description[256] = {};
        ggml_backend_vk_get_device_description(index, description, sizeof(description));
        check(expected_device.empty() || std::string(description).find(expected_device) != std::string::npos,
              "unexpected Vulkan device: " + std::string(description));
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> device(ggml_backend_vk_init(size_t(index)), ggml_backend_free);
        check(bool(device), "Vulkan backend initialization failed");
        check(!ggml_backend_supports_buft(device.get(), ggml_backend_cpu_buffer_type()),
              "regression requires a distinct device buffer type");
        for (bool resident : {false, true}) {
            std::printf("dense weight fixture: resident=%d\n", int(resident));
            Fixture f;
            PortableFfnOptions options;
            options.resident_dense_weights = resident;
            options.request_accelerator = true;
            options.allow_cpu_fallback = true;
            options.accelerator = device.get();
            std::string error;
            auto runner = PortableFfn::create(f.g, 0, f.views, options, error);
            check(bool(runner), error);
            check((runner->memory_usage().device_weights > 0) == resident, "FFN resident ledger");
            check(runner->memory_usage().host_weights + runner->memory_usage().device_weights == runner->weight_bytes(),
                  "FFN actual weight reconciliation");
            size_t matmuls = 0, post = 0, fallback = 0;
            const std::string backend = ggml_backend_name(device.get());
            for (const auto& p : runner->placements()) {
                if (!p.fallback_reason.empty()) ++fallback;
                if (p.operation == "MUL_MAT") {
                    check(p.backend == backend, "supported MUL_MAT stayed on CPU: " + p.tensor);
                    ++matmuls;
                }
                if (p.phase == "post-experts" && (p.operation == "ADD" || p.operation == "MUL")) {
                    check(p.backend == backend, "HC post computation stayed on CPU");
                    ++post;
                }
            }
            check(matmuls > 0 && post > 0, "no actual matrix/HC post device assignments");
            std::vector<float> initial(128);
            for (size_t i = 0; i < initial.size(); ++i) initial[i] = 0.2f + float(int(i % 19)-9) * 0.06f;
            const PortableExpertRows experts = [](int64_t, const PortableRoute& r, std::vector<float>& rows, std::string&) {
                rows.resize(64);
                for (size_t k = 0; k < 2; ++k)
                    for (size_t h = 0; h < 32; ++h) rows[k*32+h] = float(r.ids[k]+1) * (0.03f + float(h)*0.001f);
                return true;
            };
            PortableRoute route, reference;
            auto expected = f.oracle(initial, reference);
            check(runner->reset(initial, error) && runner->step(experts, route, error), error);
            close(route.activation, reference.activation, 3e-4f);
            check(route.ids == reference.ids, "Vulkan route IDs");
            close(route.weights, reference.weights, 3e-4f);
            close(runner->residual(), expected, 3e-4f);
            expected = f.oracle(expected, reference);
            check(runner->step(experts, route, error), error);
            close(runner->residual(), expected, 5e-4f);
            check(runner->reset(initial, error) && runner->step(experts, route, error), error);
            close(runner->residual(), f.oracle(initial, reference), 3e-4f);
            const auto saved = runner->residual();
            check(!runner->step([](int64_t, const PortableRoute&, std::vector<float>&, std::string&) { return false; }, route, error), "Vulkan callback failure");
            close(runner->residual(), saved, 0);
            options.allow_cpu_fallback = false;
            auto strict = PortableFfn::create(f.g, 0, f.views, options, error);
            check(bool(strict), "supported strict Vulkan graph rejected: " + error);
            check(strict->reset(initial, error) && strict->step(experts, route, error), error);
            close(strict->residual(), f.oracle(initial, reference), 3e-4f);
            // Match the fleet's encoding mix: six Q8_0 projection tensors plus
            // F32 norm/router/shared scalar gate, with a CPU graph as the oracle.
            std::vector<std::vector<uint8_t>> q8(9);
            auto quantized = f.views;
            for (size_t i : {size_t(1),size_t(2),size_t(3),size_t(6),size_t(7),size_t(8)}) {
                q8[i].resize(f.data[i].size()/32*34);
                ggml_quantize_chunk(GGML_TYPE_Q8_0, f.data[i].data(), q8[i].data(), 0,
                                    int64_t(f.data[i].size()/size_t(f.views[i].shape[0])), f.views[i].shape[0], nullptr);
                quantized[i].type = 8; quantized[i].bytes = q8[i].size(); quantized[i].data = q8[i].data();
            }
            auto qcpu = PortableFfn::create(f.g, 0, quantized, {}, error);
            check(bool(qcpu), error);
            auto qdevice = PortableFfn::create(f.g, 0, quantized, options, error);
            check(bool(qdevice), error);
            size_t q8_matmuls = 0;
            for (const auto& p : qdevice->placements()) if (p.operation == "MUL_MAT") {
                check(p.backend == backend, "Q8_0 projection stayed on CPU");
                ++q8_matmuls;
            }
            check(q8_matmuls == matmuls, "Q8_0 matrix coverage");
            PortableRoute qroute;
            check(qcpu->reset(initial, error) && qcpu->step(experts, qroute, error), error);
            check(qdevice->reset(initial, error) && qdevice->step(experts, route, error), error);
            close(qdevice->residual(), qcpu->residual(), 0.003f);
            close(route.activation, qroute.activation, 0.003f);
            check(route.ids == qroute.ids, "Q8_0 Vulkan route IDs");
            options.graph_budget = 1;
            check(!PortableFfn::create(f.g, 0, f.views, options, error), "Vulkan graph budget rejection");
            std::printf("portable_ffn_vulkan_test: device=%s backend=%s actual_MUL_MAT=%zu actual_HC_post=%zu Q8_mix_MUL_MAT=%zu fallback=%zu graph_bytes=%zu; execution, F32/Q8_0 parity, strict mode, reset, rollback and budget passed\n",
                        description, backend.c_str(), matmuls, post, q8_matmuls, fallback, runner->graph_bytes());
        } // validate streamed and resident dense weights
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
