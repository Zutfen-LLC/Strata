#include "ggml-vulkan.h"
#include "portable_ple_fixture.hpp"
#include <cstdlib>
using namespace ple_fixture;
int main(int argc, char **argv) {
    try {
        int index = 0;
        std::string expected;
        for (int i = 1; i < argc; i += 2) {
            check(i + 1 < argc, "usage: --device INDEX --expect-device SUBSTRING");
            if (std::string(argv[i]) == "--device") {
                char *end = nullptr;
                const long n = std::strtol(argv[i + 1], &end, 10);
                check(end != argv[i + 1] && *end == 0 && n >= 0 && n < GGML_VK_MAX_DEVICES, "bad device");
                index = int(n);
            } else if (std::string(argv[i]) == "--expect-device")
                expected = argv[i + 1];
            else
                check(false, "unknown option");
        }
        check(ggml_backend_vk_get_device_count() > index, "Vulkan device required");
        char description[256] = {};
        ggml_backend_vk_get_device_description(index, description, sizeof(description));
        check(expected.empty() || std::string(description).find(expected) != std::string::npos, "unexpected device");
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> device(ggml_backend_vk_init(size_t(index)),
                                                                           ggml_backend_free);
        check(bool(device) && !ggml_backend_supports_buft(device.get(), ggml_backend_cpu_buffer_type()),
              "distinct Vulkan buffers");
        for (int mode : {0, 1, 2}) {
            const bool resident = mode != 0;
            std::printf("dense weight fixture: mode=%d (default=0 GDN-only=1 dense=2)\n", mode);
            std::string error;
            Fixture f;
            PortableGdnOptions options;
            options.resident_attention_weights = mode == 1;
            options.execution.resident_dense_weights = mode == 2;
            options.ple_table = small_table(error);
            check(bool(options.ple_table), error);
            auto cpu = PortableGdnLayer::create(1, f.views, options, error);
            check(bool(cpu), error);
            options.execution.accelerator = device.get();
            options.execution.request_accelerator = true;
            auto layer = PortableGdnLayer::create(1, f.views, options, error);
            check(bool(layer), error);
            check((layer->memory_usage().device_weights > 0) == resident, "resident weight ledger");
            check(layer->memory_usage().host_weights + layer->memory_usage().device_weights == layer->weight_bytes(),
                  "host/device weight allocation reconciliation");
            size_t mat = 0, gdn = 0, conv = 0;
            for (const auto &p : layer->placements())
                if (!p.metadata_only && p.phase != "host") {
                    check(p.backend == ggml_backend_name(device.get()) && p.fallback_reason.empty(),
                          "strict actual placement: " + p.phase + "/" + p.operation + " backend=" + p.backend +
                              " reason=" + p.fallback_reason);
                    if (p.phase == "gdn-attention") {
                        mat += p.operation == "MUL_MAT";
                        gdn += p.operation == "GATED_DELTA_NET";
                        conv += p.operation == "SSM_CONV";
                    }
                }
            check(mat == 10 && gdn == 1 && conv == 1, "PLE+GDN operation coverage");
            exercise(*layer, f, cpu.get());
            std::printf("portable_ple_vulkan_test: device=%s backend=%s actual H2560 "
                        "layer1 PLE+GDN+FFN; attention_MUL_MAT=%zu GATED_DELTA_NET=%zu "
                        "SSM_CONV=%zu strict execution, 10 tokens, independent scalar "
                        "PLE/GDN, CPU parity, EOS/reset/rollback passed\n",
                        description, ggml_backend_name(device.get()), mat, gdn, conv);
        } // exercise both streamed and resident weight placement
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
