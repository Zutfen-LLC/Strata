#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf_fixture.hpp"
#include "strata/core/portable_model.hpp"
#ifdef STRATA_PORTABLE_MODEL_VULKAN
#include "ggml-vulkan.h"
#endif
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
using namespace strata::core;
namespace {
void check(bool ok, const std::string &error) {
    if (!ok)
        throw std::runtime_error(error);
}
using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
} // namespace
int main(int argc, char **argv) {
    try {
        Backend first(nullptr, ggml_backend_free), duplicate(nullptr, ggml_backend_free),
            other(nullptr, ggml_backend_free);
        if (argc == 1) {
            first.reset(ggml_backend_cpu_init());
            duplicate.reset(ggml_backend_cpu_init());
        } else {
            check(argc == 3 && std::string(argv[1]) == "--vulkan",
                  "usage: portable_device_admission_test [--vulkan INDEX[,OTHER]]");
#ifdef STRATA_PORTABLE_MODEL_VULKAN
            const std::string list = argv[2];
            const auto comma = list.find(',');
            auto index = [&](const std::string &value) {
                size_t used = 0;
                const auto id = std::stoul(value, &used);
                check(used == value.size() && id < size_t(ggml_backend_vk_get_device_count()), "invalid Vulkan index");
                return id;
            };
            const size_t id = index(list.substr(0, comma));
            first.reset(ggml_backend_vk_init(id));
            duplicate.reset(ggml_backend_vk_init(id));
            if (comma != std::string::npos)
                other.reset(ggml_backend_vk_init(index(list.substr(comma + 1))));
#else
            throw std::runtime_error("this build has no Vulkan backend");
#endif
        }
        check(first && duplicate && first.get() != duplicate.get(), "fixture requires distinct backend handles");
        check(ggml_backend_get_device(first.get()) == ggml_backend_get_device(duplicate.get()),
              "fixture requires same physical device identity");
        const auto root =
            std::filesystem::temp_directory_path() /
            ("strata-device-admission-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        check(std::filesystem::create_directory(root), "fixture directory");
        struct Cleanup {
            std::filesystem::path path;
            ~Cleanup() {
                std::error_code e;
                std::filesystem::remove_all(path, e);
            }
        } cleanup{root};
        std::vector<std::string> shards;
        for (size_t i = 0; i < 6; ++i) {
            const auto path = root / (std::to_string(i) + ".gguf");
            std::vector<fixture::Kv> kv{fixture::u32("split.count", 6), fixture::u32("split.no", i),
                                        fixture::u32("split.tensors.count", 0)};
            if (i == 0)
                kv.push_back(fixture::str("general.architecture", "device-admission-only"));
            fixture::write(path, kv, {});
            shards.push_back(path.string());
        }
        PortableModelOptions options;
        options.accelerators = {first.get(), duplicate.get()};
        options.device_caps = {8ull << 30, 8ull << 30};
        std::string error;
        check(!PortableModelSession::open(shards, "unused", options, error), "duplicate device accepted");
        check(error.find("duplicate physical accelerator device") != std::string::npos,
              "duplicate must fail before model/manifest/weight admission: " + error);
        options.dense_topology = PortableDenseTopology::primary;
        options.split = 0; // inactive split must not interfere with primary ownership
        check(!PortableModelSession::open(shards, "unused", options, error), "primary accepted duplicate device");
        check(error.find("duplicate physical accelerator device") != std::string::npos,
              "primary must retain physical identity guard: " + error);
        // A real two-device host can also exercise distinct physical identities.
        // The deliberately incomplete fixture then reaches the architecture guard.
        if (other) {
            check(ggml_backend_get_device(other.get()) != ggml_backend_get_device(first.get()),
                  "OTHER must be a distinct physical device");
            options.accelerators[1] = other.get();
            check(!PortableModelSession::open(shards, "unused", options, error), "incomplete fixture admitted");
            check(error == "portable model: unsupported architecture",
                  "distinct devices rejected before architecture guard: " + error);
        }
        std::printf("portable_device_admission_test: distinct handles for %s share one physical device and are "
                    "rejected; two distinct devices checked=%d\n",
                    ggml_backend_name(first.get()), bool(other));
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
