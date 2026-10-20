#include "strata/worker/local_source.hpp"
#include <chrono>
#include <csignal>
#include <cstdio>
#include <filesystem>
namespace {
volatile std::sig_atomic_t stopped = 0;
void stop(int) { stopped = 1; }
} // namespace
int main(int argc, char **argv) {
    if (argc != 8) {
        std::fprintf(
            stderr,
            "usage: strata-worker-manifest PACK SHARD1 OUTPUT LAYERS EXPERTS TIMEOUT_SECONDS --prepare\n");
        return 2;
    }
    try {
        auto number = [](const char *s) {
            size_t n = 0;
            auto v = std::stoul(s, &n);
            if (n != std::strlen(s) || !v || v > 86400)
                throw std::runtime_error("invalid positive integer");
            return v;
        };
        const auto nl = number(argv[4]), ne = number(argv[5]), seconds = number(argv[6]);
        if (nl > strata::worker::kMaxLayers || ne > strata::worker::kMaxExperts ||
            std::string(argv[7]) != "--prepare" || std::filesystem::exists(argv[3])) {
            std::fprintf(stderr,
                         "invalid geometry/mode or output already exists; preserve reusable manifests\n");
            return 2;
        }
        std::signal(SIGINT, stop);
        std::signal(SIGTERM, stop);
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
        auto cancelled = [&] { return stopped || std::chrono::steady_clock::now() >= until; };
        strata::worker::LocalManifest m;
        std::string err;
        if (!strata::worker::prepare_local_manifest(argv[1], argv[2], nl, ne, m, err, cancelled)) {
            std::fprintf(stderr, "worker-manifest: %s\n", err.c_str());
            return 1;
        }
        const std::string tmp = std::string(argv[3]) + ".tmp";
        if (cancelled() || !strata::worker::write_local_manifest(tmp, m, err)) {
            std::filesystem::remove(tmp);
            std::fprintf(stderr, "worker-manifest: cancelled or %s\n", err.c_str());
            return 1;
        }
        std::filesystem::rename(tmp, argv[3]);
        std::fprintf(stderr, "worker-manifest: complete; pin with sha256sum and transfer this manifest "
                             "through the controller\n");
    } catch (const std::exception &e) {
        std::fprintf(stderr, "worker-manifest: %s\n", e.what());
        return 1;
    }
}
