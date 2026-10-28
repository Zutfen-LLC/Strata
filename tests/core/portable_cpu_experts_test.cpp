#include "../../src/core/portable_cpu_experts.hpp"
#include "portable_model_fixture.hpp"
#include <cstdio>
using namespace strata::core;
using namespace model_fixture;
int main() {
    try {
        SparseModel fixture;
        strata::GgufModel model(fixture.paths); // owns every borrowed native view
        PortableModelOptions opts;
        portable_detail::CpuExperts candidate(model, (fixture.root / "manifest").string(), opts);
        opts.expert_dispatch_oracle = true;
        portable_detail::CpuExperts oracle(model, (fixture.root / "manifest").string(), opts);
        PortableRoute route;
        route.activation.resize(2560);
        for (size_t i = 0; i < route.activation.size(); ++i)
            route.activation[i] = float(int(i % 17) - 8) / 7;
        for (int i = 0; i < 10; ++i)
            route.ids.push_back(i);
        std::string error;
        std::vector<float> got, expected;
        auto compare = [&](int layer) {
            check(oracle.run(layer, route, expected, error), error);
            check(candidate.run(layer, route, got, error), error);
            check(got.size() == 25600 &&
                      std::memcmp(got.data(), expected.data(), got.size() * sizeof(float)) == 0,
                  "native unweighted rows byte identical in route order");
        };
        compare(0);
        const auto cold = candidate.work();
        compare(0);
        check(candidate.work().hits == 10 && candidate.work().copied_bytes == cold.copied_bytes &&
                  candidate.work().digest_bytes == cold.digest_bytes,
              "repeat removes all copy/digest work");
        std::reverse(route.ids.begin(), route.ids.end());
        route.ids[0] = 11;
        compare(0);
        check(candidate.work().verifications == 11, "new expert misses; reordered experts hit");
        compare(1);
        check(candidate.work().verifications == 21, "layer isolation");
        portable_detail::CpuExperts independent(model, (fixture.root / "manifest").string(), opts);
        check(independent.work().verifications == 0, "new provider has no shared verification state");
        strata::GgufModel second_model(fixture.paths);
        opts.expert_dispatch_oracle = false;
        portable_detail::CpuExperts isolated(second_model, (fixture.root / "manifest").string(), opts);
        check(isolated.run(0, route, got, error) && isolated.work().verifications == 10,
              "distinct model owner verifies independently");
        const auto path = fixture.paths[3];
        const auto time = std::filesystem::last_write_time(path);
        const uint8_t corrupt = 1, restored = 0;
        fixture.patch(3, "blk.1.ffn_gate_exps.weight", 0, &corrupt, 1);
        // Force a metadata delta without relying on filesystem clock resolution.
        std::filesystem::last_write_time(path, time + std::chrono::seconds(2));
        route.ids = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        check(!candidate.run(1, route, got, error) && got.empty() &&
                  error.find("digest mismatch") != std::string::npos,
              "changed verified source rejects corruption");
        check(candidate.work().source_invalidations == 1, "metadata change invalidates verification");
        check(!candidate.run(1, route, got, error) && got.empty(), "failed verification never becomes a hit");
        check(!isolated.run(1, route, got, error), "corruption rejected on first use in another model");
        fixture.patch(3, "blk.1.ffn_gate_exps.weight", 0, &restored, 1);
        std::filesystem::last_write_time(path, time + std::chrono::seconds(4));
        compare(1);
        route.ids[1] = route.ids[0];
        check(!candidate.run(0, route, got, error) && got.empty(), "invalid request clears rows");
        // Models retain mappings independently of caller-owned path strings.
        fixture.paths.clear();
        route.ids[1] = 1;
        compare(0);
        const auto work = candidate.work();
        std::printf("portable_cpu_experts_test passed: hits=%llu "
                    "verifications=%llu copied=%llu digest=%llu native=%llu\n",
                    (unsigned long long)work.hits, (unsigned long long)work.verifications,
                    (unsigned long long)work.copied_bytes, (unsigned long long)work.digest_bytes,
                    (unsigned long long)work.native_batches);
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
