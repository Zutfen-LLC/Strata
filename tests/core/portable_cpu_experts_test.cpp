#include "../../src/core/portable_cpu_experts.hpp"
#include "portable_model_fixture.hpp"
#include <cstdio>
#include <limits>
using namespace strata::core;
using namespace model_fixture;
int main() {
    try {
        const std::vector<float> finite_values{0.f, -0.f, std::numeric_limits<float>::min(),
                                               std::numeric_limits<float>::denorm_min(),
                                               std::numeric_limits<float>::max()};
        portable_detail::finite({}, "expert rows");
        portable_detail::finite(finite_values, "expert rows");
        for (float invalid : {std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()})
            for (size_t at : {size_t(0), finite_values.size()/2, finite_values.size()-1}) {
                auto values=finite_values;values[at]=invalid;
                bool refused=false;
                try { portable_detail::finite(values,"expert rows"); }
                catch (const std::runtime_error& e) {
                    refused=true;
                    check(std::string(e.what())=="portable model: nonfinite expert rows", "exact finite diagnostic");
                }
                check(refused,"every nonfinite position rejected");
            }
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
        const double PortableExpertWork::*timers[] = {
            &PortableExpertWork::pool_gu_ms, &PortableExpertWork::pool_quant_ms,
            &PortableExpertWork::pool_down_ms, &PortableExpertWork::pool_elapsed_ms};
        const auto fresh = candidate.work();
        for (auto timer : timers) check(fresh.*timer == 0, "fresh pool timer");
        compare(0);
        const auto cold = candidate.work();
        compare(0);
        check(candidate.work().hits == 10 && candidate.work().copied_bytes == cold.copied_bytes &&
                  candidate.work().digest_bytes == cold.digest_bytes,
              "repeat removes all copy/digest work");
        const auto warm = candidate.work();
        for (auto timer : timers) {
            check(std::isfinite(warm.*timer) && warm.*timer >= cold.*timer && cold.*timer >= 0,
                  "finite monotone pool timers");
            check(candidate.work().*timer == warm.*timer, "snapshot must not accumulate timers twice");
            const double delta1 = cold.*timer - fresh.*timer, delta2 = warm.*timer - cold.*timer;
            check(std::fabs((delta1 + delta2) - (warm.*timer - fresh.*timer)) < 1e-8,
                  "successive request deltas partition lifetime timer");
        }
        check(warm.pool_elapsed_ms > 0 && warm.pool_gu_ms > 0 && warm.pool_down_ms > 0,
              "native calls expose existing wall intervals");
        check(warm.pool_gu_ms + warm.pool_quant_ms + warm.pool_down_ms <= warm.pool_elapsed_ms + 1e-6,
              "native elapsed encloses phases; not an additional phase");
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
        const auto good_activation=route.activation;
        const auto before_failure=candidate.work();
        for (float invalid : {std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()})
            for (size_t at : {size_t(0), route.activation.size()/2, route.activation.size()-1}) {
                route.activation=good_activation;route.activation[at]=invalid;got={123.f};
                check(!candidate.run(0,route,got,error) && got.empty() &&
                      error=="portable model: nonfinite expert activation", "CPU provider finite failure clears rows");
                const auto failed=candidate.work();
                check(failed.native_batches==before_failure.native_batches &&
                      failed.verifications==before_failure.verifications && failed.hits==before_failure.hits,
                      "invalid activation stops before expert verification or pool dispatch");
            }
        route.activation=good_activation;
        for (int32_t invalid_id : {-1, 512})
            for (size_t at : {size_t(0), route.ids.size()/2, route.ids.size()-1}) {
                auto invalid_route=route;invalid_route.ids[at]=invalid_id;got={123.f};
                check(!candidate.run(0,invalid_route,got,error) && got.empty() &&
                      error=="portable model: expert ID duplicate/out of range", "CPU provider ID diagnostic");
                check(candidate.work().native_batches==before_failure.native_batches,
                      "invalid ID stops before pool dispatch");
            }
        compare(0); // failed validation cannot leave stale provider rows or poison later valid dispatch
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
        check(!candidate.run(0, route, got, error) && got.empty() &&
              error=="portable model: expert ID duplicate/out of range", "invalid request clears rows and retains diagnostic");
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
