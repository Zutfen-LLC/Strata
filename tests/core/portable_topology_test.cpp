#include "../../src/core/portable_topology.hpp"
#include "../../src/core/portable_memory.hpp"
#include <cstdio>
using namespace strata::core;
namespace D = strata::core::portable_detail;
namespace {
void check(bool ok, const char* error) { if (!ok) throw std::runtime_error(error); }
void rejects(const PortableModelOptions& options) {
    bool refused = false;
    try { D::validate_dense_topology(options); } catch (const std::runtime_error&) { refused = true; }
    check(refused, "invalid topology accepted");
}
PortableModelReport fresh() {
    PortableModelReport r;
    r.device_admission = r.device_weights = r.device_scheduler = r.device_state =
        r.device_workspace_bound = r.device_token_temporary_bound = r.device_shared_conversion_bound = {0,0};
    return r;
}
}
int main() {
    try {
        PortableModelOptions options;
        check(options.dense_topology == PortableDenseTopology::split, "default topology changed");
        for (size_t count : {size_t(0), size_t(1), size_t(2)}) {
            options.accelerators.resize(count); // routing test only; no backend allocation
            for (size_t split = 1; split < 48; ++split) {
                options.split = split;
                D::validate_dense_topology(options);
                for (size_t layer = 0; layer < 48; ++layer)
                    check(D::dense_device(options, layer) == size_t(count == 2 && layer >= split),
                          "legacy ownership changed");
            }
        }
        options.split = 0; rejects(options);
        options.split = 48; rejects(options);
        options.dense_topology = PortableDenseTopology::primary;
        D::validate_dense_topology(options); // split is irrelevant to native primary ownership
        auto report = fresh();
        PortableMemoryUsage dense;
        dense.device_weights = 2; dense.device_state = 1;
        for (size_t layer = 0; layer < 48; ++layer) {
            const auto slot = D::dense_device(options, layer);
            check(slot == 0, "native dense/state ownership must stay on primary");
            D::account_device_memory(report, slot, 160, dense);
        }
        PortableMemoryUsage head; head.device_weights = 16;
        D::account_device_memory(report, D::dense_device(options, 47), 160, head);
        check(report.device_admission == std::vector<size_t>({160,0}), "head or dense leaked onto helper");
        check(report.device_state == std::vector<size_t>({48,0}), "state leaked onto helper");
        // Same cap boundary as execution: helper headroom never credits primary.
        D::preflight_device_memory(report.device_admission[0], 160, 0);
        bool refused = false;
        try { D::preflight_device_memory(report.device_admission[0], 160, 1); }
        catch (const std::runtime_error&) { refused = true; }
        check(refused, "helper headroom credited primary admission");
        PortableMemoryUsage experts; experts.device_weights = 40;
        D::account_device_memory(report, 1, 40, experts);
        check(report.device_admission == std::vector<size_t>({160,40}), "expert helper budget not independent");
        options.accelerators.resize(1); rejects(options);
        options.accelerators.clear(); rejects(options);
        options.dense_topology = static_cast<PortableDenseTopology>(99); rejects(options);
        std::puts("portable_topology_test: default splits, native primary/head/state ownership and separate cap boundaries passed");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
