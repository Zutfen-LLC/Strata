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
        // The actual model expert-admission seam must identify every operand
        // on refusal, before any expert allocation or ledger mutation.
        auto refusal_report=fresh();
        refusal_report.device_weights={10,20};refusal_report.device_state={2,3};
        refusal_report.device_scheduler={5,7};refusal_report.device_workspace_bound={11,13};
        refusal_report.device_token_temporary_bound={17,19};refusal_report.device_shared_conversion_bound={23,29};
        refusal_report.device_admission={68,91};
        for(size_t die=0;die<2;++die) {
            const auto snapshot=refusal_report.device_admission;
            const size_t used=snapshot[die];
            const D::DeviceAdmissionContext context{die,48,"resident-experts",1000,512};
            D::preflight_resident_expert_memory(refusal_report,die,used+40,32,8,0,0,&context);
            std::string error;
            try { D::preflight_resident_expert_memory(refusal_report,die,used+39,32,8,800,1200,&context); }
            catch(const std::runtime_error& e) { error=e.what(); }
            for(const auto& token:std::vector<std::string>{"slot="+std::to_string(die),
                "dense_admitted_bytes="+std::to_string(used),"expert_tensor_cap_bytes=32","expert_workspace_cap_bytes=8",
                "effective_limit_bytes="+std::to_string(used+39),"remaining_bytes=39","expert_budget_shortfall_bytes=1",
                "policy_cap_bytes=1000","reserve_bytes=512","available_after_dense_bytes=800","total_bytes=1200",
                "dense_weight_bytes="+std::to_string(refusal_report.device_weights[die]),
                "dense_state_bytes="+std::to_string(refusal_report.device_state[die]),
                "dense_scheduler_bytes="+std::to_string(refusal_report.device_scheduler[die]),
                "dense_private_bound_bytes="+std::to_string(refusal_report.device_workspace_bound[die]),
                "dense_token_temporary_bound_bytes="+std::to_string(refusal_report.device_token_temporary_bound[die]),
                "dense_shared_conversion_bound_bytes="+std::to_string(refusal_report.device_shared_conversion_bound[die])})
                check(error.find(token)!=std::string::npos,"resident refusal diagnostic omitted operand");
            check(refusal_report.device_admission==snapshot,"resident refusal mutated ledger");
        }
        bool overflow_refused=false;
        try { D::preflight_resident_expert_memory(refusal_report,0,SIZE_MAX,SIZE_MAX,1,SIZE_MAX,SIZE_MAX); }
        catch(const std::runtime_error& e) { overflow_refused=std::string(e.what())=="invalid resident expert budgets"; }
        check(overflow_refused,"resident tensor/runtime sum overflow accepted");
        bool over_cap_refused=false;
        try { D::preflight_resident_expert_memory(refusal_report,0,67,32,8,SIZE_MAX,SIZE_MAX); }
        catch(const std::runtime_error& e) { over_cap_refused=std::string(e.what()).find("dense_over_cap_bytes=1")!=std::string::npos; }
        check(over_cap_refused,"free-memory observation credited an over-cap ledger");
        options.accelerators.resize(1); rejects(options);
        options.accelerators.clear(); rejects(options);
        options.dense_topology = static_cast<PortableDenseTopology>(99); rejects(options);
        std::puts("portable_topology_test: default splits, native primary/head/state ownership and separate cap boundaries passed");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
