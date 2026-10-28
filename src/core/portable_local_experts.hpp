#pragma once
#include "portable_cpu_experts.hpp"
#include "strata/worker/gpu_worker.hpp"
#include <functional>
namespace strata::core::portable_detail {
// Private adapter. Serial, frozen ownership; source/model/backend lifetimes cover it.
class LocalExperts {
public:
    using Factory = std::function<std::unique_ptr<worker::GpuExecutor>(ggml_backend_t,size_t)>;
    using Memory = std::function<std::pair<size_t,size_t>(ggml_backend_t)>;
    LocalExperts(const GgufModel&, CpuExperts&, const PortableModelOptions&, Factory, Memory);
    ~LocalExperts();
    bool run(int64_t, const PortableRoute&, std::vector<float>&, std::string&);
    const std::vector<PortableExpertDeviceReport>& report() const { return report_; }
    uint64_t gpu_entries() const { return gpu_entries_; }
    uint64_t cpu_entries() const { return cpu_entries_; }
private:
    struct Device {
        PortableExpertDevice options;
        worker::GpuPlan plan;
        std::unique_ptr<worker::GpuExecutor> executor;
        worker::GpuCounters counters;
        ggml_backend_t backend = nullptr;
    };
    const GgufModel& model_;
    CpuExperts& cpu_;
    Memory memory_;
    std::vector<Device> devices_;
    std::vector<PortableExpertDeviceReport> report_;
    std::vector<std::filesystem::file_time_type> times_;
    bool check_rows_ = false, timings_ = false;
    uint64_t sequence_=0, gpu_entries_=0, cpu_entries_=0;
    void sources() const;
    void guard(size_t);
};
} // namespace
