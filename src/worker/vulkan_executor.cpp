#include "strata/worker/vulkan_executor.hpp"
#include "vulkan_expert_graph.hpp"
#ifdef STRATA_AMD_PROBE_VULKAN
#include "ggml-vulkan.h"
#endif
#ifdef STRATA_AMD_EXECUTOR_HOST_TEST
#include "ggml-cpu.h"
#endif
#include <algorithm>

namespace strata::worker {
namespace {
using namespace vulkan_detail;
class VulkanExecutor final : public GpuExecutor {
    int device_;
    bool cpu_test_;
    Backend backend_{nullptr,ggml_backend_free};
    std::unique_ptr<ExpertGraph> graph_;
    std::vector<uint8_t> binding_;
    GpuPlan plan_;
public:
    VulkanExecutor(int device, bool cpu_test):device_(device),cpu_test_(cpu_test) {}
    ~VulkanExecutor() override { GpuCounters ignored; release(ignored); }
    bool prepare(const GpuPlan& p,uint64_t reserve,GpuCache& cache,GpuCounters& c,std::string& err) override {
        release(c); err.clear();
        try {
            require(validate_gpu_plan(p,err),err);
            require(p.n_embd == H && p.n_ff == F && p.gu_type == 8 && p.d_type == 8 && p.assets.size() == 1,
                    "Vulkan adapter requires one Q8_0 expert at H2560/FF640");
            // ggml's backend owns internal compute buffers. This slice cannot promise a VRAM reserve.
            require(reserve == 0,"Vulkan adapter does not support a runtime VRAM reserve");
            require(encode_gpu_plan(p,binding_,err),err);
            std::vector<uint8_t> blob;
            require(cache.read(p.assets[0],blob,err),err); // rehashes the frozen assigned blob
            c.host_staging_peak=std::max(c.host_staging_peak,uint64_t(blob.capacity()));
#ifdef STRATA_AMD_EXECUTOR_HOST_TEST
            if(cpu_test_) {
                backend_.reset(ggml_backend_cpu_init());
                require(bool(backend_),"CPU test backend unavailable");
                ggml_backend_cpu_set_n_threads(backend_.get(),1);
            }
#endif
#ifdef STRATA_AMD_PROBE_VULKAN
            if(!cpu_test_) {
                require(device_ >= 0 && device_ < ggml_backend_vk_get_device_count(),"requested Vulkan device absent");
                backend_.reset(ggml_backend_vk_init(size_t(device_)));
            }
#endif
            require(bool(backend_),"Vulkan unavailable; no CPU fallback");
            cpu::NativeFmt fmt;
            require(cpu::native_fmt(8,8,H,F,fmt,err),err);
            graph_=std::make_unique<ExpertGraph>(backend_.get(),fmt,blob,true);
            plan_=p;
            c.gpu_allocated_bytes=graph_->buffer_bytes();
            c.gpu_weight_bytes=blob.size();
            c.gpu_scratch_bytes=c.gpu_allocated_bytes-c.gpu_weight_bytes;
            c.gpu_allocation_count=1;
            c.gpu_peak_bytes=std::max(c.gpu_peak_bytes,c.gpu_allocated_bytes);
            return true;
        } catch(const std::exception& e) { err=e.what(); release(c); return false; }
    }
    bool supports_rows() const override { return true; }
    bool infer(const GpuPlan&,const InferRequest&,InferReply&,std::string& err) override {
        err="Vulkan adapter supports unweighted indexed rows only"; return false;
    }
    bool infer_rows(const GpuPlan& p,const InferRequest& r,EntryRowsReply& out,std::string& err) override {
        out={}; err.clear();
        try {
            require(bool(graph_),"Vulkan adapter is not prepared");
            std::vector<uint8_t> binding;
            require(encode_gpu_plan(p,binding,err),err);
            require(binding == binding_,"Vulkan adapter plan changed after prepare");
            require(r.model_id == p.model_id && r.partition_id == p.placement_id.a && r.session && r.sequence,
                    "Vulkan adapter request binding mismatch");
            require(r.T == 1 && r.K > 0 && r.K <= kMaxK && !r.packed && r.mask.size() == r.K &&
                    r.ids.size() == r.K && r.weights.size() == r.K && r.activations.size() == H,
                    "Vulkan adapter requires T1, bounded F32 input and one indexed entry");
            size_t entry=r.K;
            for(size_t i=0;i<r.K;++i) {
                require(r.mask[i] <= 1 && std::isfinite(r.weights[i]),"invalid Vulkan routing field");
                if(r.mask[i]) {
                    require(entry == r.K,"Vulkan adapter supports exactly one active entry");
                    require(r.layer == plan_.assets[0].layer && r.ids[i] == plan_.assets[0].expert,
                            "Vulkan adapter request contains unassigned expert");
                    entry=i;
                }
            }
            require(entry != r.K,"Vulkan adapter requires one active entry");
            auto rows=graph_->run(r.activations);
            out.session=r.session; out.sequence=r.sequence;
            out.indices={uint16_t(entry)}; out.rows=std::move(rows.out);
            return true;
        } catch(const std::exception& e) { err=e.what(); return false; }
    }
    void release(GpuCounters& c) override {
        graph_.reset(); backend_.reset(); binding_.clear(); plan_={};
        c.gpu_allocated_bytes=c.gpu_weight_bytes=c.gpu_scratch_bytes=c.gpu_allocation_count=0;
    }
};
} // namespace
std::unique_ptr<GpuExecutor> create_vulkan_gpu_executor(int device) {
    return std::make_unique<VulkanExecutor>(device,false);
}
#ifdef STRATA_AMD_EXECUTOR_HOST_TEST
std::unique_ptr<GpuExecutor> create_vulkan_executor_cpu_test() {
    return std::make_unique<VulkanExecutor>(-1,true);
}
#endif
} // namespace strata::worker
