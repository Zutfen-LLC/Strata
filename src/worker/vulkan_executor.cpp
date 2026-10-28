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
    uint64_t tensor_budget_;
    Backend owned_backend_{nullptr,ggml_backend_free};
    ggml_backend_t borrowed_ = nullptr, backend_ = nullptr;
    std::vector<std::unique_ptr<ExpertGraph>> graphs_;
    std::vector<uint8_t> binding_;
    GpuPlan plan_;
public:
    VulkanExecutor(int device, bool cpu_test, uint64_t budget):device_(device),cpu_test_(cpu_test),tensor_budget_(budget) {}
    VulkanExecutor(ggml_backend_t backend, uint64_t budget):device_(-1),cpu_test_(false),tensor_budget_(budget),borrowed_(backend) {}
    ~VulkanExecutor() override { GpuCounters ignored; release(ignored); }
    bool prepare(const GpuPlan& p,uint64_t reserve,GpuCache& cache,GpuCounters& c,std::string& err) override {
        return prepare_impl(p,reserve, [&](const GpuAsset& a, std::vector<uint8_t>& blob, std::string& error) {
            return cache.read(a, blob, error);
        }, c, err);
    }
    bool prepare_source(const GpuPlan& p,const GpuAssetReader& read,GpuCounters& c,std::string& err) override {
        return prepare_impl(p,0,read,c,err);
    }
    bool prepare_impl(const GpuPlan& p,uint64_t reserve,const GpuAssetReader& read,GpuCounters& c,std::string& err) {
        release(c); err.clear();
        try {
            require(validate_gpu_plan(p,err),err);
            require(p.n_embd == H && p.n_ff == F && p.gu_type == 8 && p.d_type == 8,
                    "Vulkan adapter requires Q8_0 experts at H2560/FF640");
            // ggml's backend owns internal compute buffers. This slice cannot promise a VRAM reserve.
            require(reserve == 0,"Vulkan adapter does not support a runtime VRAM reserve");
            require(tensor_budget_ > 0 && tensor_budget_ <= SIZE_MAX && p.assigned_bytes() <= tensor_budget_,
                    "Vulkan resident weights exceed tensor budget");
            require(encode_gpu_plan(p,binding_,err),err);
            std::vector<uint8_t> blob;
            require(read(p.assets[0],blob,err),err); // rehashes the frozen assigned blob
            c.host_staging_peak=std::max(c.host_staging_peak,uint64_t(blob.capacity()));
#ifdef STRATA_AMD_EXECUTOR_HOST_TEST
            if(cpu_test_) {
                owned_backend_.reset(ggml_backend_cpu_init());
                require(bool(owned_backend_),"CPU test backend unavailable");
                backend_ = owned_backend_.get();
                ggml_backend_cpu_set_n_threads(backend_,1);
            }
#endif
#ifdef STRATA_AMD_PROBE_VULKAN
            if(!cpu_test_ && !borrowed_) {
                require(device_ >= 0 && device_ < ggml_backend_vk_get_device_count(),"requested Vulkan device absent");
                owned_backend_.reset(ggml_backend_vk_init(size_t(device_)));
                backend_ = owned_backend_.get();
            }
#endif
            if (borrowed_) {
#ifdef STRATA_AMD_PROBE_VULKAN
                require(ggml_backend_is_vk(borrowed_), "local expert backend must be Vulkan");
                backend_ = borrowed_;
#else
                throw std::runtime_error("Vulkan unavailable; no CPU fallback");
#endif
            }
            require(bool(backend_),"Vulkan unavailable; no CPU fallback");
            cpu::NativeFmt fmt;
            require(cpu::native_fmt(8,8,H,F,fmt,err),err);
            graphs_.reserve(p.assets.size());
            for(size_t i=0;i<p.assets.size();++i) {
                if(i) require(read(p.assets[i],blob,err),err);
                c.host_staging_peak=std::max(c.host_staging_peak,uint64_t(blob.capacity()));
                require(blob.size() == p.assets[i].bytes && blob_digest(blob) == p.assets[i].digest,
                        "Vulkan local asset size/digest mismatch");
                graphs_.push_back(std::make_unique<ExpertGraph>(backend_,fmt,blob,true,
                    size_t(tensor_budget_-c.gpu_allocated_bytes)));
                c.gpu_allocated_bytes+=graphs_.back()->buffer_bytes();
                c.gpu_weight_bytes+=blob.size();
                c.gpu_scratch_bytes=c.gpu_allocated_bytes-c.gpu_weight_bytes;
                c.gpu_allocation_count=graphs_.size();
                c.gpu_peak_bytes=std::max(c.gpu_peak_bytes,c.gpu_allocated_bytes);
            }
            plan_=p;
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
            require(!graphs_.empty(),"Vulkan adapter is not prepared");
            std::vector<uint8_t> binding;
            require(encode_gpu_plan(p,binding,err),err);
            require(binding == binding_,"Vulkan adapter plan changed after prepare");
            require(r.model_id == p.model_id && r.partition_id == p.placement_id.a && r.session && r.sequence,
                    "Vulkan adapter request binding mismatch");
            require(r.T >= 1 && r.T <= kMaxT && r.K > 0 && r.K <= kMaxK && !r.packed && r.mask.size() == size_t(r.T)*r.K &&
                    r.ids.size() == r.mask.size() && r.weights.size() == r.mask.size() && r.activations.size() == size_t(r.T)*H,
                    "Vulkan adapter requires bounded T1..8 F32 input and indexed entries");
            // Validate the entire window before compute; one resident single-row graph per entry.
            finite(r.activations);
            // Slots follow the frozen plan; returned indices stay in the original T*K space.
            std::vector<std::pair<uint16_t,size_t>> entries;
            for(size_t i=0;i<r.mask.size();++i) {
                require(r.mask[i] <= 1 && std::isfinite(r.weights[i]),"invalid Vulkan routing field");
                if(r.mask[i]) {
                    const auto* asset=plan_.find(r.layer,r.ids[i]);
                    require(asset != nullptr,"Vulkan adapter request contains unassigned expert");
                    const size_t slot=size_t(asset-plan_.assets.data());
                    require(slot < graphs_.size(),"Vulkan resident slot missing");
                    entries.emplace_back(uint16_t(i),slot);
                }
            }
            require(!entries.empty(),"Vulkan adapter requires active entries");
            EntryRowsReply result;
            result.session=r.session; result.sequence=r.sequence;
            result.rows.reserve(entries.size()*H);
            for(const auto& entry:entries) {
                const auto begin=r.activations.begin()+(entry.first/r.K)*H;
                std::vector<float> token(begin,begin+H);
                auto rows=graphs_[entry.second]->run(token);
                result.indices.push_back(entry.first);
                result.rows.insert(result.rows.end(),rows.out.begin(),rows.out.end());
            }
            out=std::move(result);
            return true;
        } catch(const std::exception& e) { err=e.what(); return false; }
    }
    void release(GpuCounters& c) override {
        graphs_.clear(); owned_backend_.reset(); backend_ = nullptr; binding_.clear(); plan_={};
        c.gpu_allocated_bytes=c.gpu_weight_bytes=c.gpu_scratch_bytes=c.gpu_allocation_count=0;
    }
};
} // namespace
std::unique_ptr<GpuExecutor> create_vulkan_executor_on_backend(ggml_backend_t backend, uint64_t budget) {
    return std::make_unique<VulkanExecutor>(backend,budget);
}
std::unique_ptr<GpuExecutor> create_vulkan_gpu_executor(int device, uint64_t tensor_budget) {
    return std::make_unique<VulkanExecutor>(device,false,tensor_budget);
}
#ifdef STRATA_AMD_EXECUTOR_HOST_TEST
std::unique_ptr<GpuExecutor> create_vulkan_executor_cpu_test(uint64_t tensor_budget) {
    return std::make_unique<VulkanExecutor>(-1,true,tensor_budget);
}
#endif
} // namespace strata::worker
