#include "portable_local_experts.hpp"
#include <cmath>
#include <set>
namespace strata::core::portable_detail {
void LocalExperts::sources() const {
    for (size_t s=0;s<model_.size();++s) {
        const auto& f=model_.shard(s);
        require(std::filesystem::file_size(f.path())==f.file_size() &&
                std::filesystem::last_write_time(f.path())==times_[s],
                "resident expert source changed; reopen model");
    }
}
void LocalExperts::guard(size_t i) {
    auto& r=report_[i]; const auto& d=devices_[i];
    const auto [free,total]=memory_(d.backend);
    require(free<=total && free>=d.options.free_reserve,"resident expert free reserve breached");
    r.tensor_bytes=d.counters.gpu_allocated_bytes;
    const uint64_t growth=free<r.free_before ? r.free_before-free : 0;
    r.free_after=free; r.peak_growth=std::max(r.peak_growth,growth);
    require(growth<=r.tensor_bytes+d.options.workspace_cap,
            "resident expert tensor/runtime growth budget exceeded");
}
LocalExperts::LocalExperts(const GgufModel& model,CpuExperts& cpu,const PortableModelOptions& opts,
                           Factory factory,Memory memory):model_(model),cpu_(cpu),memory_(std::move(memory)),check_rows_(opts.expert_check_rows) {
    for (size_t s=0;s<model.size();++s) times_.push_back(std::filesystem::last_write_time(model.shard(s).path()));
    std::set<std::pair<int,int>> owned;
    std::set<size_t> dies;
    // Validate every plan and ownership before any device allocation.
    for (const auto& option:opts.expert_devices) {
        require(option.accelerator<opts.accelerators.size() && dies.insert(option.accelerator).second,
                "resident expert device missing or duplicated");
        require(option.tensor_cap>0 && option.workspace_cap>0 && option.free_reserve>0 &&
                option.tensor_cap<=SIZE_MAX-option.workspace_cap,
                "invalid resident expert budgets");
        Device d; d.options=option; d.backend=opts.accelerators[option.accelerator];
        std::string error;
        require(worker::read_gpu_plan(option.plan_path,d.plan,error),error);
        std::vector<std::pair<uint16_t,uint16_t>> pairs;
        for (const auto& a:d.plan.assets) {
            require(owned.emplace(a.layer,a.expert).second,"duplicate resident expert ownership");
            pairs.emplace_back(a.layer,a.expert);
        }
        worker::GpuPlan expected;
        require(worker::make_gpu_plan(cpu.identity(),pairs,expected,error),error);
        std::vector<uint8_t> a,b;
        require(worker::encode_gpu_plan(d.plan,a,error) && worker::encode_gpu_plan(expected,b,error) && a==b,
                "resident expert plan/model identity mismatch");
        require(d.plan.assigned_bytes()<=option.tensor_cap,"resident weights exceed tensor cap");
        PortableExpertDeviceReport r;
        r.accelerator=option.accelerator; r.assigned=d.plan.assets.size();
        r.tensor_cap=option.tensor_cap; r.workspace_cap=option.workspace_cap;
        // ExpertGraph has one 2 MiB ggml metadata arena per expert; plus plan/map slack.
        require(r.assigned<=(SIZE_MAX-(4u<<20))/(2u<<20),"resident metadata bound overflow");
        r.metadata_bound=r.assigned*((2u<<20)+(32u<<10))+(4u<<20);
        devices_.push_back(std::move(d)); report_.push_back(r);
    }
    for (size_t i=0;i<devices_.size();++i) {
        auto& d=devices_[i]; auto& r=report_[i];
        const auto [free,total]=memory_(d.backend);
        require(free<=total && free>=d.options.free_reserve &&
                d.options.tensor_cap<=free-d.options.free_reserve &&
                d.options.workspace_cap<=free-d.options.free_reserve-d.options.tensor_cap,
                "insufficient per-device free memory for expert tensors/runtime/reserve");
        r.free_before=free;
        d.executor=factory(d.backend,d.options.tensor_cap);
        require(bool(d.executor) && d.executor->supports_rows(),"resident executor lacks indexed rows");
        std::string error;
        const worker::GpuAssetReader reader=[&](const worker::GpuAsset& asset,std::vector<uint8_t>& blob,std::string& e) {
            sources(); guard(i);
            return cpu.load_blob(asset.layer,asset.expert,blob,e);
        };
        require(d.executor->prepare_source(d.plan,reader,d.counters,error),error);
        require(d.counters.gpu_allocated_bytes<=d.options.tensor_cap,"executor exceeded tensor cap");
        r.tensor_bytes=d.counters.gpu_allocated_bytes; r.weight_bytes=d.counters.gpu_weight_bytes;
        r.tensor_scratch_bytes=d.counters.gpu_scratch_bytes;
        r.staging_peak=d.counters.host_staging_peak;
        guard(i); sources();
    }
}
LocalExperts::~LocalExperts() {
    for(auto& d:devices_) if(d.executor) d.executor->release(d.counters);
}
bool LocalExperts::run(int64_t layer,const PortableRoute& route,std::vector<float>& rows,std::string& error) {
    rows.clear(); error.clear();
    try {
        sources();
        require(layer>=0 && layer<int64_t(L) && route.ids.size()==K && route.weights.size()==K &&
                route.activation.size()==H,"resident route shape mismatch");
        finite(route.activation,"resident activation"); finite(route.weights,"resident routing weights");
        std::set<int32_t> ids;
        for(auto id:route.ids) require(id>=0 && id<int32_t(NE) && ids.insert(id).second,"invalid resident expert IDs");
        require(sequence_<UINT64_MAX,"resident request sequence exhausted");
        const uint64_t sequence=++sequence_;
        std::array<bool,K> cold; cold.fill(true);
        std::vector<float> result(K*H,0);
        for(size_t die=0;die<devices_.size();++die) {
            auto& d=devices_[die]; auto& report=report_[die];
            worker::InferRequest req;
            req.model_id=d.plan.model_id; req.partition_id=d.plan.placement_id.a;
            req.session=1; req.sequence=sequence; req.layer=uint16_t(layer); req.T=1; req.K=K;
            req.ids.assign(route.ids.begin(),route.ids.end()); req.weights=route.weights; req.activations=route.activation; req.mask.assign(K,0);
            size_t count=0;
            for(size_t slot=0;slot<K;++slot) if(d.plan.owns(int(layer),route.ids[slot])) {
                req.mask[slot]=1; cold[slot]=false; ++count;
            }
            if(!count) continue;
            guard(die);
            worker::EntryRowsReply reply;
            require(d.executor->infer_rows(d.plan,req,reply,error),error);
            guard(die);
            require(reply.session==req.session && reply.sequence==sequence && reply.indices.size()==count &&
                    reply.rows.size()==count*H,"resident reply identity/shape mismatch");
            finite(reply.rows,"resident expert output");
            std::array<bool,K> seen{};
            for(size_t n=0;n<count;++n) {
                const size_t slot=reply.indices[n];
                require(slot<K && req.mask[slot] && !seen[slot],"resident reply duplicate/unowned slot");
                seen[slot]=true;
                std::copy_n(reply.rows.begin()+n*H,H,result.begin()+slot*H);
            }
            if (check_rows_) {
                std::array<bool,K> active{};
                for(size_t slot=0;slot<K;++slot) active[slot]=req.mask[slot]!=0;
                std::vector<float> reference;
                require(cpu_.run(layer,route,reference,error,&active),error);
                for(size_t n=0;n<count;++n) {
                    const size_t slot=reply.indices[n];
                    double square=0,norm=0,maximum=0;
                    for(size_t h=0;h<H;++h) {
                        const double ref=reference[slot*H+h], delta=double(reply.rows[n*H+h])-ref;
                        square+=delta*delta; norm+=ref*ref; maximum=std::max(maximum,std::abs(delta));
                    }
                    ++report.row_checks;
                    report.row_max_abs=std::max(report.row_max_abs,maximum);
                    report.row_relative_l2=std::max(report.row_relative_l2,std::sqrt(square/(norm+1e-30)));
                }
            }
            ++report.requests; report.entries+=count; gpu_entries_+=count;
        }
        std::vector<float> cpu_rows;
        require(cpu_.run(layer,route,cpu_rows,error,&cold),error);
        for(size_t slot=0;slot<K;++slot) if(cold[slot]) {
            std::copy_n(cpu_rows.begin()+slot*H,H,result.begin()+slot*H); ++cpu_entries_;
        }
        rows=std::move(result); return true;
    } catch(const std::exception& e) { rows.clear(); error=e.what(); return false; }
}
} // namespace
