#pragma once
// Offline loopback harness: real SGW1 service/client, no LAN or model access.
struct CountedVulkanExecutor final : strata::worker::GpuExecutor {
    std::unique_ptr<strata::worker::GpuExecutor> inner;
    std::atomic<unsigned> prepares{0};
    explicit CountedVulkanExecutor(std::unique_ptr<strata::worker::GpuExecutor> e):inner(std::move(e)) {}
    bool prepare(const strata::worker::GpuPlan& p,uint64_t reserve,strata::worker::GpuCache& cache,
                 strata::worker::GpuCounters& c,std::string& err) override {
        ++prepares; return inner->prepare(p,reserve,cache,c,err);
    }
    bool supports_rows() const override { return inner->supports_rows(); }
    bool infer_rows(const strata::worker::GpuPlan& p,const strata::worker::InferRequest& r,
                    strata::worker::EntryRowsReply& rows,std::string& err) override { return inner->infer_rows(p,r,rows,err); }
    bool infer(const strata::worker::GpuPlan& p,const strata::worker::InferRequest& r,
               strata::worker::InferReply& out,std::string& err) override { return inner->infer(p,r,out,err); }
    void release(strata::worker::GpuCounters& c) override { inner->release(c); }
};
struct VulkanLoopbackService {
    strata::worker::GpuWorkerService service;
    std::thread thread;
    uint16_t port=0;
    VulkanLoopbackService(strata::worker::GpuWorkerOptions o,std::unique_ptr<strata::worker::GpuExecutor> e)
        :service(std::move(o),std::move(e)) {
        std::string err; require(service.listen(port,err),err);
        thread=std::thread([this]{service.run();});
    }
    void stop() { service.stop(); if(thread.joinable()) thread.join(); }
    ~VulkanLoopbackService() { stop(); }
};
void loopback_check(const std::string& backend,int device,const std::filesystem::path& output,
                    const strata::worker::GpuPlan& plan,const std::vector<uint8_t>& first,
                    const std::vector<uint8_t>& second,const std::vector<float>& x,
                    const std::vector<float>& expected) {
    using namespace strata::worker;
    const uint64_t budget=16ull<<20;
    auto executor=[&]() { return backend == "cpu" ? create_vulkan_executor_cpu_test(budget) : create_vulkan_gpu_executor(device,budget); };
    GpuWorkerOptions o; o.plan=plan; o.vulkan=true; o.runtime_reserve=0;
    o.cache_root=(output/"loopback-cache").string(); o.cache_budget=2*first.size()+4096;
    o.deadline_ms=5000; o.setup_ms=60000;
    require(std::filesystem::create_directory(o.cache_root),"new loopback cache required");
    std::string err;
    // Valid but unsupported shape must fail before binding/listening or device initialization.
    LocalManifest m; m.n_layers=1; m.n_expert=4; m.n_embd=1280; m.n_ff=640;
    m.gu_type=m.d_type=8; m.identity={123,456}; m.index_digest={789,101};
    m.layers.resize(1); m.digests.resize(4,blob_digest(first));
    auto bad=o; require(make_gpu_plan(m,{{0,1},{0,3}},bad.plan,err),err);
    { GpuWorkerService refused(bad,executor()); uint16_t port=0;
      require(!refused.listen(port,err),"Vulkan service accepted unsupported shape"); }
    bad=o; bad.runtime_reserve=1;
    { GpuWorkerService refused(bad,executor()); uint16_t port=0;
      require(!refused.listen(port,err),"Vulkan service accepted nonzero reserve"); }
    auto counted=std::make_unique<CountedVulkanExecutor>(executor()); auto* observed=counted.get();
    VulkanLoopbackService worker(o,std::move(counted));
    GpuWorkerTier::Open open; open.plan=plan; open.port=worker.port; open.vulkan=true;
    open.entry_rows=true; open.deadline_ms=5000; open.setup_ms=60000;
    unsigned fetches=0;
    open.fetch=[&](int64_t layer,int64_t expert,uint8_t* dst,uint32_t bytes) {
        if(layer != 0 || (expert != 1 && expert != 3)) return false;
        const auto& blob=expert == 1 ? first : second;
        if(bytes != blob.size()) return false;
        ++fetches; std::memcpy(dst,blob.data(),bytes); return true;
    };
    { auto invalid=open; invalid.entry_rows=false; GpuWorkerTier tier;
      require(!tier.open(invalid,err),"Vulkan client accepted weighted mode"); }
    { auto invalid=open; invalid.host="127.0.0.2"; GpuWorkerTier tier;
      require(!tier.open(invalid,err),"Vulkan client accepted nonselected loopback address"); }
    { auto cuda=open; cuda.vulkan=false; GpuWorkerTier tier;
      require(!tier.open(cuda,err) && observed->prepares == 0 && fetches == 0,
              "Vulkan service accepted CUDA numerical contract"); }
    GpuWorkerTier tier;
    require(tier.open(open,err),err);
    require(fetches == 2 && observed->prepares == 1,"loopback setup did not load exactly two experts once");
    GpuCounters before,after;
    require(tier.status(before,err),err);
    const int32_t ids[]={3,0,1}; const uint8_t mask[]={1,0,1}; const float weights[]={0.125f,0,0.75f};
    for(unsigned request=1;request<=3;++request) {
        require(tier.post(0,x.data(),ids,weights,mask,1,3,err) && tier.finish(err) && tier.served(),err);
        const auto& rows=tier.entry_rows();
        require(rows.session == tier.stats().session && rows.sequence == request &&
                rows.indices == std::vector<uint16_t>({0,2}) && same(rows.rows,expected),
                "loopback indexed rows differ from direct hybrid reference");
        save(output/("loopback.request"+std::to_string(request)+".rows.f32"),rows.rows.data(),rows.rows.size()*4);
        require(tier.status(after,err),err);
        require(observed->prepares == 1 && fetches == 2 && after.gpu_allocation_count == 2 &&
                after.gpu_allocated_bytes == before.gpu_allocated_bytes && after.gpu_weight_bytes == before.gpu_weight_bytes &&
                after.verified_bytes == before.verified_bytes && after.transferred_bytes == before.transferred_bytes &&
                after.requests == request && after.entries == 2*request,"request reloaded/reallocated resident experts");
    }
    std::string report; tier.report(report);
    require(report.find("Vulkan/native-Q8-host-boundaries") == 0,"mislabelled Vulkan report");
    std::vector<float> batched=x; batched.insert(batched.end(),x.begin(),x.end());
    const int32_t ids2[]={3,0,1,3,0,1}; const uint8_t mask2[]={1,0,1,1,0,1};
    const float weights2[]={0.125f,0,0.75f,0.125f,0,0.75f};
    require(!tier.post(0,batched.data(),ids2,weights2,mask2,2,3,err) && tier.dead(),
            "loopback service accepted unsupported T2 execution");
    tier.close(); worker.stop();
    const auto& released=worker.service.counters(); // thread joined; no concurrent reads
    require(observed->prepares == 1 && released.gpu_allocated_bytes == 0 && released.gpu_weight_bytes == 0 &&
            released.gpu_scratch_bytes == 0 && released.gpu_allocation_count == 0,"loopback cleanup leaked owned counters");
    require(write_gpu_plan((output/"loopback.plan").string(),plan,err),err);
    std::printf("LOOPBACK_ACCEPTED backend=%s requests=3 experts=2 prepares=1 fetches=2 unsupported_t2_refused=1 indices=0,2 expert_order=3,1 row_bytes=20480 bit_differences=0 owned_tensor_bytes=%llu released_bytes=0 live_allocations=0\n",
                backend.c_str(),(unsigned long long)before.gpu_allocated_bytes);
    const std::string receipt="three loopback client/service requests matched direct hybrid bytes; one prepare; owned counters cleared\n";
    save(output/"LOOPBACK_COMPLETED",receipt.data(),receipt.size());
}
