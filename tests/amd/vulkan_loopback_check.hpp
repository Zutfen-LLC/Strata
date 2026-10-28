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
void loopback_check(const std::string& backend,int device,ggml_backend_t selected,const std::filesystem::path& output,
                    const strata::worker::GpuPlan& plan,const std::vector<uint8_t>& first,
                    const std::vector<uint8_t>& second,const std::vector<float>& x,
                    const std::vector<float>& expected) {
    using namespace strata::worker;
    const uint64_t budget=16ull<<20;
    auto executor=[&]() { return backend == "cpu" ? create_vulkan_executor_cpu_test(budget) : create_vulkan_gpu_executor(device,budget); };
#ifdef STRATA_TEST_TLS
    TlsFixture tls_fixture;
#endif
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
    bad=o;bad.bind_ip="10.0.0.5";bad.allow_peer_ips={"10.0.0.2"};
    { GpuWorkerService refused(bad,executor());uint16_t port=0;
      require(!refused.listen(port,err) && err.find("mTLS")!=std::string::npos,"private Vulkan plaintext listener accepted"); }
#ifdef STRATA_TEST_TLS
    o.tls=tls_fixture.config("worker","coordinator.test");
#endif
    auto counted=std::make_unique<CountedVulkanExecutor>(executor()); auto* observed=counted.get();
    VulkanLoopbackService worker(o,std::move(counted));
    GpuWorkerTier::Open open; open.plan=plan; open.port=worker.port; open.vulkan=true;
#ifdef STRATA_TEST_TLS
    open.tls=tls_fixture.config("coordinator","worker.test");
#endif
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
    { auto invalid=open; invalid.host="10.0.0.5"; invalid.tls={}; GpuWorkerTier tier;
      require(!tier.open(invalid,err),"Vulkan client accepted private plaintext"); }
    { auto cuda=open; cuda.vulkan=false; GpuWorkerTier tier;
      require(!tier.open(cuda,err) && observed->prepares == 0 && fetches == 0,
              "Vulkan service accepted CUDA numerical contract"); }
    GpuWorkerTier tier;
    require(tier.open(open,err),err);
    require(fetches == 2 && observed->prepares == 1,"loopback setup did not load exactly two experts once");
    GpuCounters before,after;
    require(tier.status(before,err),err);
    cpu::NativeFmt fmt; require(cpu::native_fmt(8,8,H,F,fmt,err),err);
    ExpertGraph direct_first(selected,fmt,first,true), direct_second(selected,fmt,second,true);
    unsigned total_entries=0, request=0;
    for(int T=1;T<=kMaxT;++T) {
        ++request;
        std::vector<float> activations;
        std::vector<int32_t> ids; std::vector<uint8_t> mask; std::vector<float> weights, reference;
        std::vector<uint16_t> indices;
        for(int t=0;t<T;++t) {
            auto token=x;
            for(auto& value:token)value*=float(t+1)/T; // distinct token input detects entry/K mistakes
            activations.insert(activations.end(),token.begin(),token.end());
            ids.insert(ids.end(),{3,0,1}); mask.insert(mask.end(),{1,0,1}); weights.insert(weights.end(),{0.125f,0,0.75f});
            const auto a=direct_second.run(token),b=direct_first.run(token);
            reference.insert(reference.end(),a.out.begin(),a.out.end());
            reference.insert(reference.end(),b.out.begin(),b.out.end());
            indices.push_back(uint16_t(t*3));indices.push_back(uint16_t(t*3+2));
        }
        if(T==1)require(same(reference,expected),"T1 direct reference changed");
        require(tier.post(0,activations.data(),ids.data(),weights.data(),mask.data(),T,3,err) && tier.finish(err) && tier.served(),err);
        const auto& rows=tier.entry_rows();
        require(rows.session == tier.stats().session && rows.sequence == request &&
                rows.indices == indices && same(rows.rows,reference),
                "window indexed rows differ from same-token direct hybrid reference");
        save(output/("loopback.request"+std::to_string(request)+".rows.f32"),rows.rows.data(),rows.rows.size()*4);
        save(output/("loopback.request"+std::to_string(request)+".direct.f32"),reference.data(),reference.size()*4);
        total_entries+=2*T;
        require(tier.status(after,err),err);
        require(observed->prepares == 1 && fetches == 2 && after.gpu_allocation_count == 2 &&
                after.gpu_allocated_bytes == before.gpu_allocated_bytes && after.gpu_weight_bytes == before.gpu_weight_bytes &&
                after.verified_bytes == before.verified_bytes && after.transferred_bytes == before.transferred_bytes &&
                after.requests == request && after.entries == total_entries,"request reloaded/reallocated resident experts");
    }
    std::string report; tier.report(report);
    require(report.find("Vulkan/native-Q8-host-boundaries") == 0,"mislabelled Vulkan report");
    std::vector<float> oversized((kMaxT+1)*H,0.f),weights((kMaxT+1)*3,0.f);
    std::vector<int32_t> ids((kMaxT+1)*3,1); std::vector<uint8_t> mask(ids.size(),1);
    require(!tier.post(0,oversized.data(),ids.data(),weights.data(),mask.data(),kMaxT+1,3,err) && tier.dead(),
            "oversized token window accepted");
    tier.close(); worker.stop();
    const auto& released=worker.service.counters(); // thread joined; no concurrent reads
    require(observed->prepares == 1 && released.gpu_allocated_bytes == 0 && released.gpu_weight_bytes == 0 &&
            released.gpu_scratch_bytes == 0 && released.gpu_allocation_count == 0,"loopback cleanup leaked owned counters");
    require(write_gpu_plan((output/"loopback.plan").string(),plan,err),err);
    std::printf("LOOPBACK_ACCEPTED backend=%s requests=%u experts=2 prepares=1 fetches=2 bounded_t1_through_t8=1 oversized_refused=1 t1_indices=0,2 t1_expert_order=3,1 t1_row_bytes=20480 bit_differences=0 owned_tensor_bytes=%llu released_bytes=0 live_allocations=0\n",
                backend.c_str(),request,(unsigned long long)before.gpu_allocated_bytes);
    const std::string receipt="all bounded token-window loopback client/service requests matched direct hybrid bytes; one prepare; owned counters cleared\n";
    save(output/"LOOPBACK_COMPLETED",receipt.data(),receipt.size());
}
