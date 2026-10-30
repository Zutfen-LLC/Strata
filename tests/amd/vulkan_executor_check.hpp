#pragma once
// Offline acceptance harness only. Never linked into a runtime worker.

void executor_check(const std::string& backend, int device, const std::filesystem::path& output,
                    const std::vector<uint8_t>& blob, const std::vector<float>& x,
                    const std::vector<float>& direct) {
    using namespace strata::worker;
    std::string err;
    LocalManifest m;
    m.n_layers=1; m.n_expert=4; m.n_embd=2560; m.n_ff=640; m.gu_type=m.d_type=8;
    m.identity={123,456}; m.index_digest={789,101}; m.layers.resize(1);
    m.digests.resize(4,blob_digest(blob));
    GpuPlan plan;
    require(make_gpu_plan(m,{{0,2}},plan,err),err);
    const auto root=output/"adapter-cache";
    require(std::filesystem::create_directory(root),"new adapter cache required");
    GpuCache cache;
    require(cache.open(root.string(),blob.size()+4096,err),err);
    std::vector<uint8_t> missing;
    require(cache.prepare(plan,missing,err) && missing == std::vector<uint8_t>{1},err);
    auto executor=backend == "cpu" ? create_vulkan_executor_cpu_test() : create_vulkan_gpu_executor(device);
    GpuCounters c;
    InferRequest r;
    r.session=13; r.sequence=17; r.model_id=plan.model_id; r.partition_id=plan.placement_id.a;
    r.layer=0; r.T=1; r.K=3; r.mask={0,0,1}; r.ids={0,0,2};
    r.weights={0,0,0.125f}; r.activations=x;
    EntryRowsReply reply;
    require(!executor->infer_rows(plan,r,reply,err),"infer before prepare accepted");
    require(!executor->prepare(plan,0,cache,c,err),"missing cache accepted");
    require(c.gpu_allocated_bytes == 0 && c.gpu_allocation_count == 0,"failed prepare leaked counters");
    require(cache.publish(plan.assets[0],blob.data(),blob.size(),err),err);
    auto smaller=m; smaller.n_embd=1280;
    GpuPlan unsupported;
    require(make_gpu_plan(smaller,{{0,2}},unsupported,err),err);
    require(!executor->prepare(unsupported,0,cache,c,err),"unsupported shape accepted");
    require(!executor->prepare(plan,1,cache,c,err),"unsupported reserve accepted");
    require(executor->prepare(plan,0,cache,c,err),err);
    require(c.gpu_weight_bytes == blob.size() && c.gpu_allocated_bytes >= blob.size() &&
            c.gpu_allocated_bytes < (16u<<20) && c.gpu_allocation_count == 2,"owned allocation accounting");
    InferReply weighted;
    require(!executor->infer(plan,r,weighted,err),"weighted inference accepted");
    auto bad=r; bad.T=2;
    require(!executor->infer_rows(plan,bad,reply,err),"malformed T2 accepted");
    bad=r; bad.mask={0,0,2};
    require(!executor->infer_rows(plan,bad,reply,err),"invalid entry mask accepted");
    bad=r; bad.ids[2]=1;
    require(!executor->infer_rows(plan,bad,reply,err),"unassigned expert accepted");
    bad=r; bad.activations.pop_back();
    require(!executor->infer_rows(plan,bad,reply,err),"short activation accepted");
    bad=r; bad.activations[0]=std::numeric_limits<float>::quiet_NaN();
    require(!executor->infer_rows(plan,bad,reply,err),"nonfinite input accepted");
    auto other=m; other.identity.a++;
    GpuPlan changed;
    require(make_gpu_plan(other,{{0,2}},changed,err),err);
    require(!executor->infer_rows(changed,r,reply,err),"changed plan accepted");
    require(executor->infer_rows(plan,r,reply,err),err);
    require(reply.session == r.session && reply.sequence == r.sequence &&
            reply.indices == std::vector<uint16_t>{2} && same(reply.rows,direct),
            "adapter row differs in bytes/index/binding from direct hybrid graph");
    const auto owned=c.gpu_allocated_bytes;
    executor->release(c); executor->release(c);
    require(c.gpu_allocated_bytes == 0 && c.gpu_weight_bytes == 0 && c.gpu_scratch_bytes == 0 &&
            c.gpu_allocation_count == 0,"release did not clear owned allocation counters");
    require(!executor->infer_rows(plan,r,reply,err) && reply.rows.empty(),"infer after release accepted");
    // Reprepare lifecycle costs another graph only on the explicitly substituted host backend.
    if(backend == "cpu") {
        require(executor->prepare(plan,0,cache,c,err),err);
        require(executor->prepare(plan,0,cache,c,err),err);
        require(!executor->prepare(unsupported,0,cache,c,err),"unsupported reprepare accepted");
        require(!executor->infer_rows(plan,r,reply,err),"failed reprepare retained old graph");
        executor->release(c);
        require(c.gpu_allocated_bytes == 0 && c.gpu_allocation_count == 0,"reprepare release counters");
    }
    std::printf("ADAPTER_ACCEPTED backend=%s index=2 row_bytes=10240 bit_differences=0 owned_tensor_bytes=%llu released_bytes=0 live_allocations=0\n",
                backend.c_str(),(unsigned long long)owned);
    const std::string receipt="same-device/build direct hybrid row bytes matched; owned tensor counters cleared\n";
    save(output/"ADAPTER_COMPLETED",receipt.data(),receipt.size());
}
