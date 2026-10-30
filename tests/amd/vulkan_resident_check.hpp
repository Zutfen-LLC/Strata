#pragma once
// Two-expert hardware discriminator plus narrow host lifecycle/budget checks.
void resident_check(const std::string& backend, int device, const std::filesystem::path& output,
                    ggml_backend_t selected, const cpu::NativeFmt& fmt,
                    const std::vector<uint8_t>& first, const std::vector<float>& x,
                    const Rows& first_direct, std::ofstream& metrics, bool loopback=false) {
    using namespace strata::worker;
    // A distinct legal fixture: negate only the second expert's Q8 up-projection codes.
    auto second=first;
    for(size_t off=fmt.up_off;off<fmt.down_off;off+=34)
        for(size_t j=2;j<34;++j) {
            const int code=second[off+j]<128 ? int(second[off+j]) : int(second[off+j])-256;
            second[off+j]=uint8_t(-code);
        }
    Rows second_direct;
    { ExpertGraph direct(selected,fmt,second,true); second_direct=direct.run(x); }
    require(!same(first_direct.out,second_direct.out),"resident fixtures do not distinguish expert association");
    std::vector<uint8_t> aq,hq;
    const auto native=native_reference(fmt,second,x,aq,hq);
    compare(metrics,"signal.expert3.direct_hybrid_vs_native",second_direct,native);
    save(output/"expert3.q8_0",second.data(),second.size());
    save_rows(output,"signal.expert3.direct_hybrid",second_direct);
    save_rows(output,"signal.expert3.native",native);

    LocalManifest m;
    m.n_layers=1; m.n_expert=4; m.n_embd=H; m.n_ff=F; m.gu_type=m.d_type=8;
    m.identity={123,456}; m.index_digest={789,101}; m.layers.resize(1);
    m.digests.resize(4,blob_digest(first)); m.digests[3]=blob_digest(second);
    GpuPlan plan;
    std::string err;
    require(make_gpu_plan(m,{{0,1},{0,3}},plan,err),err);
    if(loopback) {
        std::vector<float> expected=second_direct.out;
        expected.insert(expected.end(),first_direct.out.begin(),first_direct.out.end());
        loopback_check(backend,device,selected,output,plan,first,second,x,expected); return;
    }
    const auto root=output/"resident-cache";
    require(std::filesystem::create_directory(root),"new resident cache required");
    GpuCache cache;
    require(cache.open(root.string(),3*first.size()+4096,err),err);
    std::vector<uint8_t> missing;
    require(cache.prepare(plan,missing,err),err);
    require(cache.publish(plan.assets[0],first.data(),first.size(),err),err);
    require(cache.publish(plan.assets[1],second.data(),second.size(),err),err);
    constexpr uint64_t budget=16ull<<20;
    auto executor=backend == "cpu" ? create_vulkan_executor_cpu_test(budget) : create_vulkan_gpu_executor(device,budget);
    GpuCounters c;
    require(executor->prepare(plan,0,cache,c,err),err);
    require(c.gpu_allocation_count == 2 && c.gpu_weight_bytes == 2*first.size() &&
            c.gpu_allocated_bytes <= budget && c.gpu_scratch_bytes > 0 && c.gpu_allocated_bytes-c.gpu_scratch_bytes >= c.gpu_weight_bytes,
            "resident tensor accounting exceeds budget");
    const auto owned=c.gpu_allocated_bytes;
    InferRequest r;
    r.session=13; r.sequence=17; r.model_id=plan.model_id; r.partition_id=plan.placement_id.a;
    r.layer=0; r.T=1; r.K=3; r.mask={1,0,1}; r.ids={3,0,1};
    r.weights={0.125f,0,0.75f}; r.activations=x;
    EntryRowsReply reply;
    auto bad=r; bad.ids[2]=2; // Valid first entry must not hide an invalid later entry.
    require(!executor->infer_rows(plan,bad,reply,err) && reply.rows.empty(),"unassigned resident expert accepted");
    bad=r; bad.ids[2]=m.n_expert;
    require(!executor->infer_rows(plan,bad,reply,err),"out-of-range expert accepted");
    bad=r; bad.mask[2]=2;
    require(!executor->infer_rows(plan,bad,reply,err),"invalid entry mask accepted");
    bad=r; bad.K=2;
    require(!executor->infer_rows(plan,bad,reply,err),"entry length/index mismatch accepted");
    require(executor->infer_rows(plan,r,reply,err),err);
    std::vector<float> expected=second_direct.out;
    expected.insert(expected.end(),first_direct.out.begin(),first_direct.out.end());
    require(reply.session == r.session && reply.sequence == r.sequence &&
            reply.indices == std::vector<uint16_t>({0,2}) && same(reply.rows,expected),
            "reverse resident rows differ in bytes/index/binding from direct hybrid");
    save(output/"resident.rows.f32",reply.rows.data(),reply.rows.size()*sizeof(float));
    executor->release(c); executor->release(c);
    require(c.gpu_allocated_bytes == 0 && c.gpu_weight_bytes == 0 && c.gpu_scratch_bytes == 0 &&
            c.gpu_allocation_count == 0,"resident release counters not cleared");
    require(!executor->infer_rows(plan,r,reply,err),"resident infer after release accepted");
    if(backend == "cpu") {
        require(executor->prepare(plan,0,cache,c,err),err);
        require(executor->prepare(plan,0,cache,c,err) && c.gpu_allocated_bytes == owned &&
                c.gpu_allocation_count == 2,"repeat resident prepare accounting");
        executor->release(c);
        // Refuse both a payload-only shortage and a one-byte tensor/alignment shortage.
        for(uint64_t limit : {2*uint64_t(first.size())-1,owned-1}) {
            auto limited=create_vulkan_executor_cpu_test(limit);
            GpuCounters refused;
            require(!limited->prepare(plan,0,cache,refused,err) && err.find("budget") != std::string::npos,
                    "resident budget was not enforced");
            require(refused.gpu_allocated_bytes == 0 && refused.gpu_weight_bytes == 0 &&
                    refused.gpu_scratch_bytes == 0 && refused.gpu_allocation_count == 0 &&
                    refused.gpu_peak_bytes <= limit,"budget refusal leaked or exceeded owned allocation");
            require(!limited->infer_rows(plan,r,reply,err),"budget refusal left resident state");
        }
        // A third resident proves the implementation is budget-bounded rather than capped at two.
        GpuPlan three;
        require(make_gpu_plan(m,{{0,0},{0,1},{0,3}},three,err),err);
        require(cache.prepare(three,missing,err),err);
        require(cache.publish(three.assets[0],first.data(),first.size(),err),err);
        const auto three_bytes=owned+first.size();
        auto larger=create_vulkan_executor_cpu_test(three_bytes);
        require(larger->prepare(three,0,cache,c,err) && c.gpu_allocation_count == 2 &&
                c.gpu_allocated_bytes == three_bytes,"third resident or exact tensor budget rejected");
        larger->release(c);
        require(c.gpu_allocated_bytes == 0 && c.gpu_allocation_count == 0,"third resident release counters");
    }
    std::ofstream manifest(output/"resident_manifest.txt");
    manifest << "resident_experts=2\nentry_indices=0,2\nentry_experts=3,1\ntensor_budget=" << budget
             << "\nowned_tensor_bytes=" << owned << "\nweight_bytes=" << 2*first.size()
             << "\nexpert1_fnv1a64=" << std::hex << digest(first.data(),first.size())
             << "\nexpert3_fnv1a64=" << digest(second.data(),second.size()) << std::dec
             << "\nreleased_bytes=0\nlive_allocations=0\n";
    manifest.close(); require(bool(manifest),"resident manifest write failed");
    std::printf("RESIDENT_ACCEPTED backend=%s experts=2 indices=0,2 expert_order=3,1 row_bytes=20480 bit_differences=0 tensor_budget=%llu owned_tensor_bytes=%llu released_bytes=0 live_allocations=0\n",
                backend.c_str(),(unsigned long long)budget,(unsigned long long)owned);
    const std::string receipt="two reversed indexed rows matched direct hybrid bytes; owned tensor counters cleared\n";
    save(output/"RESIDENT_COMPLETED",receipt.data(),receipt.size());
}
