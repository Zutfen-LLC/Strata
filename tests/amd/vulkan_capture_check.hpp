#pragma once
extern "C" unsigned strata_amd_native_reference_isa(void);
extern "C" const char* strata_amd_native_reference_compiler(void);
bool native_reference_qualified(unsigned flags) { return flags==7u; }
void require_native_reference_isa() {
    const unsigned flags=strata_amd_native_reference_isa();
    std::printf("NATIVE_CPU_ISA compiler=%s flags=%u avx2=%u fma=%u f16c=%u avx512=%u avx_vnni=%u avx512_vnni=%u avx_vnni_int8=%u fast_math=%u qualified_avx2_non_vnni=%u\n",
        strata_amd_native_reference_compiler(),flags,bool(flags&1),bool(flags&2),bool(flags&4),bool(flags&8),bool(flags&16),bool(flags&32),bool(flags&64),bool(flags&128),native_reference_qualified(flags));
    require(native_reference_qualified(flags),"native arithmetic check requires qualified AVX2/FMA/F16C, non-VNNI/non-AVX512, no-fast-math ggml-cpu; no CPU parity claim");
}
void capture_native_check(const std::filesystem::path& output,size_t record,size_t entry,
                          const strata::worker::InferRequest& q,const strata::worker::GpuAsset& asset,
                          const cpu::NativeFmt& fmt,const std::vector<uint8_t>& blob,
                          const std::vector<float>& x,const Rows& selected,const Boundaries& boundaries) {
    std::vector<uint8_t> input_q,hidden_q;
    const Rows native=native_reference(fmt,blob,x,input_q,hidden_q);
    const auto prefix=output/"capture.native";
    std::ofstream metrics(prefix.string()+".metrics.tsv");require(bool(metrics),"cannot create capture native metrics");
    metrics << "comparison\tstage\tvalues\tbit_different_values\tmax_abs\trelative_L2\n";
    compare(metrics,"capture.selected_vs_native",selected,native);
    metrics.close();require(bool(metrics),"cannot finish capture native metrics");
    const bool input_equal=input_q==boundaries.input_q,hidden_equal=hidden_q==boundaries.hidden_q;
    save_rows(output,"capture.native",native);save_rows(output,"capture.selected",selected);
    save(output/"capture.original_input.f32",x.data(),x.size()*4);
    save(output/"capture.native_input.q8_0",input_q.data(),input_q.size());
    save(output/"capture.selected_input.q8_0",boundaries.input_q.data(),boundaries.input_q.size());
    save(output/"capture.native_hidden.q8_0",hidden_q.data(),hidden_q.size());
    save(output/"capture.selected_hidden.q8_0",boundaries.hidden_q.data(),boundaries.hidden_q.size());
    std::ofstream manifest(prefix.string()+".txt");
    manifest << "record=" << record << "\nentry=" << entry << "\nlayer=" << q.layer
             << "\nexpert=" << asset.expert << "\ntoken=" << entry/q.K << "\nT=" << unsigned(q.T) << "\nK=" << unsigned(q.K)
             << "\nblob_digest=" << std::hex << asset.digest.a << ':' << asset.digest.b << std::dec
             << "\nnative_isa_flags=" << strata_amd_native_reference_isa()
             << "\ninput_q8_equal=" << input_equal << "\nhidden_q8_equal=" << hidden_equal
             << "\nCPU_model_parity=unestablished\n";
    manifest.close();require(bool(manifest),"cannot finish capture native manifest");
    std::printf("CAPTURE_NATIVE_OBSERVED record=%zu entry=%zu layer=%u expert=%u token=%zu input_q8_equal=%d hidden_q8_equal=%d qualified_avx2_non_vnni=1 CPU_model_parity=unestablished\n",
        record,entry,q.layer,asset.expert,entry/q.K,input_equal,hidden_equal);
}
// Bounded real routed inputs in the existing expert probe, using existing capture/plan/cache codecs.
void capture_check(const std::string& backend,int device,ggml_backend_t selected,
                   const std::filesystem::path& output,const std::string& capture_path,
                   const std::string& plan_path,const std::string& manifest_path,const std::string& blobs,bool native_check=false) {
    using namespace strata::worker;
    if(native_check)require_native_reference_isa();
    std::string err;
    CpuCaptureFile capture; LocalManifest manifest; GpuPlan plan,trusted;
    require(read_cpu_capture(capture_path,capture,err),err);
    require(read_local_manifest(manifest_path,manifest,err),err);
    require(read_gpu_plan(plan_path,plan,err),err);
    require(plan.assets.size()>=1 && plan.assets.size()<=2 && plan.n_embd==H && plan.n_ff==F,
            "real capture acceptance requires one or two H2560/FF640 Q8 experts");
    require(capture.identity==manifest.identity && capture.index==manifest.index_digest &&
            capture.setup.model_id==plan.model_id && capture.setup.n_layers==plan.n_layers &&
            capture.setup.n_expert==plan.n_expert && capture.setup.n_embd==H && capture.setup.n_ff==F &&
            capture.setup.gu_type==8 && capture.setup.d_type==8,"capture/source model binding mismatch");
    std::vector<std::pair<uint16_t,uint16_t>> assigned;
    for(const auto& a:plan.assets)assigned.emplace_back(a.layer,a.expert);
    require(make_gpu_plan(manifest,assigned,trusted,err),err);
    std::vector<uint8_t> a,b;
    require(encode_gpu_plan(plan,a,err) && encode_gpu_plan(trusted,b,err) && a==b,"plan does not match trusted source");
    // Validate and select at most three real routed records before initializing a worker.
    std::vector<size_t> records; std::vector<bool> seen(plan.assets.size(),false);
    for(size_t r=0;r<capture.records.size() && records.size()<3;++r) {
        const auto& record=capture.records[r]; const auto& q=record.request;
        if(!record.sent)continue;
        require(!q.packed && q.T>=1 && q.T<=kMaxT && q.K>=1 && q.K<=kMaxK &&
            q.mask.size()==size_t(q.T)*q.K && q.ids.size()==q.mask.size() && q.weights.size()==q.mask.size() &&
            q.activations.size()==size_t(q.T)*H,"capture window shape mismatch");
        bool active=false;
        for(size_t i=0;i<q.mask.size();++i)if(q.mask[i])if(const auto* asset=plan.find(q.layer,q.ids[i])) {
            seen[size_t(asset-plan.assets.data())]=true;active=true;
        }
        if(active)records.push_back(r);
    }
    require(!records.empty() && std::all_of(seen.begin(),seen.end(),[](bool v){return v;}),
            "first three intersecting capture records must exercise every assigned expert");
    const auto cache_root=output/"capture-cache";
    require(std::filesystem::create_directory(cache_root),"new capture scratch cache required");
    GpuCache cache; require(cache.open(cache_root.string(),16ull<<20,err),err);
    std::vector<uint8_t> missing;require(cache.prepare(plan,missing,err),err);
    std::vector<std::vector<uint8_t>> weights;
    for(const auto& asset:plan.assets) {
        const auto path=std::filesystem::path(blobs)/cache.key(asset);
        std::ifstream file(path,std::ios::binary|std::ios::ate);
        require(file && file.tellg()==std::streamoff(asset.bytes),"staged real blob absent or wrong size: "+path.string());
        std::vector<uint8_t> blob(asset.bytes);file.seekg(0);
        require(bool(file.read(reinterpret_cast<char*>(blob.data()),blob.size())) && blob_digest(blob)==asset.digest,
                "staged real blob digest mismatch");
        require(cache.publish(asset,blob.data(),blob.size(),err),err);weights.push_back(std::move(blob));
    }
    cache.close(); // service exclusively owns its scratch cache from here
    cpu::NativeFmt fmt;require(cpu::native_fmt(8,8,H,F,fmt,err),err);
    std::vector<std::unique_ptr<ExpertGraph>> direct;
    for(const auto& blob:weights)direct.push_back(std::make_unique<ExpertGraph>(selected,fmt,blob,true));
    auto inner=backend=="cpu" ? create_vulkan_executor_cpu_test(16ull<<20) : create_vulkan_gpu_executor(device,16ull<<20);
    auto counted=std::make_unique<CountedVulkanExecutor>(std::move(inner));auto* observed=counted.get();
    GpuWorkerOptions options;options.vulkan=true;options.runtime_reserve=0;options.plan=plan;
    options.cache_root=cache_root.string();options.cache_budget=16ull<<20;options.deadline_ms=30000;
    VulkanLoopbackService worker(options,std::move(counted));
    GpuWorkerTier::Open open;open.plan=plan;open.port=worker.port;open.entry_rows=open.vulkan=true;open.deadline_ms=30000;
    // Prepublished, digest-verified scratch blobs must be reused; never fetch synthetic weights.
    unsigned fetches=0;open.fetch=[&](int64_t,int64_t,uint8_t*,uint32_t){++fetches;return false;};
    GpuWorkerTier tier;require(tier.open(open,err),err);
    GpuCounters before,after;require(tier.status(before,err),err);
    size_t entries=0;bool native_observed=false;
    for(size_t r:records) {
        auto q=capture.records[r].request;std::vector<uint16_t> indices;std::vector<float> expected;
        for(size_t i=0;i<q.mask.size();++i) {
            const auto* asset=plan.find(q.layer,q.ids[i]);q.mask[i]=q.mask[i] && asset;
            if(!q.mask[i])continue;
            const size_t token=i/q.K;const auto start=q.activations.begin()+token*H;
            const size_t slot=size_t(asset-plan.assets.data());const std::vector<float> token_input(start,start+H);
            const auto row=direct[slot]->run(token_input);
            if(native_check && !native_observed) {
                capture_native_check(output,r,i,q,*asset,fmt,weights[slot],token_input,row,direct[slot]->boundaries());
                native_observed=true; // exactly one original entry, not one per token/record
            }
            indices.push_back(uint16_t(i));expected.insert(expected.end(),row.out.begin(),row.out.end());
        }
        std::vector<int32_t> ids(q.ids.begin(),q.ids.end());
        require(tier.post(q.layer,q.activations.data(),ids.data(),q.weights.data(),q.mask.data(),q.T,q.K,err) && tier.finish(err),err);
        const auto& got=tier.entry_rows();
        require(got.indices==indices && same(got.rows,expected),"real captured indexed rows differ from same-backend direct hybrid");
        entries+=indices.size();
        save(output/("capture.record"+std::to_string(r)+".rows.f32"),got.rows.data(),got.rows.size()*4);
        save(output/("capture.record"+std::to_string(r)+".direct.f32"),expected.data(),expected.size()*4);
        std::printf("CAPTURE_ROW record=%zu T=%u K=%u entries=%zu bit_differences=0\n",r,q.T,q.K,indices.size());
    }
    require(tier.status(after,err),err);
    require(observed->prepares==1 && fetches==0 && after.requests==records.size() && after.entries==entries &&
            before.gpu_allocated_bytes==after.gpu_allocated_bytes && before.verified_bytes==after.verified_bytes &&
            after.transferred_bytes==0,"real capture lifecycle reloaded resident experts");
    tier.close();worker.stop();
    require(worker.service.counters().gpu_allocated_bytes==0 && worker.service.counters().gpu_allocation_count==0,
            "real capture release counters not zero");
    std::printf("CAPTURE_ROWS_ACCEPTED backend=%s records=%zu experts=%zu entries=%zu prepares=1 fetches=0 released_bytes=0 CPU_model_parity=unestablished\n",
                backend.c_str(),records.size(),plan.assets.size(),entries);
    const std::string receipt="captured indexed rows match same-backend direct hybrid; full prompt/decode CPU gates remain mandatory\n";
    save(output/"CAPTURE_ROWS_COMPLETED",receipt.data(),receipt.size());
}
// Host-only codec/lifecycle regression. These inputs are explicitly synthetic, never model evidence.
void capture_fixture_check(ggml_backend_t selected,const std::filesystem::path& output,
                           const std::vector<uint8_t>& first,const std::vector<float>& x) {
    using namespace strata::worker;
    std::string err;cpu::NativeFmt fmt;require(cpu::native_fmt(8,8,H,F,fmt,err),err);
    auto second=first;
    for(size_t off=fmt.up_off;off<fmt.down_off;off+=34)for(size_t j=2;j<34;++j)second[off+j]=uint8_t(-int8_t(second[off+j]));
    LocalManifest m;m.n_layers=1;m.n_expert=2;m.n_embd=H;m.n_ff=F;m.gu_type=m.d_type=8;
    m.index_digest={789,101};m.files={{"synthetic.gguf",2*fmt.bytes}};m.layers.resize(1);
    m.layers[0]={LocalRange{0,0,fmt.up_off},LocalRange{0,2*fmt.up_off,fmt.up_off},LocalRange{0,4*fmt.up_off,fmt.bytes-fmt.down_off}};
    m.digests={blob_digest(first),blob_digest(second)};m.role_digests.resize(2);
    const auto manifest=(output/"synthetic.local").string();require(write_local_manifest(manifest,m,err),err);
    require(read_local_manifest(manifest,m,err),err); // derive the codec's source identity
    GpuPlan plan;require(make_gpu_plan(m,{{0,0},{0,1}},plan,err),err);
    const auto plan_path=(output/"synthetic.plan").string();require(write_gpu_plan(plan_path,plan,err),err);
    const auto staged=output/"synthetic-blobs";require(std::filesystem::create_directory(staged),"fixture blob directory");
    GpuCache cache;std::vector<uint8_t> missing;
    require(cache.open(staged.string(),16ull<<20,err) && cache.prepare(plan,missing,err),err);
    require(cache.publish(plan.assets[0],first.data(),first.size(),err) && cache.publish(plan.assets[1],second.data(),second.size(),err),err);
    cache.close();
    SetupManifest setup;setup.nonce=1;setup.model_id=m.identity.a;setup.n_layers=1;setup.n_expert=2;
    setup.n_embd=H;setup.n_ff=F;setup.gu_type=setup.d_type=8;setup.partition={3};setup.partition_id=blob_digest(setup.partition).a;
    std::strcpy(setup.model_name,"synthetic-capture");std::strcpy(setup.quant_name,"Q8_0");
    CpuCapture capture;const auto path=(output/"synthetic.capture").string();
    require(capture.prepare(setup,m.identity,m.index_digest,path,err,1ull<<20),err);
    require(capture.request("GEN 1 1 2 3",{1,2,3},1),capture.error());
    InferRequest q;q.session=q.sequence=1;q.model_id=setup.model_id;q.partition_id=setup.partition_id;q.layer=0;q.K=2;
    for(int T:{2,1}) {
        q.T=T;q.ids.clear();q.mask.assign(T*2,1);q.weights.assign(T*2,0.5f);q.activations.clear();
        for(int t=0;t<T;++t){q.ids.insert(q.ids.end(),{1,0});auto token=x;for(auto& v:token)v*=t+1;
            q.activations.insert(q.activations.end(),token.begin(),token.end());}
        std::vector<float> partials(T*H,0.f);
        require(capture.window(T==2?0:2,T,T==2?1:2) && capture.begin(q) && capture.finish(partials.data()),capture.error());
    }
    const float logits[]={0.f,1.f};require(capture.seal(logits,2),capture.error());
    const auto result=output/"synthetic-capture-result";require(std::filesystem::create_directory(result),"fixture result directory");
    const bool qualified=native_reference_qualified(strata_amd_native_reference_isa());
    capture_check("cpu",-1,selected,result,path,plan_path,manifest,staged.string(),qualified);
    require(native_reference_qualified(7) && !native_reference_qualified(3) && !native_reference_qualified(7|8) &&
            !native_reference_qualified(7|16) && !native_reference_qualified(7|32) && !native_reference_qualified(7|64) &&
            !native_reference_qualified(7|128),"native reference ISA policy");
    if(!qualified) {
        bool refused=false;try{require_native_reference_isa();}catch(...){refused=true;}
        require(refused,"unqualified native reference accepted");
    }
    auto refuse=[&](const std::string& cap,const std::string& source,const std::string& label) {
        const auto bad=output/label;require(std::filesystem::create_directory(bad),"fixture refusal directory");
        bool refused=false;try{capture_check("cpu",-1,selected,bad,cap,plan_path,source,staged.string());}catch(...){refused=true;}
        require(refused,"capture binding/corruption refusal missing");
    };
    auto foreign=m;foreign.index_digest.a++;
    const auto foreign_path=(output/"foreign.local").string();require(write_local_manifest(foreign_path,foreign,err),err);
    refuse(path,foreign_path,"foreign-result");
    save(output/"truncated.capture","bad",3);refuse((output/"truncated.capture").string(),manifest,"truncated-result");
    std::puts("SYNTHETIC_CAPTURE_SELF_CHECK T2=1 T1=1 trusted_plan=1 digest=1 binding_refused=1 corruption_refused=1 actual_model=0");
}
