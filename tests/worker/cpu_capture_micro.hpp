#pragma once
// Record0 discriminator: real selected weights, never a full resident endpoint or full-size arena.
#include "cpu_subset_replay.hpp" // retained owned-child/resource helpers; no subset runner is called
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/worker/setup_keepalive.hpp"
#include "cpu_reply_readiness.hpp"

namespace strata::worker::diagnostic {
inline constexpr uint64_t kMicroPayloadMax=104448000;
struct MicroPlan {
    strata::kernels::cpu::NativeFmt fmt;
    std::vector<uint16_t> experts;
    std::vector<int8_t> owners;
    std::vector<uint8_t> expected,partition[2],mask[2];
    uint64_t bytes[2]{};
};
struct MicroReference {std::vector<float> full,peer[2];};
struct MicroReport {
    bool reference=false,peer[2]{},group=false,reaped=true;
    pid_t children[2]{-1,-1};uint64_t payload=0,peak_rss=0,min_available=UINT64_MAX;
};
inline uint64_t readiness_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline void measure_reply_readiness(const MicroPlan& p,const InferRequest& q,
    const std::vector<std::vector<uint8_t>>& blobs,const std::vector<uint8_t>& act,
    const std::vector<float>& reference,const std::function<void()>& guard) {
    using namespace strata::kernels::cpu;
    // Runs before any child starts. The selected verified weights total at most 104,448,000 bytes.
    ExpertPool pool(12);
    auto trace=std::make_unique<ExpertPool::NativeReadiness>();
    uint64_t clock_min=UINT64_MAX;
    for(int i=0;i<128;++i){const auto a=readiness_ns(),b=readiness_ns();clock_min=std::min(clock_min,b-a);}
    std::printf("READINESS_CONFIG workers=12 host_works=1 T=2 K=10 H=2560 rate_bytes_s=125000000 "
        "trace_bytes=%zu clock_pair_min_ns=%llu warmups=2 pairs=3 order=baseline-then-trace "
        "model=zero-cost-ascending-frame no_chunk_headers=1 LAN=0\n",sizeof(*trace),(unsigned long long)clock_min);
    for(int peer=0;peer<2;++peer) {
        guard();
        std::vector<ExpertJobMulti> jobs;
        std::vector<int> job_of(p.owners.size(),-1),entry_job(q.mask.size(),-1);
        std::vector<float> rows(reference.size(),0.f);
        for(size_t i=0;i<q.mask.size();++i)if(p.mask[peer][i]) {
            const auto expert=q.ids[i];
            if(job_of[expert]<0) {
                job_of[expert]=(int)jobs.size();ExpertJobMulti j;
                const auto at=std::lower_bound(p.experts.begin(),p.experts.end(),expert)-p.experts.begin();
                j.blob=blobs.at(at).data();jobs.push_back(j);
            }
            auto& j=jobs[job_of[expert]];entry_job[i]=job_of[expert];
            need(j.nt<MAXT,"readiness job token bound");
            j.nact[j.nt]=act.data()+(i/q.K)*kNativeActBytes;
            j.out[j.nt++]=rows.data()+i*2560;
        }
        auto exact=[&] {
            for(size_t i=0;i<q.mask.size();++i)if(p.mask[peer][i])
                need(!std::memcmp(rows.data()+i*2560,reference.data()+i*2560,2560*sizeof(float)),
                    "readiness native pool row bits");
        };
        for(int warm=0;warm<2;++warm){guard();pool.run_split_multi_native(p.fmt,jobs.data(),(int)jobs.size());exact();}
        for(int repeat=0;repeat<3;++repeat) {
            guard();auto start=readiness_ns();
            pool.run_split_multi_native(p.fmt,jobs.data(),(int)jobs.size());
            const auto baseline=readiness_ns()-start;exact();
            start=readiness_ns();
            pool.run_split_multi_native(p.fmt,jobs.data(),(int)jobs.size(),trace.get());
            const auto traced=readiness_ns()-start;exact();
            const auto model=reply_readiness(*trace,(int)jobs.size(),2560,entry_job,start);
            std::printf("READINESS peer=%d repeat=%d jobs=%zu tasks=%d bytes=%llu baseline_pool_ns=%llu "
                "traced_pool_ns=%llu paired_delta_ns=%lld down_start_ns=%llu down_done_ns=%llu pool_done_ns=%llu "
                "first_range_ns=%llu first_prefix_ns=%llu first_row_ns=%llu first_wire_row_ns=%llu batch_finish_ns=%llu "
                "prefix_finish_ns=%llu row_finish_ns=%llu prefix_saved_ns=%llu row_saved_ns=%llu row_bits=1\n",
                peer,repeat,jobs.size(),trace->count,(unsigned long long)model.bytes,
                (unsigned long long)baseline,(unsigned long long)traced,(long long)traced-(long long)baseline,
                (unsigned long long)(trace->down_start_ns-start),(unsigned long long)(trace->down_done_ns-start),
                (unsigned long long)(trace->pool_done_ns-start),(unsigned long long)(model.first_range-start),
                (unsigned long long)(model.first_prefix-start),(unsigned long long)(model.first_row-start),
                (unsigned long long)(model.first_wire_row-start),
                (unsigned long long)(model.batch_finish-start),(unsigned long long)(model.prefix_finish-start),
                (unsigned long long)(model.row_finish-start),(unsigned long long)(model.batch_finish-model.prefix_finish),
                (unsigned long long)(model.batch_finish-model.row_finish));
            for(int task=0;task<trace->count;++task)
                std::printf("READINESS_TASK peer=%d repeat=%d task=%d begin_dim=%lld end_dim=%lld done_ns=%llu\n",
                    peer,repeat,task,(long long)(jobs.size()*2560ull*task/trace->count),
                    (long long)(jobs.size()*2560ull*(task+1)/trace->count),
                    (unsigned long long)(trace->tasks[task].done_ns-start));
        }
    }
    std::puts("READINESS_DONE row_bits=1 ordered_reference_checked=1 timings=perturbed paired_delta_is_not_calibrated_overhead");
}
inline bool micro_bits(const std::vector<float>& a,const std::vector<float>& b){
    return a.size()==b.size() && std::all_of(a.begin(),a.end(),[](float v){return std::isfinite(v);}) &&
        std::all_of(b.begin(),b.end(),[](float v){return std::isfinite(v);}) &&
        !std::memcmp(a.data(),b.data(),a.size()*sizeof(float));
}
inline MicroPlan capture_micro_plan(const CpuCaptureFile& f,const LocalManifest& local){
    need(!f.records.empty() && local.matches(f.setup) && local.identity==f.identity &&
         local.index_digest==f.index,"micro capture/manifest identity mismatch");
    const auto& r=f.records.front();const auto& q=r.request;
    need(q.layer==0 && q.T==2 && q.K==10 && f.setup.n_embd==2560 && f.setup.n_ff==640 &&
         f.setup.gu_type==8 && f.setup.d_type==8 && q.model_id==f.setup.model_id &&
         q.partition_id==f.setup.partition_id && q.ids.size()==20 && q.mask.size()==20 &&
         q.weights.size()==20 && q.activations.size()==5120 && r.partials.size()==5120 && r.sent,
         "micro requires record0 layer0 T2 K10 Q8_0 2560x640");
    need(std::all_of(q.weights.begin(),q.weights.end(),[](float v){return std::isfinite(v);}) &&
         std::all_of(q.activations.begin(),q.activations.end(),[](float v){return std::isfinite(v);}) &&
         std::all_of(r.partials.begin(),r.partials.end(),[](float v){return std::isfinite(v);}),"micro nonfinite input/reference");
    MicroPlan p;std::string err;
    need(strata::kernels::cpu::native_fmt(8,8,2560,640,p.fmt,err) && p.fmt.bytes==kMicroPayloadMax/20,err);
    // Ownership comes from the COMPLETE original CPU bitmap, before selecting any record IDs.
    p.owners=cpu_capture_owners(f.setup);p.expected.assign(f.setup.partition.size(),0);
    for(int peer=0;peer<2;++peer){p.partition[peer]=p.expected;p.mask[peer].assign(20,0);}
    std::set<uint16_t> selected;
    for(size_t i=0;i<20;++i){
        need(q.ids[i]<f.setup.n_expert && q.mask[i]==(f.setup.owns(0,q.ids[i])?1:0),"micro ownership/ID mismatch");
        if(q.mask[i]){
            const int peer=p.owners[q.ids[i]];need(peer==0 || peer==1,"micro original owner missing");
            p.mask[peer][i]=1;selected.insert(q.ids[i]);
        }
    }
    p.experts.assign(selected.begin(),selected.end());need(!p.experts.empty() && p.experts.size()<=20,"micro expert bound");
    for(uint16_t e:p.experts){const int peer=p.owners[e];p.expected[e/8]|=(uint8_t)(1u<<(e%8));
        p.partition[peer][e/8]|=(uint8_t)(1u<<(e%8));p.bytes[peer]+=p.fmt.bytes;}
    need(p.bytes[0] && p.bytes[1] && p.bytes[0]+p.bytes[1]<=kMicroPayloadMax,"micro needs two active bounded peers");
    return p;
}
inline MicroReference capture_micro_reference(const CpuCaptureFile& f,const LocalManifest& local,
    const std::string& root,const MicroPlan& p,const std::function<void()>& guard,bool readiness=false){
    const auto& q=f.records.front().request;const size_t H=f.setup.n_embd;
    // At most 60 source ranges. This private receipt gives exact missing-asset bindings without
    // asking the operator to copy whole model shards. LocalSource still requires original files.
    for(uint16_t e:p.experts)for(size_t role=0;role<3;++role){const auto& r=local.layers[0][role];
        std::printf("MICRO_RANGE expert=%u role=%zu file=%s offset=%llu bytes=%llu source_file_bytes=%llu\n",
            (unsigned)e,role,local.files.at(r.file).name.c_str(),(unsigned long long)(r.offset+(uint64_t)e*r.bytes),
            (unsigned long long)r.bytes,(unsigned long long)local.files.at(r.file).bytes);}
    LocalSource source;std::string err;need(source.open(root,local,err),err);
    for(const auto& r:local.layers[0])need(source.file(r.file,err)>=0,err);
    std::vector<float> rows(20*H,0.f),hidden(p.fmt.n_ff);
    std::vector<uint8_t> blob(p.fmt.bytes),act(2*strata::kernels::cpu::kNativeActBytes,0),hq(p.fmt.h_bytes);
    std::vector<std::vector<uint8_t>> selected; // diagnostic only; released before owned workers start
    using namespace strata::kernels::cpu;
    for(int t=0;t<2;++t)native_quant_act(p.fmt,q.activations.data()+t*H,act.data()+t*kNativeActBytes);
    for(uint16_t e:p.experts){
        size_t dst=0;
        for(const auto& r:local.layers[0]){
            need(r.bytes<=blob.size()-dst,"micro role geometry exceeds one blob");
            for(size_t at=0;at<r.bytes;){guard();const size_t n=std::min<size_t>(256u<<10,r.bytes-at);
                need(source.read_opened(r.file,r.offset+(uint64_t)e*r.bytes+at,blob.data()+dst+at,n,err),err);at+=n;}
            dst+=r.bytes;
        }
        need(dst==blob.size() && blob_digest(blob)==local.digests.at(e),
             "micro selected source digest mismatch");
        if(readiness)selected.push_back(blob);
        // Independent native row evaluation, not the worker pool/group scheduling path.
        for(size_t i=0;i<20;++i)if(q.mask[i] && q.ids[i]==e){guard();
            const void* a=act.data()+(i/10)*kNativeActBytes;float* h=hidden.data();
            native_gu_rows(p.fmt,blob.data(),&a,1,&h,0,(int)p.fmt.n_ff);
            native_quant_h(p.fmt,hidden.data(),hq.data());const void* packed=hq.data();float* out=rows.data()+i*H;
            native_down_rows(p.fmt,blob.data(),&packed,1,&out,0,(int)H);
        }
    }
    source.close_files();MicroReference ref;ref.full.assign(2*H,0.f);
    for(auto& peer:ref.peer)peer.assign(2*H,0.f);
    // Frame-entry order within each token, with signed weights kept exactly as captured.
    for(size_t i=0;i<20;++i)if(q.mask[i])for(size_t h=0;h<H;++h){const size_t at=(i/10)*H+h;
        ref.full[at]=ref.full[at]+q.weights[i]*rows[i*H+h];
        auto& peer=ref.peer[p.owners[q.ids[i]]];peer[at]=peer[at]+q.weights[i]*rows[i*H+h];}
    need(micro_bits(ref.full,f.records.front().partials),"micro full reference/capture bit mismatch; no workers started");
    if(readiness)measure_reply_readiness(p,q,selected,act,rows,guard);
    std::puts("MICRO_BITS stage=independent-full-vs-capture bits=1 workers_started=0");return ref;
}
inline bool micro_reaped(pid_t pid){if(pid<=0)return true;int status=0;errno=0;
    return ::waitpid(pid,&status,WNOHANG)<0 && errno==ECHILD;}
inline bool run_capture_micro(const CpuCaptureFile& f,const LocalManifest& local,const std::string& manifest,
    const std::string& root,const std::string& output,const std::string& binary_a,const std::string& binary_b,
    MicroReport& report,std::string& err,const std::function<void()>& before_second_setup={},bool readiness=false){
    report={};
    try{
        const auto p=capture_micro_plan(f,local);report.payload=p.bytes[0]+p.bytes[1];
        for(const auto& path:{manifest,root,output,binary_a,binary_b})need(std::filesystem::path(path).is_absolute(),"micro paths must be absolute");
        need(std::filesystem::is_directory(root) && std::filesystem::is_regular_file(manifest),"micro source paths missing");
        LocalManifest checked;need(read_local_manifest(manifest,checked,err) && checked.identity==local.identity &&
            checked.index_digest==local.index_digest,"micro worker manifest binding mismatch");
        for(const auto& path:{binary_a,binary_b})need(std::filesystem::is_regular_file(path) && ::access(path.c_str(),X_OK)==0,"micro worker executable missing");
        need(!std::filesystem::exists(output) && std::filesystem::is_directory(std::filesystem::path(output).parent_path()),"micro output must be new");
        SubsetConfig limits;limits.expected_bytes=report.payload;limits.max_owned_rss=1ull<<30;limits.total_ms=180000;
        SubsetGuard ga(limits),gb(limits);auto guard=[&]{ga.check();gb.check();};guard();
        struct ResourceReceipt {SubsetGuard& a;SubsetGuard& b;MicroReport& r;
            ~ResourceReceipt(){r.peak_rss=a.peak+b.peak;r.min_available=std::min(a.min_available,b.min_available);}} receipt{ga,gb,report};
        const auto ref=capture_micro_reference(f,local,root,p,guard,readiness);report.reference=true;ga.check(true);gb.check(true);
        need(std::filesystem::create_directory(output),"micro output creation");
        std::filesystem::permissions(output,std::filesystem::perms::owner_all,std::filesystem::perm_options::replace);
        std::printf("MICRO_PLAN record=0 layer=0 T=2 K=10 selected=%zu arena_payload=%llu workers=12 loaders=4 full_size_slot_ranks=NOT_TESTED\n",
            p.experts.size(),(unsigned long long)report.payload);std::fflush(stdout);
        uint64_t ordinal=0;unsigned micro_slot[2]{};
        for(int e=0;e<f.setup.n_expert;++e)if(f.setup.owns(0,e)){
            if(std::binary_search(p.experts.begin(),p.experts.end(),(uint16_t)e)){
                const int peer=p.owners[e];std::printf("MICRO_SLOT expert=%d peer=%d original_split_slot=%llu micro_slot=%u\n",
                    e,peer,(unsigned long long)(ordinal/2),micro_slot[peer]++);}
            ++ordinal;
        }
        ChildSpec sa;sa.binary=binary_a;sa.root=root;sa.manifest=manifest;sa.log_path=output+"/peer0.log";
        sa.budget_mib=(p.bytes[0]+1048575)/1048576;Child a(sa);report.children[0]=a.pid;ga.child=&a;
        std::printf("MICRO_OWNED peer=0 pid=%d port=%u bytes=%llu\n",a.pid,(unsigned)a.port,(unsigned long long)p.bytes[0]);std::fflush(stdout);guard();
        ChildSpec sb=sa;sb.binary=binary_b;sb.log_path=output+"/peer1.log";sb.budget_mib=(p.bytes[1]+1048575)/1048576;
        Child b(sb);report.children[1]=b.pid;gb.child=&b;
        std::printf("MICRO_OWNED peer=1 pid=%d port=%u bytes=%llu\n",b.pid,(unsigned)b.port,(unsigned long long)p.bytes[1]);std::fflush(stdout);guard();
        std::vector<ExpertWorkerTier::Open> options(2);
        for(int peer=0;peer<2;++peer){auto& o=options[peer];o.port=peer?b.port:a.port;o.local=&local;o.resident=true;
            o.n_layers=f.setup.n_layers;o.n_expert=f.setup.n_expert;o.n_embd=f.setup.n_embd;o.n_ff=f.setup.n_ff;
            o.gu_type=o.d_type=8;o.partition=p.partition[peer];o.setup_ms=30000;o.deadline_ms=5000;
            o.cancelled=[&]{guard();return false;};}
        need(ExpertWorkerGroup::validate(options,p.expected,err),err);
        const auto& q=f.records.front().request;std::vector<int32_t> ids(q.ids.begin(),q.ids.end());
        std::vector<float> manual[2];
        {
            ExpertWorkerTier ta,tb;ga.check(true);need(ta.open(options[0],err),err);
            {
                CpuSetupKeepalive keeper(ta,50);auto second=options[1];
                second.cancelled=[&]{guard();return keeper.failed();};
                if(before_second_setup)before_second_setup(); // bounded synthetic seam; CLI supplies none
                guard();gb.check(true);const bool opened=tb.open(second,err);std::string prior;
                const bool alive=keeper.finish(prior);need(alive,prior);need(opened,err);
            } // first peer's keeper joins before either numerical post
            guard();
            std::string ea,eb;
            const bool pa=ta.post(0,q.activations.data(),ids.data(),q.weights.data(),p.mask[0].data(),2,10,ea);
            const bool pb=tb.post(0,q.activations.data(),ids.data(),q.weights.data(),p.mask[1].data(),2,10,eb);
            const bool fa=ta.finish(ea),fb=tb.finish(eb);need(pa && fa,ea);need(pb && fb,eb);
            manual[0]=tier_partial(ta,5120);manual[1]=tier_partial(tb,5120);
            for(int peer=0;peer<2;++peer){report.peer[peer]=micro_bits(manual[peer],ref.peer[peer]);
                need(report.peer[peer],"micro actual peer "+std::to_string(peer)+" bit mismatch");
                std::printf("MICRO_BITS stage=actual-peer%d-vs-independent bits=1\n",peer);}
        }
        auto sum=manual[0];for(size_t i=0;i<sum.size();++i)sum[i]=sum[i]+manual[1][i];guard();
        {
            for(auto& o:options)o.resident_reuse_only=true;
            ExpertWorkerGroup group;need(group.open(options,p.expected,err),err);
            for(int peer=0;peer<2;++peer)need(group.peer_stats(peer).cache_hit &&
                group.peer_stats(peer).reused_bytes==p.bytes[peer] && !group.peer_stats(peer).local_bytes &&
                !group.peer_stats(peer).setup_network_bytes,"micro group must reuse selected subsets");
            need(group.post(0,q.activations.data(),ids.data(),q.weights.data(),q.mask.data(),2,10,err) && group.finish(err),err);
            report.group=group.served() && micro_bits(std::vector<float>(group.partials(),group.partials()+5120),sum);
            need(report.group && group.peer_stats(0).requests==1 && group.peer_stats(1).requests==1,"micro actual group/manual bit mismatch");
            std::puts("MICRO_BITS stage=actual-group-vs-manual bits=1");
        }
        guard();double maximum=0,delta2=0,norm2=0;
        for(size_t i=0;i<sum.size();++i){double d=(double)sum[i]-ref.full[i];maximum=std::max(maximum,std::abs(d));delta2+=d*d;norm2+=(double)ref.full[i]*ref.full[i];}
        std::printf("MICRO_REGROUP_VS_UNSPLIT maxabs=%.9g relativeL2=%.9g (descriptive; full-model screens unchanged)\n",
            maximum,std::sqrt(delta2/std::max(1e-30,norm2)));
        for(int peer=0;peer<2;++peer){Child& child=peer?b:a;need(child.owns_listener(),"micro cache control ownership");
            ExpertWorkerClient c;WorkerCacheInfo cache;WorkerStatusInfo status;
            need(c.connect_local("127.0.0.1",child.port,5000,{},err) && c.cache_status(cache,err) && c.status(status,err),err);
            need(cache.complete && !cache.session && cache.idle_remaining_ms==kResidencyNoExpiry &&
                cache.resident_bytes==p.bytes[peer] && cache.loaded_bytes==p.bytes[peer] && cache.verified_bytes==p.bytes[peer] &&
                cache.arena_allocations==1 && cache.peak_arena_bytes==p.bytes[peer] && !cache.expirations && !cache.unloads && !status.failures &&
                !status.setup_weight_bytes && status.requests==2,"micro selected cache/request accounting mismatch");
            std::printf("MICRO_CACHE peer=%d bytes=%llu allocations=1 requests=2 failures=0 uploads=0\n",peer,(unsigned long long)cache.resident_bytes);
            c.close();child.stop();if(peer)gb.child=nullptr;else ga.child=nullptr;guard();}
        report.reaped=micro_reaped(report.children[0]) && micro_reaped(report.children[1]);need(report.reaped,"micro owned child reap");
        return true;
    }catch(const std::exception& e){err=e.what();report.reaped=micro_reaped(report.children[0]) && micro_reaped(report.children[1]);return false;}
}
}
