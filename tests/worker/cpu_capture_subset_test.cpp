// Small real Q8 workers only. No engine, model assets or external endpoint.
#include "local_fixture.hpp"
#include "cpu_subset_replay.hpp"
using namespace strata::worker::diagnostic;
namespace {
std::vector<int> allowed(){cpu_set_t mask;CPU_ZERO(&mask);require(::sched_getaffinity(0,sizeof mask,&mask)==0,"allowed CPUs");
    std::vector<int> out;for(int i=0;i<CPU_SETSIZE;++i)if(CPU_ISSET(i,&mask))out.push_back(i);return out;}
ExpertWorkerTier::Open full_option(const Fixture& f,const Child& child){
    ExpertWorkerTier::Open o;o.port=child.port;o.local=&f.m;o.resident=true;o.n_layers=2;o.n_expert=4;
    o.n_embd=2560;o.n_ff=640;o.gu_type=o.d_type=8;o.partition={7,7};o.setup_ms=30000;o.deadline_ms=5000;return o;
}
CpuCaptureFile capture(const Fixture& fixture){
    CpuCaptureFile f;f.setup=fixture.setup();f.setup.partition={7,7};f.setup.partition_id=blob_digest(f.setup.partition).a;
    f.identity=fixture.m.identity;f.index=fixture.m.index_digest;
    for(int c=0;c<13;++c){CpuCaptureRecord r;auto& q=r.request;q.T=c==0?2:1;q.K=4;q.layer=c>=4?0:c%2;
        q.model_id=f.setup.model_id;q.partition_id=f.setup.partition_id;q.session=1;q.sequence=c+1;
        q.activations.resize((size_t)q.T*2560);for(size_t i=0;i<q.activations.size();++i)q.activations[i]=(int(i%23)-11)*.03125f;
        q.weights.resize(q.T*q.K);q.mask.resize(q.T*q.K);q.ids.resize(q.T*q.K);
        std::vector<const uint8_t*> blobs(q.T*q.K);
        for(size_t j=0;j<q.mask.size();++j){q.ids[j]=c==3?3:c==12?1:c>=4?0:j%4;q.mask[j]=f.setup.owns(q.layer,q.ids[j]);
            q.weights[j]=(j%2?-.2f:.3f);blobs[j]=fixture.blobs[q.layer*4+q.ids[j]].data();r.sent=r.sent||q.mask[j];}
        std::string err;require(ExpertWorkerService::direct_native_partial(fixture.fmt,q.T,q.K,q.mask.data(),q.weights.data(),
            q.activations.data(),blobs.data(),r.partials,err),err);f.records.push_back(std::move(r));}
    return f;
}
WorkerCacheInfo cache(const Child& child){ExpertWorkerClient c;std::string e;WorkerCacheInfo info;
    require(c.connect_local("127.0.0.1",child.port,5000,{},e) && c.cache_status(info,e),e);return info;}
void test(const char* binary){
    Fixture fixture(4);const auto cpus=allowed();require(!cpus.empty(),"CPU affinity available");
    const auto f=capture(fixture);ChildSpec full_spec;full_spec.binary=binary;full_spec.root=fixture.root.string();
    full_spec.manifest=(fixture.root/"local.manifest").string();full_spec.workers=2;full_spec.budget_mib=32;
    full_spec.setup_cpus={cpus[0]};Child full(full_spec);
    MaskedReferences references(f);CpuReplayReport replay;std::string err;
    {
        ExpertWorkerTier tier;require(tier.open(full_option(fixture,full),err),err);
        require(replay_cpu_capture(f,[&](const InferRequest& q,std::vector<float>& y,std::string& e){
            std::vector<int32_t> ids(q.ids.begin(),q.ids.end());
            if(!tier.post(q.layer,q.activations.data(),ids.data(),q.weights.data(),q.mask.data(),q.T,q.K,e) || !tier.finish(e))return false;
            y=tier_partial(tier,(size_t)q.T*2560);return true;
        },replay,err,[&](size_t i,int p,const std::vector<float>& y){references.save(i,p,y);}),err);
    }
    const auto before=cache(full);check(before.complete && before.arena_allocations==1,"full fixture remains resident before subsets");
    SubsetConfig config;config.binary=binary;config.local_root=fixture.root.string();config.expected_bytes=temporary_bytes(f);
    config.output_root=(fixture.root/"success").string();config.workers=2;config.loaders=4;config.setup_ms=30000;config.total_ms=60000;
    config.caller=cpus[0];config.pool={cpus[0],cpus.size()>1?cpus[1]:cpus[0]};config.reserve_bytes=1u<<20;config.max_owned_rss=1ull<<30;
    const auto config_path=(fixture.root/"binding.conf").string();
    {std::ofstream out(config_path);out<<"worker_binary "<<std::quoted(config.binary)<<"\nlocal_root "<<std::quoted(config.local_root)
        <<"\noutput_root "<<std::quoted((fixture.root/"config-out").string())<<"\ntemporary_arena_bytes "<<config.expected_bytes
        <<"\nreserve_bytes 8589934592\nmax_owned_rss_bytes 1073741824\nworkers 2\nloader_threads 4\nframe_ms 1000\nsetup_ms 30000\ntotal_ms 60000\ncaller_cpu "
        <<config.caller<<"\npool_cpus "<<config.pool[0]<<","<<config.pool[1]<<"\n";}
    const auto binding=read_subset_config(config_path,f,fixture.m);
    check(binding.expected_bytes==config.expected_bytes && binding.pool==config.pool,"private config binds actual files, geometry and allowed CPUs");
    {std::ofstream out(config_path,std::ios::app);out<<"extra_field 1\n";}
    bool unknown=false;try{read_subset_config(config_path,f,fixture.m);}catch(const std::exception&){unknown=true;}
    check(unknown && !fs::exists(binding.output_root),"unknown config refused before child/output creation");
    const auto limits=fixture.root/"synthetic-cgroup";fs::create_directory(limits);
    {std::ofstream(limits/"memory.current")<<(108ull<<30);std::ofstream(limits/"memory.max")<<(200ull<<30);std::ofstream(limits/"memory.high")<<"max";}
    check(!cgroup_room(limits,(108ull+8)<<30),"finite shared cap refuses temporary arenas plus reserve despite global RAM");
    {std::ofstream(limits/"memory.max")<<(256ull<<30);}
    check(cgroup_room(limits,(108ull+8)<<30),"sufficient finite cgroup headroom");
    {std::ofstream(limits/"memory.high")<<(200ull<<30);}
    check(!cgroup_room(limits,(108ull+8)<<30),"finite memory.high checked independently");
    config.frame_ms=1000;
    SubsetReport report;require(run_local_subsets(f,fixture.m,full_spec.manifest,references,config,report,err,
        []{std::this_thread::sleep_for(std::chrono::milliseconds(200));}),err);
    check(report.peers_checked==2*f.records.size() && report.groups_checked==f.records.size() && report.reaped,
          "multi-token, signed weights, zero/one-peer, real subset mapping and group/manual bits");
    check(report.manual_ms>1000,"consecutive one-peer work exceeds idle deadline before later peer reactivation");
    for(pid_t child:report.children){int status=0;errno=0;check(::waitpid(child,&status,WNOHANG)<0 && errno==ECHILD,"successful temporary child reaped");}
    auto after=cache(full);check(after.complete && !after.session && before.resident_bytes==after.resident_bytes &&
        before.loaded_bytes==after.loaded_bytes && before.verified_bytes==after.verified_bytes &&
        after.arena_allocations==1 && !after.unloads && !after.expirations,"full resident unaffected by subset success");
    references.peer[0][0]+=1;config.output_root=(fixture.root/"mismatch").string();
    check(!run_local_subsets(f,fixture.m,full_spec.manifest,references,config,report,err) &&
        report.reaped && err.find("subset peer 0 bit mismatch at record 0")!=std::string::npos,"peer failure preserves original cause and reaps both owned children");
    for(pid_t child:report.children){int status=0;errno=0;check(::waitpid(child,&status,WNOHANG)<0 && errno==ECHILD,"failed temporary child reaped");}
    after=cache(full);check(after.complete && after.resident_bytes==before.resident_bytes && after.arena_allocations==1 &&
        !after.unloads && !after.expirations,"full resident unaffected by failure cleanup");
    const auto fake=(fixture.root/"fake-worker.sh").string();
    {std::ofstream out(fake);out<<"#!/bin/sh\nprintf '%s\\n' 'strata-worker: listening on 127.0.0.1:"<<full.port
        <<" (budget 32 MiB, 2 pool workers)' >&2\nexec /bin/sleep 30\n";}
    require(::chmod(fake.c_str(),0700)==0,"owned fake executable");
    ChildSpec forged=full_spec;forged.binary=fake;bool wrong_owner=false;
    try{Child rejected(forged);}catch(const std::exception& e){wrong_owner=std::string(e.what()).find("not owned")!=std::string::npos;}
    check(wrong_owner && full.owns_listener() && cache(full).complete,"forged full-resident port is refused without control or stop");
    config.output_root=(fixture.root/"guard").string();config.max_owned_rss=1;
    check(!run_local_subsets(f,fixture.m,full_spec.manifest,references,config,report,err) &&
        report.children[0]<0 && !fs::exists(config.output_root),"resource guard refuses before starting children");
    full.stop();
    ChildSpec bad=full_spec;bad.binary="/bin/false";
    bool refused=false;try{Child exited(bad);}catch(const std::exception&){refused=true;}
    int status=0;errno=0;check(refused && ::waitpid(-1,&status,WNOHANG)<0 && errno==ECHILD,"pre-listener death is reaped without a child leak");
}
}
int main(int argc,char** argv){
    if(argc!=2)return 2;
    try{test(argv[1]);}catch(const std::exception& e){std::fprintf(stderr,"CPU_CAPTURE_SUBSET exception: %s\n",e.what());return 2;}
    std::printf("CPU_CAPTURE_SUBSET_TEST failures=%d\n",failures);return failures?1:0;
}
