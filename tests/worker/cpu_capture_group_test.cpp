// One small resident Q8 arena, direct replay then actual one-member group. No engine/model.
#include "local_fixture.hpp"
#include "diagnostic_child.hpp"
#include "cpu_single_group_replay.hpp"
using namespace strata::worker::diagnostic;
namespace {
WorkerCacheInfo cache(const Child& child){
    ExpertWorkerClient c;WorkerCacheInfo result;std::string err;
    require(c.connect_local("127.0.0.1",child.port,5000,{},err) && c.cache_status(result,err),err);return result;
}
CpuCaptureFile capture(const Fixture& fixture){
    CpuCaptureFile f;f.setup=fixture.setup();f.setup.partition={7,7};f.setup.partition_id=blob_digest(f.setup.partition).a;
    f.identity=fixture.m.identity;f.index=fixture.m.index_digest;
    for(int c=0;c<8;++c){
        CpuCaptureRecord r;auto& q=r.request;q.T=c<2?2:1;q.K=10;q.layer=c%2;
        q.model_id=f.setup.model_id;q.partition_id=f.setup.partition_id;q.session=1;q.sequence=c+1;
        q.activations.resize((size_t)q.T*2560);
        for(size_t i=0;i<q.activations.size();++i)q.activations[i]=(int(i%29)-14)*.03125f;
        q.mask.resize(q.T*q.K);q.ids.resize(q.T*q.K);q.weights.resize(q.T*q.K);
        std::vector<const uint8_t*> blobs(q.mask.size());
        for(size_t i=0;i<q.mask.size();++i){
            q.ids[i]=c==4 || c==5?3:(i+c)%4;q.mask[i]=f.setup.owns(q.layer,q.ids[i]);
            q.weights[i]=i%2?-.25f:.5f;blobs[i]=fixture.blobs[q.layer*4+q.ids[i]].data();r.sent=r.sent||q.mask[i];
        }
        std::string err;
        require(ExpertWorkerService::direct_native_partial(fixture.fmt,q.T,q.K,q.mask.data(),q.weights.data(),
            q.activations.data(),blobs.data(),r.partials,err),err);
        f.records.push_back(std::move(r));
    }
    return f;
}
bool direct(const CpuCaptureFile& f,const Fixture& fixture,const Child& child,CpuReplayReport& report,std::string& err){
    ExpertWorkerClient client;
    if(!client.connect_local("127.0.0.1",child.port,5000,{},err) ||
       !client.setup_local(f.setup,fixture.m.identity,false,err,true) || !client.cache_hit() || !client.mark_ready(err))return false;
    uint64_t sequence=0;
    return replay_cpu_capture(f,[&](const InferRequest& saved,std::vector<float>& y,std::string& why){
        auto q=saved;q.session=client.session();q.sequence=++sequence;InferReply reply;
        if(!client.infer(q,reply,why))return false;y=std::move(reply.partials);return true;
    },report,err);
}
void test(const char* binary){
    Fixture fixture(4);const auto f=capture(fixture);
    ChildSpec spec;spec.binary=binary;spec.root=fixture.root.string();spec.manifest=(fixture.root/"local.manifest").string();
    spec.workers=2;spec.budget_mib=32;Child child(spec);
    SingleGroupReport group;CpuReplayReport validated;std::string err;
    const auto empty=cache(child);
    check(!replay_single_group(f,fixture.m,"127.0.0.1",child.port,empty,validated,group,err) &&
        err=="single group requires completed full-reference validation","incomplete reference refuses before group setup");
    validated.full_checked=validated.grouped_checked=(uint32_t)f.records.size();
    check(!replay_single_group(f,fixture.m,"127.0.0.1",child.port,empty,validated,group,err) &&
        !cache(child).arena_allocations,"empty residency refused without loading");
    {
        ExpertWorkerTier tier;ExpertWorkerTier::Open o;o.local=&fixture.m;o.port=child.port;o.resident=true;
        o.n_layers=2;o.n_expert=4;o.n_embd=2560;o.n_ff=640;o.partition=f.setup.partition;
        require(tier.open(o,err),err); // fixture's only cold load; before the diagnostic
    }
    const auto before=cache(child);
    require(before.complete && !before.session && before.arena_allocations==1,"single fixture resident");
    require(direct(f,fixture,child,validated,err),err);
    require(replay_single_group(f,fixture.m,"127.0.0.1",child.port,before,validated,group,err),err);
    check(group.checked==8 && group.active==6 && group.zero==2 && group.cache_hit && group.resident_unchanged,
        "multi-token K10, zero work and reactivation bitmatch real one-member group");
    check(unchanged_resident(before,cache(child)),"one arena, no reload/unload/expiry after group reconnect");
    auto bad=f;bad.records[0].partials[0]+=1.f;
    CpuReplayReport failed;
    check(!direct(bad,fixture,child,failed,err) && failed.full_checked==0,"direct mismatch stops full validation");
    const auto no_group=cache(child);
    check(!replay_single_group(bad,fixture.m,"127.0.0.1",child.port,before,failed,group,err) &&
        cache(child).cache_hits==no_group.cache_hits,"failed full validation cannot start group setup");
    check(!replay_single_group(bad,fixture.m,"127.0.0.1",child.port,before,validated,group,err) &&
        group.checked==0 && group.resident_unchanged && err=="single group bit mismatch at record 0",
        "group mismatch disconnects and preserves resident accounting and primary error");
    require(replay_single_group(f,fixture.m,"127.0.0.1",child.port,before,validated,group,err),err);
    check(group.checked==8 && unchanged_resident(before,cache(child)),"valid replay reconnects after refused comparison");
    auto stale=before;++stale.arena_allocations;
    check(!replay_single_group(f,fixture.m,"127.0.0.1",child.port,stale,validated,group,err) &&
        unchanged_resident(before,cache(child)),"changed cache accounting refuses before group setup");
    check(child.owns_listener(),"fixture still owns its live listener");child.stop();
    int status=0;errno=0;check(::waitpid(-1,&status,WNOHANG)<0 && errno==ECHILD,"fixture child cleanly reaped");
}
}
int main(int argc,char** argv){
    if(argc!=2)return 2;
    try{test(argv[1]);}catch(const std::exception& e){std::fprintf(stderr,"CPU_CAPTURE_GROUP exception: %s\n",e.what());return 2;}
    std::printf("CPU_CAPTURE_GROUP_TEST failures=%d\n",failures);return failures?1:0;
}
