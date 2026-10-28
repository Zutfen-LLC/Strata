// One small resident Q8 arena, direct replay then actual one-member group. No engine/model.
#include "local_fixture.hpp"
#include "diagnostic_child.hpp"
#include "cpu_single_group_replay.hpp"
#include "cpu_regroup.hpp"
using namespace strata::worker::diagnostic;
namespace {
void ordered_checks() {
    InferRequest q;q.T=2;q.K=3;q.sequence=10;q.ids={0,1,2,3,4,5};q.mask={1,0,1,0,1,0};
    q.weights={.25f,.7f,-.5f,.9f,2.f,.1f};q.activations={.25f,-.5f,.75f,1.f};
    const uint16_t H=2;std::vector<float> rows={-4.f,2.f,9.f,9.f,3.f,-8.f,9.f,9.f,-2.f,.125f,9.f,9.f};
    std::vector<float> reference(4);
    auto reduce=[&]{ExpertWorkerService::accumulate_partial(q.T,q.K,q.mask.data(),q.weights.data(),rows.data(),H,reference.data());};
    reduce();uint64_t sequence=10,calls=0,bytes=0;InferReply result;std::string err;int entered=0,fail_at=0;
    auto exchange=[&](const InferRequest& r,InferReply& reply,std::string& why) {
        ++entered;
        check(r.ids==q.ids && r.activations==q.activations && r.sequence==q.sequence+entered-1,
              "ordered preserves IDs/activation and monotonic sequences");
        if(entered==fail_at){why="owned exchange refused";return false;}
        if(entered==1) {
            check(r.weights==q.weights && r.mask==q.mask,"ordered shadow uses original weights/mask");
            reply.partials=reference;return true;
        }
        check(std::all_of(r.weights.begin(),r.weights.end(),[](float w){return w==1.f;}),"row requests use unit weights");
        for(size_t i=0;i<r.mask.size();++i)check(r.mask[i]==(q.mask[i] && i%3==size_t(entered-2)),
            "row rank order and masks are exact");
        reply.partials.assign(4,0.f);
        for(int t=0;t<2;++t)for(int rank=0;rank<3;++rank)if(r.mask[t*3+rank]) {
            const size_t i=t*3+rank;check(q.mask[i]==1,"row cannot acquire GPU entry");
            std::copy_n(rows.data()+i*H,H,reply.partials.data()+t*H);
        }
        return true;
    };
    require(ordered_cpu_request(q,H,sequence,exchange,result,calls,bytes,err),err);
    check(calls==4 && bytes==48 && !std::memcmp(result.partials.data(),reference.data(),16),
          "shadow plus three unit rank replies reconstruct original weighted bits");
    reference[0]=std::nextafter(reference[0],std::numeric_limits<float>::infinity());q.sequence=++sequence;entered=0;
    check(!ordered_cpu_request(q,H,sequence,exchange,result,calls,bytes,err) && result.partials.empty() &&
          err.find("ordered bit gate mismatch")==0,"one-ULP rounding mismatch is refused without publication");
    q.mask={1,0,0,1,0,0};reduce();q.sequence=++sequence;entered=0;
    require(ordered_cpu_request(q,H,sequence,exchange,result,calls,bytes,err),err);
    check(calls==2,"empty ranks are skipped and one rank can carry different token entries");
    q.mask={1,0,1,0,1,0};reduce();q.sequence=++sequence;entered=0;fail_at=3;
    check(!ordered_cpu_request(q,H,sequence,exchange,result,calls,bytes,err) && result.partials.empty() &&
          calls==3 && err=="owned exchange refused","row failure after shadow/successful row publishes nothing");
    // A numerically equal sign difference must fail the bit gate, never be normalized away.
    InferRequest zero;zero.T=zero.K=1;zero.ids={0};zero.mask={1};zero.weights={1.f};zero.activations={0.f};zero.sequence=++sequence;
    entered=0;
    check(!ordered_cpu_request(zero,1,sequence,[&](const InferRequest&,InferReply& reply,std::string&) {
        reply.partials={++entered==1?-0.f:0.f};return true;
    },result,calls,bytes,err) && result.partials.empty() && err.find("ordered bit gate mismatch")==0,
          "signed-zero bit mismatch refuses without partial publication");
    InferRequest maximum;maximum.T=kMaxT;maximum.K=kMaxK;maximum.sequence=++sequence;
    maximum.ids.assign(kMaxT*kMaxK,0);maximum.mask.assign(kMaxT*kMaxK,1);maximum.weights.assign(kMaxT*kMaxK,1.f);
    maximum.activations.assign(kMaxT*kMaxNEmbd,0.f);
    require(ordered_cpu_request(maximum,kMaxNEmbd,sequence,[&](const InferRequest&,InferReply& reply,std::string&) {
        reply.partials.assign(kMaxT*kMaxNEmbd,0.f);return true;
    },result,calls,bytes,err),err);
    check(calls==11 && bytes==819200,"maximum shape bounded to eleven exchanges and819200 row bytes");
    maximum.T=kMaxT+1;entered=0;
    check(!ordered_cpu_request(maximum,kMaxNEmbd,sequence,[&](const InferRequest&,InferReply&,std::string&) {
        ++entered;return true;
    },result,calls,bytes,err) && !entered && !bytes && result.partials.empty(),"oversize refuses before exchange/rows");
}
void regroup_order() {
    SetupManifest m;m.n_layers=2;m.n_expert=5;m.partition={21,18};
    const auto owners=cpu_capture_owners(m); // layer1 IDs1/4 belong to peers1/0, not local parity
    InferRequest q;q.layer=1;q.T=1;q.K=2;q.ids={4,1};q.mask={1,1};q.weights={-.25f,.75f};
    q.activations={.5f};q.sequence=7;
    uint64_t sequence=7,calls[2]{};InferReply result;std::string err;int entered=0;
    auto exchange=[&](const InferRequest& r,InferReply& part,std::string&) {
        check(r.ids==q.ids && r.weights==q.weights && r.activations==q.activations,"regroup preserves input bytes");
        const std::vector<uint8_t> expected_mask=entered?std::vector<uint8_t>{0,1}:std::vector<uint8_t>{1,0};
        check(r.mask==expected_mask && r.sequence==uint64_t(7+entered),"original ownership, peer0 before peer1, monotonic wire sequence");
        part.partials=entered++?std::vector<float>{-1e20f,-0.f,2.f}:std::vector<float>{1e20f,-0.f,1.f};return true;
    };
    require(regroup_cpu_request(q,owners,5,3,sequence,exchange,result,calls,err),err);
    check(result.partials==std::vector<float>({0.f,-0.f,3.f}) && std::signbit(result.partials[1]) &&
        calls[0]==1 && calls[1]==1 && sequence==8,"ordered group float addition and signed zero");
    for(int active=0;active<2;++active) {
        q.mask=active?std::vector<uint8_t>{0,1}:std::vector<uint8_t>{1,0};q.sequence=++sequence;
        uint64_t one[2]{};int frames=0;
        require(regroup_cpu_request(q,owners,5,3,sequence,[&](const InferRequest& r,InferReply& part,std::string&) {
            ++frames;check(r.mask==q.mask && r.sequence==q.sequence,"empty peer skipped without sequence use");
            part.partials={-0.f,1.f,2.f};return true;
        },result,one,err),err);
        check(frames==1 && one[active]==1 && !one[1-active] && std::signbit(result.partials[0]),
            "one active peer copied without zero addition");
    }
    q.mask={1,1};q.sequence=++sequence;entered=0;
    check(!regroup_cpu_request(q,owners,5,3,sequence,[&](const InferRequest&,InferReply& part,std::string& why) {
        part.partials={1.f,2.f,3.f};if(++entered==2){why="second exchange failed";return false;}return true;
    },result,calls,err) && entered==2 && result.partials.empty() && err=="second exchange failed",
        "failed second exchange publishes no first-peer partial");
}
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
void live_regroup(const Fixture& fixture,const CpuCaptureFile& f,const Child& child) {
    ExpertWorkerGroup::Open option;option.local=&fixture.m;option.port=child.port;
    option.resident=option.resident_reuse_only=option.diagnostic_regroup=true;
    option.n_layers=2;option.n_expert=4;option.n_embd=2560;option.n_ff=640;option.partition=f.setup.partition;
    std::string err;ExpertWorkerTier refused;auto bad=option;bad.resident_reuse_only=false;
    check(!refused.open(bad,err),"regroup cannot cold-load or fall back");
    check(!ExpertWorkerGroup::validate({option,option},f.setup.partition,err),"regroup refuses multiple endpoints");
    ExpertWorkerGroup group;require(group.open({option},f.setup.partition,err),err);
    check(group.peer_stats(0).cache_hit && !group.peer_stats(0).local_bytes &&
        !group.peer_stats(0).setup_network_bytes,"regroup reuses the only arena without reads/uploads");
    const auto owners=cpu_capture_owners(f.setup);
    uint64_t expected[2]{};
    for(int mode=0;mode<5;++mode) {
        auto q=f.records[mode<3?mode:0].request;
        if(mode>=3) {q.T=1;q.K=1;q.ids={uint16_t(mode-3)};q.mask={1};q.weights={-.25f};q.activations.resize(2560);}
        std::vector<float> combined;bool first=true;
        for(int p=0;p<2;++p) {
            auto mask=q.mask;bool active=false;
            std::vector<const uint8_t*> blobs(q.ids.size());
            for(size_t i=0;i<mask.size();++i) {mask[i]=mask[i] && owners[q.layer*4+q.ids[i]]==p;
                active=active||mask[i];blobs[i]=fixture.blobs[q.layer*4+q.ids[i]].data();}
            if(!active)continue;++expected[p];std::vector<float> part;
            require(ExpertWorkerService::direct_native_partial(fixture.fmt,q.T,q.K,mask.data(),q.weights.data(),
                q.activations.data(),blobs.data(),part,err),err);
            if(first){combined=part;first=false;}else for(size_t i=0;i<combined.size();++i)combined[i]=combined[i]+part[i];
        }
        std::vector<int32_t> ids(q.ids.begin(),q.ids.end());
        require(group.post(q.layer,q.activations.data(),ids.data(),q.weights.data(),q.mask.data(),q.T,q.K,err) && group.finish(err),err);
        check(group.served() && !std::memcmp(combined.data(),group.partials(),combined.size()*sizeof(float)),
            "live tier/group bitmatches independent ordered masked reference across windows");
    }
    check(group.peer_stats(0).regroup_peer_requests[0]==expected[0] &&
        group.peer_stats(0).regroup_peer_requests[1]==expected[1],"actual wire counts include only active virtual peers");
    // Peer0 succeeds; peer1's nine repeats exceed the existing worker limit of eight.
    std::vector<int32_t> ids(10,1);ids[0]=0;std::vector<uint8_t> mask(10,1);std::vector<float> weights(10,.1f),x(2560,.2f);
    require(group.post(0,x.data(),ids.data(),weights.data(),mask.data(),1,10,err),err);
    check(!group.finish(err) && group.dead() && !group.served() && !group.partials(),
        "real second-peer refusal drains and publishes no partial result");
    group.close();
}
void live_ordered(const Fixture& fixture,const CpuCaptureFile& f,const Child& child) {
    ExpertWorkerGroup::Open option;option.local=&fixture.m;option.port=child.port;
    option.resident=option.resident_reuse_only=option.diagnostic_ordered=true;
    option.n_layers=2;option.n_expert=4;option.n_embd=2560;option.n_ff=640;option.partition=f.setup.partition;
    ExpertWorkerGroup group;std::string err;require(group.open({option},f.setup.partition,err),err);
    check(group.peer_stats(0).cache_hit && !group.peer_stats(0).local_bytes && !group.peer_stats(0).setup_network_bytes,
          "ordered uses the existing complete arena without load/upload");
    uint64_t active=0;
    for(const auto& record:f.records) {
        const auto& q=record.request;std::vector<int32_t> ids(q.ids.begin(),q.ids.end());
        require(group.post(q.layer,q.activations.data(),ids.data(),q.weights.data(),q.mask.data(),q.T,q.K,err) && group.finish(err),err);
        check(group.served()==record.sent,"ordered preserves zero-work semantics");
        if(record.sent) {++active;check(!std::memcmp(group.partials(),record.partials.data(),record.partials.size()*sizeof(float)),
            "real shadow/row reconstruction is bitwise full reference across live windows");}
    }
    const auto& stats=group.peer_stats(0);
    check(stats.ordered_checked_windows==active && stats.ordered_wire_requests>=2*active &&
        stats.ordered_max_exchanges<=11 && stats.ordered_peak_row_bytes<=819200,"ordered checked-window and bounded wire/scratch receipts");
    group.close();
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
    live_ordered(fixture,f,child);
    check(unchanged_resident(before,cache(child)),"ordered disconnect preserves one complete resident arena");
    live_regroup(fixture,f,child);
    check(unchanged_resident(before,cache(child)),"regroup failure disconnect retains the same complete arena");
    check(child.owns_listener(),"fixture still owns its live listener");child.stop();
    int status=0;errno=0;check(::waitpid(-1,&status,WNOHANG)<0 && errno==ECHILD,"fixture child cleanly reaped");
}
}
int main(int argc,char** argv){
    if(argc!=2)return 2;
    try{ordered_checks();regroup_order();test(argv[1]);}catch(const std::exception& e){std::fprintf(stderr,"CPU_CAPTURE_GROUP exception: %s\n",e.what());return 2;}
    std::printf("CPU_CAPTURE_GROUP_TEST failures=%d\n",failures);return failures?1:0;
}
