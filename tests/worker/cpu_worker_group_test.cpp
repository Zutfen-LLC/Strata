// Fixed CPU groups against real native services/manifest fixtures; no GPU/model.
#include "local_fixture.hpp"
#include "strata/worker/cpu_group.hpp"
#include "strata/worker/packed_activation.hpp"
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <algorithm>
namespace {
struct Gate {
    std::mutex mutex; std::condition_variable wake;
    int entered=0; bool released[2]{}; std::vector<int> completed;
    bool wait(int n) {std::unique_lock<std::mutex> lock(mutex);return wake.wait_for(lock,std::chrono::seconds(3),[&]{return entered>=n;});}
    void allow(int i) {std::lock_guard<std::mutex> lock(mutex);released[i]=true;wake.notify_all();}
    void all() {allow(0);allow(1);}
};
// Transparent bounded SEW1 proxy. Only INFER may be held/reordered/refused; all setup/digest/native
// work still uses the real service. Each proxy thread alone owns its descriptors.
struct Proxy {
    Gate& gate; int index; uint16_t backend,port=0; sock_fd listener=kBadFd;
    std::atomic<bool> stopping{false}; std::thread thread;
    std::atomic<bool> drop{false},nonfinite{false},overflow{false},refuse_rows{false},wrong_sequence{false},old_status{false},bad_caps{false};
    std::atomic<int> setups{0};
    Proxy(Gate& g,int i,uint16_t back):gate(g),index(i),backend(back) {
        std::string err;require(sock::listen_loopback(port,listener,err),err);
        thread=std::thread([this]{run();});
    }
    static bool frame(sock_fd fd,std::vector<uint8_t>& bytes) {
        bytes.resize(kHeaderBytes);
        if(!sock::recv_all(fd,bytes.data(),bytes.size(),5000))return false;
        uint64_t n=0;for(int i=0;i<8;++i)n|=(uint64_t)bytes[8+i]<<(8*i);
        if(n>kMaxFrameBytes)return false;
        bytes.resize(kHeaderBytes+(size_t)n);
        return !n || sock::recv_all(fd,bytes.data()+kHeaderBytes,(size_t)n,5000);
    }
    void run() {
        while(!stopping.load()) {
            sock_fd front=kBadFd,back=kBadFd;
            if(!sock::accept_one(listener,front,100))continue;
            std::string err;
            if(!sock::connect_host("127.0.0.1",backend,3000,back,err)) {sock::close_fd(front);continue;}
            std::vector<uint8_t> request,response;
            while(!stopping.load() && frame(front,request)) {
                const uint16_t msg=request[4]|(request[5]<<8);
                const bool infer=msg==MSG_INFER || msg==MSG_INFER_ROWS || msg==MSG_INFER_PACKED || msg==MSG_INFER_ROWS_PACKED;
                const bool rows=msg==MSG_INFER_ROWS || msg==MSG_INFER_ROWS_PACKED;
                const bool probe=msg==MSG_STATUS && request.size()==kHeaderBytes+16;
                if(probe && old_status)encode_frame(request,MSG_STATUS,ST_OK,nullptr,0);
                if(msg==MSG_LOCAL_SETUP && refuse_rows && request.size()>=kHeaderBytes+8 && (request[kHeaderBytes+4]&kLocalEntryRows)) {
                    ++setups;std::vector<uint8_t> refused;encode_frame(refused,MSG_LOCAL_SETUP_ACK,ST_LOCAL_UNAVAILABLE,nullptr,0);
                    sock::send_all(front,refused.data(),refused.size(),3000);break; // retained old-worker flag refusal
                }
                if(infer) {
                    std::unique_lock<std::mutex> lock(gate.mutex);++gate.entered;gate.wake.notify_all();
                    if(!gate.wake.wait_for(lock,std::chrono::seconds(3),[&]{return gate.released[index];}))break;
                    if(drop)break;
                }
                if(!sock::send_all(back,request.data(),request.size(),3000) || !frame(back,response))break;
                if(probe && bad_caps && response.size()==kHeaderBytes+20)response.back()^=1;
                if(infer && (nonfinite || overflow) && response.size()>=kHeaderBytes+26) {
                    // INFER_ACK starts session/sequence then F32 partials. Preserve framing/identity.
                    const uint32_t nan=overflow?0x7f7fffffu:0x7fc00000u;
                    const size_t offset=rows?22:16;
                    for(int i=0;i<4;++i)response[kHeaderBytes+offset+i]=(uint8_t)(nan>>(8*i));
                }
                if(rows && wrong_sequence)response[kHeaderBytes+8]^=1;
                if(!sock::send_all(front,response.data(),response.size(),3000))break;
                if(infer) {std::lock_guard<std::mutex> lock(gate.mutex);gate.completed.push_back(index);gate.wake.notify_all();}
            }
            sock::close_fd(front);sock::close_fd(back);
        }
    }
    ~Proxy() {stopping.store(true);gate.all();if(thread.joinable())thread.join();sock::close_fd(listener);}
};
ExpertWorkerTier::Open option(const Fixture& f,uint16_t port,std::vector<uint8_t> bits) {
    ExpertWorkerTier::Open o;o.port=port;o.local=&f.m;o.resident=true;o.setup_ms=5000;o.deadline_ms=3000;
    o.n_layers=2;o.n_expert=2;o.n_embd=2560;o.n_ff=640;o.gu_type=o.d_type=8;o.partition=std::move(bits);
    return o;
}
void configuration(Fixture& f,Service& a,Service& b) {
    auto opts=std::vector<ExpertWorkerTier::Open>{option(f,a.port,{1,1}),option(f,b.port,{2,2})};
    std::string err;check(ExpertWorkerGroup::validate(opts,{3,3},err),"two exhaustive disjoint trusted partitions");
    auto bad=opts;bad[1].port=bad[0].port;
    check(!ExpertWorkerGroup::validate(bad,{3,3},err),"duplicate endpoint refused before connecting");
    bad=opts;bad[1].partition={1,1};check(!ExpertWorkerGroup::validate(bad,{3,3},err),"overlapping ownership refused");
    bad=opts;bad[1].partition={0,2};check(!ExpertWorkerGroup::validate(bad,{3,3},err),"missing ownership refused");
    bad=opts;bad[0].partition={1};check(!ExpertWorkerGroup::validate(bad,{3,3},err),"malformed bitmap refused");
    bad=opts;bad[0].partition={5,1};check(!ExpertWorkerGroup::validate(bad,{3,3},err),"non-CPU/padding ownership refused");
    auto other=f.m;++other.identity.a;bad=opts;bad[1].local=&other;
    check(!ExpertWorkerGroup::validate(bad,{3,3},err),"different trusted model refused");
    bad=opts;bad[1].n_ff=641;check(!ExpertWorkerGroup::validate(bad,{3,3},err),"different geometry refused");
    bad=opts;bad[1].entry_rows=true;check(!ExpertWorkerGroup::validate(bad,{3,3},err),"mixed reply modes refused");
    check(!ExpertWorkerGroup::validate({}, {3,3},err),"empty endpoint list refused");
    bad.push_back(opts[0]);check(!ExpertWorkerGroup::validate(bad,{3,3},err),"endpoint bound enforced");
    CpuEndpoint endpoint;check(parse_cpu_endpoint("127.0.0.1:1234",endpoint) &&
        !parse_cpu_endpoint("localhost:1234",endpoint) && !parse_cpu_endpoint("127.0.0.1:0",endpoint),"CPU endpoint numeric contract");
}
struct Input {
    int T,K=2;std::vector<int32_t> ids;std::vector<float> x,w;std::vector<uint8_t> mask;
    explicit Input(int tokens,int ranks=2):T(tokens),K(ranks),ids(T*K),x(T*2560),w(T*K),mask(T*K,1) {
        for(size_t i=0;i<x.size();++i)x[i]=0.1f*(int(i%11)-5);
        const float weights[]={.7f,-.3f,.2f,.4f};
        for(int i=0;i<T*K;++i) {ids[i]=i%2;w[i]=weights[i%K];}
    }
    bool post(ExpertWorkerGroup& g,std::string& err) {return g.post(0,x.data(),ids.data(),w.data(),mask.data(),T,K,err);}
};
void reference(Fixture& f,ExpertWorkerGroup& group,const Input& in,int count,bool rows=false) {
    std::string err;std::vector<float> grouped,unsplit;
    std::vector<const uint8_t*> pointers(in.ids.size());
    for(size_t i=0;i<pointers.size();++i)pointers[i]=f.blobs[in.ids[i]].data();
    for(int p=0;p<count;++p) {
        auto mask=in.mask;
        for(size_t i=0;i<mask.size();++i)mask[i]=mask[i] && (count==1 || in.ids[i]==p);
        std::vector<float> partial;
        require(ExpertWorkerService::direct_native_partial(f.fmt,in.T,in.K,mask.data(),in.w.data(),in.x.data(),pointers.data(),partial,err),err);
        if(p==0)grouped=partial;else for(size_t i=0;i<grouped.size();++i)grouped[i]=grouped[i]+partial[i];
    }
    require(ExpertWorkerService::direct_native_partial(f.fmt,in.T,in.K,in.mask.data(),in.w.data(),in.x.data(),pointers.data(),unsplit,err),err);
    check(!std::memcmp((rows?unsplit:grouped).data(),group.partials(),grouped.size()*sizeof(float)),
          rows?"entry rows bitmatch independent unsplit original order":"bitwise configuration-ordered grouped native reference");
    std::vector<double> magnitude(grouped.size(),0.0);
    for(size_t entry=0;entry<in.ids.size();++entry) {
        std::vector<uint8_t> one(in.mask.size(),0);one[entry]=in.mask[entry];std::vector<float> contribution;
        require(ExpertWorkerService::direct_native_partial(f.fmt,in.T,in.K,one.data(),in.w.data(),in.x.data(),pointers.data(),contribution,err),err);
        for(size_t i=0;i<magnitude.size();++i)magnitude[i]+=std::abs((double)contribution[i]);
    }
    double maximum=0,relative=0,scaled=0,squared=0;
    for(size_t i=0;i<grouped.size();++i) {
        const double got=group.partials()[i],want=unsplit[i],delta=std::abs(got-want),scale=std::max(1.0,magnitude[i]);
        require(std::isfinite(got)&&std::isfinite(want),"finite grouped/independent unsplit result");
        maximum=std::max(maximum,delta);relative=std::max(relative,delta/std::max(1e-12,std::abs(want)));scaled=std::max(scaled,delta/scale);squared+=delta*delta;
        // Frozen before execution: 4e-6 (~32 float eps) times the independently computed sum of
        // absolute weighted contributions covers <=10-term regrouping, including cancellation.
        // The absolute floor is 1e-4. Existing single-tier bitwise gates remain untouched.
        check(delta<=1e-4+4e-6*magnitude[i],"unsplit envelope 1e-4 + 4e-6*sum_abs_contributions");
    }
    std::printf("CPU_GROUP_NUMERIC T=%d peers=%d max_abs=%.9g max_rel=%.9g max_scaled_rel=%.9g rmse=%.9g atol=1e-4 scaled_rtol=4e-6\n",
        in.T,count,maximum,relative,scaled,std::sqrt(squared/grouped.size()));
}
void native(bool rows=false,int packed_mode=0) {
    Fixture f;auto settings=f.options();settings.residency_forever=true;
    if(packed_mode)settings.packed_qualification="123456789abcdef1";
    Service a(settings);if(packed_mode==2)settings.packed_qualification.clear();
    Service b(settings);configuration(f,a,b);std::string err;
    for(int count:{1,2}) for(int round=0;round<2;++round) {
        auto opts=std::vector<ExpertWorkerTier::Open>{option(f,a.port,count==1?std::vector<uint8_t>{3,3}:std::vector<uint8_t>{1,1})};
        if(count==2)opts.push_back(option(f,b.port,{2,2}));
        for(auto& o:opts){o.entry_rows=rows;if(packed_mode){o.packed_qualification="123456789abcdef1";o.rpc_timing=true;}}
        // A changed partition must not silently reuse the previous resident arena.
        if(round==0) {ExpertWorkerClient c;connect(c,a);require(c.unload(err),err);c.close();connect(c,b);require(c.unload(err),err);}
        ExpertWorkerGroup group;require(group.open(opts,{3,3},err),err);
        if(round)check(group.peer_stats(0).cache_hit,"reconnect reuses matching frozen resident subset");
        for(int T:{8,1,2,1}) {Input in(T);require(in.post(group,err),err);require(group.finish(err),err);reference(f,group,in,count,rows);}
        for(int T:{1,2,4}) {Input in(T,4);require(in.post(group,err)&&group.finish(err),err);reference(f,group,in,count,rows);}
        if(count==2) {
            Input one(1);one.ids={0,0};one.mask={1,1};one.w={.5f,.5f};
            auto before=group.peer_stats(1).requests;
            require(one.post(group,err)&&group.finish(err),err);
            check(group.peer_stats(1).requests==before,"zero-work peer sends no INFER");
        }
        if(packed_mode) {
            check(group.peer_stats(0).packed_requests>0 && group.peer_stats(0).raw_fallbacks>0,
                  "T1/T2 packed and unsupported T4/T8 deliberate raw fallback");
            check(group.peer_stats(0).pack_calls==0 && (count==1 || group.peer_stats(1).pack_calls==0),
                  "group peers never repeat shared sender quantization");
            if(count==2 && packed_mode==2)check(group.peer_stats(1).packed_requests==0 && group.peer_stats(1).raw_fallbacks>0,
                "unqualified peer receives raw alongside packed capable peer");
            std::string report;group.report(report);
            check(report.find(count==1?"pack_calls=7 quantized_rows=7":"pack_calls=8 quantized_rows=8")!=std::string::npos,
                "group quantizes active union exactly once, independent of eligible peer count");
            std::printf("PACKED_PLACEMENT %s\n",report.c_str());
            Input borrowed(1);require(borrowed.post(group,err),err);
            borrowed.x.assign(2560,99.f);borrowed.ids.assign(2,0);borrowed.w.assign(2,99.f);
            require(group.finish(err),err);Input immutable(1);reference(f,group,immutable,count,rows);
        }
        group.close();
    }
}
void default_selection() {
    check(WorkerOptions{}.packed_requests && ExpertWorkerTier::Open{}.packed_requests,
          "automatic preference is the default independently of qualification");
    Fixture f;
    for(bool rows:{false,true}) for(int mode:{0,1,2,3}) {
        auto settings=f.options();settings.residency_forever=true;
        if(mode==1 || mode==2)settings.packed_qualification="123456789abcdef1";
        if(mode==2)settings.packed_requests=false;
        if(mode==3)settings.packed_qualification.clear();
        Service service(settings);
        auto o=option(f,service.port,{3,3});o.entry_rows=rows;
        check(o.packed_qualification==default_packed_qualification(),"sender inherits build eligibility");
        if(mode==1 || mode==2)o.packed_qualification=settings.packed_qualification;
        if(mode==1)o.packed_requests=false;
        if(mode==3)o.packed_qualification.clear();
        ExpertWorkerGroup group;std::string err;require(group.open({o},{3,3},err),err);
        Input in(2);require(in.post(group,err) && group.finish(err),err);reference(f,group,in,1,rows);
        const bool expected=mode==0 && !default_packed_qualification().empty();
        check(group.peer_stats(0).packed_requests==(expected?1u:0u),
              "qualified defaults pack without runtime opt-in; raw override/unqualified stay raw");
        check(group.peer_stats(0).raw_fallbacks==((mode==2)?1u:0u),
              "receiver raw override declines capability; sender override skips packing entirely");
        std::string report;group.report(report);
        if(!expected)check(report.find("quantized_rows=0")!=std::string::npos ||
                           report.find("quantized_rows=")==std::string::npos,
                           "raw selection does no sender quantization");
        group.close();
    }
}
void controlled(int mode,bool rows=false) {
    Fixture f;auto settings=f.options();settings.residency_forever=true;settings.deadline_ms=3000;
    Service a(settings);settings.deadline_ms=300;Service b(settings);Gate gate;
    Proxy pa(gate,0,a.port),pb(gate,1,b.port);pb.drop=mode==1;pb.nonfinite=mode==2;
    pa.overflow=pb.overflow=mode==4;
    auto opts=std::vector<ExpertWorkerTier::Open>{option(f,pa.port,{1,1}),option(f,pb.port,{2,2})};
    for(auto& o:opts)o.entry_rows=rows;
    pb.wrong_sequence=mode==5;
    opts[1].deadline_ms=1000;ExpertWorkerGroup group;std::string err;require(group.open(opts,{3,3},err),err);
    struct Release {Gate& gate;~Release(){gate.all();}} release{gate};
    Input in(1);if(mode==3) {in.ids={0,0};in.w={.5f,.5f};}
    if(rows && mode==4)in.w={2.f,2.f}; // raw finite rows overflow only at the central weighted reduction
    require(in.post(group,err),err);require(gate.wait(mode==3?1:2),"all active peers reach INFER barrier before any reply");
    in.x[0]=99;in.ids[0]=1;in.w[0]=99; // copied input lifetime
    if(mode==3) {
        std::this_thread::sleep_for(std::chrono::milliseconds(700)); // inactive peer's service idle budget is 300ms
        gate.allow(0);require(group.finish(err),err);
        check(!group.dead()&&group.peer_stats(1).requests==0,"short-deadline zero-work peer kept alive during long valid work");
    } else {
        gate.allow(1);
        if(mode!=1) {std::unique_lock<std::mutex> lock(gate.mutex);require(gate.wake.wait_for(lock,std::chrono::seconds(2),[&]{return !gate.completed.empty();}),"peer1 replies first");}
        bool finished=false,result=false;std::mutex mutex;std::condition_variable wake;
        std::thread waiter([&]{result=group.finish(err);{std::lock_guard<std::mutex> lock(mutex);finished=true;}wake.notify_all();});
        struct Join {Gate& gate;std::thread& thread;~Join(){gate.all();if(thread.joinable())thread.join();}} join{gate,waiter};
        {std::unique_lock<std::mutex> lock(mutex);check(!wake.wait_for(lock,std::chrono::milliseconds(80),[&]{return finished;}),"finish drains still-blocked peer even after another failure");}
        gate.allow(0);waiter.join();
        if(mode)check(!result&&group.dead()&&group.partials()==nullptr,"failure/nonfinite exposes no group partial and latches dead");
        else {require(result,err);
            {std::unique_lock<std::mutex> lock(gate.mutex);
             require(gate.wake.wait_for(lock,std::chrono::seconds(2),[&]{return gate.completed.size()==2;}),"proxy completion receipts");
             check(gate.completed==std::vector<int>({1,0}),"reversed completion order retained independently of merge order");}
            Input expected(1);reference(f,group,expected,2,rows);}
    }
    group.close();
}
void setup_failure_and_zero_work(bool rows=false) {
    Fixture f;auto settings=f.options();settings.residency_forever=true;
    Service a(settings);auto small=settings;small.memory_budget=f.fmt.bytes-1;Service b(small);
    std::string err;ExpertWorkerGroup broken;
    auto first=option(f,a.port,{1,1}),second=option(f,b.port,{2,2});first.entry_rows=second.entry_rows=rows;
    check(!broken.open({first,second},{3,3},err)&&broken.dead(),
          "second-peer setup refusal drains/closes the acquired first peer");
    ExpertWorkerClient control;connect(control,a);WorkerCacheInfo cache;
    require(control.cache_status(cache,err),err);
    check(cache.complete && cache.resident_bytes==2*f.fmt.bytes,"failed group does not gratuitously unload verified first-peer weights");
    control.close();
    Service c(settings),d(settings);ExpertWorkerGroup empty;
    first=option(f,c.port,{1,0});second=option(f,d.port,{0,2});first.entry_rows=second.entry_rows=rows;
    require(empty.open({first,second},{1,2},err),err);
    Input in(1);in.ids={1,1};in.mask={0,0};
    require(in.post(empty,err)&&empty.finish(err),err);
    check(!empty.served()&&empty.partials()==nullptr&&empty.peer_stats(0).requests==0&&empty.peer_stats(1).requests==0,
          "global zero CPU work exposes no stale partial and sends no INFER");
}
void slow_second_setup() {
    Fixture f;auto settings=f.options();settings.residency_forever=true;settings.deadline_ms=300;Service a(settings),b(settings);
    auto first=option(f,a.port,{1,1}),second=option(f,b.port,{2,2});
    first.deadline_ms=second.deadline_ms=1000;bool delayed=false;
    second.cancelled=[&]{if(!delayed){delayed=true;std::this_thread::sleep_for(std::chrono::milliseconds(700));}return false;};
    ExpertWorkerGroup group;std::string err;require(group.open({first,second},{3,3},err),err);
    Input in(1);require(in.post(group,err)&&group.finish(err),err);
    check(delayed&&!group.dead(),"first READY peer kept alive while the second setup is delayed beyond idle deadline");
}
void row_codec() {
    InferRequest q;q.session=0x0807060504030201ull;q.sequence=9;q.T=2;q.K=2;q.mask={7,0,0,1};
    const float source[]={-0.f,1.f,99.f,99.f,99.f,99.f,-2.f,.5f};std::vector<uint8_t> bytes,expected;
    require(encode_infer_rows_ack(q,2,source,bytes),"encode bounded rows");
    put<uint64_t>(expected,q.session);put<uint64_t>(expected,q.sequence);put<uint16_t>(expected,1);put<uint16_t>(expected,2);
    put<uint16_t>(expected,0);put<uint32_t>(expected,0x80000000);put<uint32_t>(expected,0x3f800000);
    put<uint16_t>(expected,3);put<uint32_t>(expected,0xc0000000);put<uint32_t>(expected,0x3f000000);
    check(bytes==expected,"indexed row codec exact little-endian bytes including signed zero/nonzero mask normalization");
    EntryRowsReply reply;require(decode_infer_rows_ack(bytes.data(),bytes.size(),2,q,reply),"decode rows");
    check(reply.indices==std::vector<uint16_t>({0,3}) && std::signbit(reply.rows[0]) && reply.rows[3]==.5f,"exact requested row set/bits");
    for(int mode=0;mode<7;++mode) {
        auto bad=bytes;
        if(mode==0)bad[16]=2; // schema
        if(mode==1)bad[18]=1; // row count
        if(mode==2)bad[20]=1; // foreign index
        if(mode==3)bad[30]=0; // duplicate index
        if(mode==4)bad.push_back(0);
        if(mode==5)bad.pop_back();
        if(mode==6){bad[22]=0;bad[23]=0;bad[24]=0xc0;bad[25]=0x7f;}
        check(!decode_infer_rows_ack(bad.data(),bad.size(),2,q,reply) && reply.rows.empty() && reply.indices.empty(),
              "malformed/nonfinite row reply rejected without output");
    }
    InferReply legacy;legacy.session=q.session;legacy.sequence=q.sequence;legacy.partials={-0.f,1.f};
    auto old=encode_infer_ack(legacy);expected.resize(16);put<uint32_t>(expected,0x80000000);put<uint32_t>(expected,0x3f800000);
    check(old==expected,"legacy weighted reply bytes unchanged");
    q.mask.assign(4,0);require(encode_infer_rows_ack(q,2,nullptr,bytes),"empty row reply");
    require(decode_infer_rows_ack(bytes.data(),bytes.size(),2,q,reply),"decode empty rows");
    check(bytes.size()==20 && reply.rows.empty() && reply.indices.empty(),"empty reply cannot expose prior rows");
    q.T=kMaxT;q.K=kMaxK;q.mask.assign(kMaxT*kMaxK,1);std::vector<float> maximum(kMaxT*kMaxK*kMaxNEmbd,0.f);
    require(encode_infer_rows_ack(q,kMaxNEmbd,maximum.data(),bytes),"encode maximum rows");
    check(bytes.size()==kEntryRowsPayloadMax && bytes.size()+kHeaderBytes==819396,"strict maximum row payload/frame bytes");
    require(decode_infer_rows_ack(bytes.data(),bytes.size(),kMaxNEmbd,q,reply),"decode maximum rows");
    check(reply.rows.capacity()<=kEntryRowsMax*kMaxNEmbd && reply.indices.capacity()<=kEntryRowsMax,"decoded row retention bounded after warmup");
}
void packed_capability() {
    Fixture f;auto settings=f.options();settings.residency_forever=true;
    settings.packed_qualification="123456789abcdef1";Service service(settings);Gate gate;
    for(bool malformed:{false,true}) {
        Proxy proxy(gate,0,service.port);proxy.old_status=!malformed;proxy.bad_caps=malformed;
        auto o=option(f,proxy.port,{3,3});o.packed_qualification=settings.packed_qualification;
        ExpertWorkerTier tier;std::string err;
        if(malformed)check(!tier.open(o,err) && err=="malformed packed capability reply",
            "malformed capability fails setup without raw fallback");
        else {
            require(tier.open(o,err),err);check(!tier.packed_enabled(),"old peer normal STATUS selects raw without reconnect");
            gate.all();Input in(1);require(tier.post(0,in.x.data(),in.ids.data(),in.w.data(),in.mask.data(),1,2,err) && tier.finish(err),err);
            check(tier.stats().packed_requests==0 && tier.stats().raw_fallbacks==1,"old peer still serves raw after capability probe");
        }
        tier.close();
    }
}
void row_capability() {
    Fixture f;auto settings=f.options();settings.residency_forever=true;Service service(settings);Gate gate;
    {
        Proxy old(gate,0,service.port);old.refuse_rows=true;
        auto o=option(f,old.port,{3,3});o.entry_rows=true;ExpertWorkerTier refused;std::string err;
        check(!refused.open(o,err) && err.find("entry-row capability unavailable")!=std::string::npos && old.setups.load()==1,
              "old-worker LOCAL_SETUP flag refusal is clear and never retries/falls back");
    }
    ExpertWorkerClient c;connect(c,service);auto m=f.setup();std::string err;
    require(c.setup_local(m,f.m.identity,false,err,true,true),err);
    std::vector<uint8_t> hits;require(c.load_local({{0,0},{0,1},{1,0},{1,1}},hits,err) && c.mark_ready(err),err);
    Input in(1);InferRequest q;q.session=c.session();q.model_id=m.model_id;q.partition_id=m.partition_id;
    q.T=1;q.K=2;q.ids={0,1};q.mask={1,1};q.weights=in.w;q.activations=in.x;q.sequence=1;
    InferReply legacy;EntryRowsReply rows;require(c.infer(q,legacy,err),err);
    check(!c.infer_rows(q,rows,err) && c.last_status()==ST_BAD_REQUEST,"sequence ordering shared across weighted and row messages");
    q.sequence=2;require(c.infer_rows(q,rows,err),err);
    std::vector<float> ordered(2560);ExpertWorkerService::accumulate_partial(1,2,q.mask.data(),q.weights.data(),rows.rows.data(),2560,ordered.data());
    check(!std::memcmp(ordered.data(),legacy.partials.data(),ordered.size()*sizeof(float)),"actual unweighted rows reproduce weighted reply bits");
    c.close();connect(c,service);require(c.setup_local(m,f.m.identity,false,err,true) && c.cache_hit() && c.mark_ready(err),err);
    WorkerStatusInfo before,after;require(c.status(before,err),err);
    check(!c.infer_rows(q,rows,err) && err=="entry-row capability was not negotiated","capability cleared on reconnect; old ACK accepted");
    require(c.status(after,err),err);check(before.requests==after.requests,"unnegotiated call sends no inference frame");
}
void float_codec_edges() {
    // Independent scalar LE words cover signed zero, subnormals, normal boundaries and max finite.
    const std::vector<uint32_t> words={0,0x80000000u,1,0x80000001u,0x007fffffu,0x00800000u,
                                     0x3f800000u,0xbf800000u,0x7f7fffffu,0xff7fffffu};
    const uint16_t H=(uint16_t)words.size();
    InferRequest q;q.session=17;q.sequence=19;q.T=1;q.K=2;q.mask={0,7};
    std::vector<float> rows(2*H);const uint32_t nan=0x7fc12345u;
    for(size_t i=0;i<H;++i){std::memcpy(&rows[i],&nan,4);std::memcpy(&rows[H+i],&words[i],4);}
    std::vector<uint8_t> bytes,expected;
    require(encode_infer_rows_ack(q,H,rows.data(),bytes),"finite selected rows ignore unselected NaNs");
    put<uint64_t>(expected,q.session);put<uint64_t>(expected,q.sequence);put<uint16_t>(expected,1);
    put<uint16_t>(expected,1);put<uint16_t>(expected,1);for(auto w:words)put<uint32_t>(expected,w);
    check(bytes==expected,"bulk row bytes equal independent scalar edge reference");
    std::vector<uint8_t> unaligned(1,0);unaligned.insert(unaligned.end(),bytes.begin(),bytes.end());
    EntryRowsReply reply;require(decode_infer_rows_ack(unaligned.data()+1,bytes.size(),H,q,reply),"unaligned row decode");
    check(reply.indices==std::vector<uint16_t>{1} && !std::memcmp(reply.rows.data(),rows.data()+H,H*4),
          "unaligned row decoding preserves all finite edge bits");
    for(uint32_t bad:{0x7f800000u,0xff800000u,0x7fc12345u}) {
        auto corrupt=bytes;for(int b=0;b<4;++b)corrupt[22+4*(H-1)+b]=(uint8_t)(bad>>(8*b));
        check(!decode_infer_rows_ack(corrupt.data(),corrupt.size(),H,q,reply) && reply.rows.empty() && reply.indices.empty(),
              "bulk row decoding still refuses infinity/NaN without output");
        std::memcpy(&rows.back(),&bad,4);
        check(!encode_infer_rows_ack(q,H,rows.data(),corrupt) && corrupt.empty(),"bulk row encoding still refuses infinity/NaN");
    }
    InferReply legacy;legacy.session=q.session;legacy.sequence=q.sequence;
    auto legacy_words=words;legacy_words.insert(legacy_words.end(),{0x7f800000u,0xff800000u,nan});
    legacy.partials.resize(legacy_words.size());std::memcpy(legacy.partials.data(),legacy_words.data(),legacy_words.size()*4);
    expected.clear();put<uint64_t>(expected,q.session);put<uint64_t>(expected,q.sequence);
    for(auto w:legacy_words)put<uint32_t>(expected,w);
    encode_infer_ack(legacy,bytes);check(bytes==expected,"legacy reply edge bytes including permitted nonfinite payloads unchanged");
    unaligned.assign(1,0);unaligned.insert(unaligned.end(),bytes.begin(),bytes.end());InferReply decoded;
    require(decode_infer_ack(unaligned.data()+1,bytes.size(),(uint16_t)legacy_words.size(),1,decoded),"unaligned legacy decode");
    check(!std::memcmp(decoded.partials.data(),legacy.partials.data(),legacy_words.size()*4),"legacy float edge payload bits unchanged");
    check(!decode_infer_ack(bytes.data(),bytes.size()-1,(uint16_t)legacy_words.size(),1,decoded),"legacy truncated reply refused");
}
}
int main(int argc,char** argv) {
    if(argc==2 && std::string(argv[1])=="--packed") {
        try {default_selection();packed_capability();native(false,1);native(true,1);native(true,2);}
        catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());++failures;}
        std::printf("CPU_PACKED_GROUP failures=%d models=0\n",failures);return failures?1:0;
    }
    try {row_codec();float_codec_edges();row_capability();native();native(true);setup_failure_and_zero_work();setup_failure_and_zero_work(true);slow_second_setup();
        for(int mode=0;mode<5;++mode)controlled(mode);for(int mode=0;mode<6;++mode)controlled(mode,true);}
    catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());++failures;}
    std::printf("CPU_WORKER_GROUP failures=%d models=0\n",failures);return failures?1:0;
}
