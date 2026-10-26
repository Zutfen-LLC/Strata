// Real SGW1 clients/services/cache; controlled executor only. No CUDA or model.
#include "local_fixture.hpp"
#ifdef STRATA_TEST_TLS
#include "tls_fixture.hpp"
#endif
#include "strata/worker/gpu_group.hpp"
#include "strata/worker/gpu_worker.hpp"
#include "strata/worker/expert_worker.hpp"
#include "strata/core/cpu_first_remote_post.hpp"
#include <algorithm>
#include <condition_variable>
#include <cmath>
#include <mutex>
#include <limits>
#include <sys/resource.h>

namespace {
struct Gate {
    std::mutex mutex;std::condition_variable wake;
    int entered=0;bool cpu_started=false,release_cpu=false,release[3]{},fail=false;
    std::vector<int> completed;
    bool wait_entered(int count) {
        std::unique_lock<std::mutex> lock(mutex);
        return wake.wait_for(lock,std::chrono::seconds(2),[&]{return entered==count;});
    }
    void allow(int i) {std::lock_guard<std::mutex> lock(mutex);release[i]=true;wake.notify_all();}
    void allow_all() {std::lock_guard<std::mutex> lock(mutex);release_cpu=true;for(auto& r:release)r=true;wake.notify_all();}
};
struct Executor final : GpuExecutor {
    Gate& gate;int index,delay;
    Executor(Gate& g,int i,int d=0):gate(g),index(i),delay(d){}
    bool prepare(const GpuPlan& p,uint64_t,GpuCache& cache,GpuCounters& c,std::string& err) override {
        std::vector<uint8_t> bytes;for(const auto& a:p.assets)if(!cache.read(a,bytes,err))return false;
        if(delay)std::this_thread::sleep_for(std::chrono::milliseconds(delay));
        c.gpu_weight_bytes=p.assigned_bytes();c.gpu_allocated_bytes=p.assigned_bytes()+1024;
        return true;
    }
    bool infer(const GpuPlan& p,const InferRequest& r,InferReply& out,std::string& err) override {
        if(r.ids[0]!=0 || r.activations[0]!=0) {err="borrowed caller input changed";return false;}
        std::unique_lock<std::mutex> lock(gate.mutex);++gate.entered;gate.wake.notify_all();
        if(!gate.cpu_started || !gate.wake.wait_for(lock,std::chrono::seconds(3),[&]{return gate.release[index];})) {
            err="CPU not in flight or parallel request barrier timed out";return false;
        }
        if(gate.fail && index==1){err="controlled failed peer";return false;}
        gate.completed.push_back(index);gate.wake.notify_all();
        const float values[]={1e20f,-1e20f,1.f};
        out.partials.assign((size_t)r.T*p.n_embd,values[index]);return true;
    }
    bool supports_rows() const override { return true; }
    bool infer_rows(const GpuPlan& p,const InferRequest& r,EntryRowsReply& out,std::string& err) override {
        InferReply ignored;if(!infer(p,r,ignored,err))return false;
        for(size_t i=0;i<r.mask.size();++i)if(r.mask[i]) {
            out.indices.push_back((uint16_t)i);
            for(size_t h=0;h<p.n_embd;++h)out.rows.push_back(h==1?-0.f:ignored.partials[h]);
        }
        return true;
    }
    void release(GpuCounters& c) override {c.gpu_allocated_bytes=c.gpu_weight_bytes=0;}
};
struct Running {
    GpuWorkerService service;uint16_t port=0;std::thread thread;
    Running(GpuWorkerOptions o,std::unique_ptr<GpuExecutor> e):service(std::move(o),std::move(e)) {
        std::string err;require(service.listen(port,err),err);thread=std::thread([&]{service.run();});
    }
    void stop(){service.stop();if(thread.joinable())thread.join();}
    ~Running(){stop();}
};
LocalManifest metadata(const Fixture& f) {
    auto m=f.m;m.n_layers=1;m.n_expert=4;m.layers.resize(1);m.digests.clear();
    for(const auto& b:f.blobs)m.digests.push_back(blob_digest(b));return m;
}
struct Network {
    Fixture fixture;Gate gate;
    std::vector<std::unique_ptr<Running>> servers;
    std::vector<GpuWorkerTier::Open> options;
    int fetches=0;
    explicit Network(bool slow=false) {
        const auto m=metadata(fixture);std::string err;
        for(int i=0;i<3;++i) {
            GpuWorkerOptions o;require(make_gpu_plan(m,{{0,(uint16_t)i}},o.plan,err),err);
            o.cache_root=(fixture.root/("group-"+std::to_string(i))).string();fs::create_directory(o.cache_root);
            o.cache_budget=16ull<<20;o.deadline_ms=300;o.setup_ms=5000;
            auto server=std::make_unique<Running>(o,std::make_unique<Executor>(gate,i,slow&&i==2?800:0));
            GpuWorkerTier::Open in;in.port=server->port;in.plan=o.plan;in.deadline_ms=1000;in.setup_ms=5000;
            in.fetch=[this](int64_t,int64_t e,uint8_t* out,uint32_t n) {
                ++fetches;const auto& b=fixture.blobs.at((size_t)e);
                if(n!=b.size())return false;std::memcpy(out,b.data(),n);return true;
            };
            options.push_back(std::move(in));servers.push_back(std::move(server));
        }
    }
};
void configuration(const Network& n) {
    std::string err;check(GpuWorkerGroup::validate(n.options,err),"three distinct endpoint plans accepted");
    auto bad=n.options;bad[1].host=bad[0].host;bad[1].port=bad[0].port;
    check(!GpuWorkerGroup::validate(bad,err),"duplicate endpoint rejected before connection");
    bad=n.options;bad[1].plan=bad[0].plan;
    check(!GpuWorkerGroup::validate(bad,err),"duplicate execution owner rejected");
    bad=n.options;auto m=metadata(n.fixture);++m.identity.a;
    require(make_gpu_plan(m,{{0,1}},bad[1].plan,err),err);
    check(!GpuWorkerGroup::validate(bad,err),"different trusted model/layout rejected");
    bad.assign(kMaxGpuWorkers+1,n.options[0]);
    check(!GpuWorkerGroup::validate(bad,err)&&!GpuWorkerGroup::validate({},err),"bounded nonempty endpoint collection");
    bad=n.options;for(auto& o:bad)o.entry_rows=false;bad[0].vulkan=true;
    check(!GpuWorkerGroup::validate(bad,err),"Vulkan weighted mode rejected before connection");
    for(auto& o:bad)o.entry_rows=true;
    check(GpuWorkerGroup::validate(bad,err),"mixed CUDA/Vulkan indexed endpoint plans accepted");
    bad[0].host="10.0.0.5";
    check(!GpuWorkerGroup::validate(bad,err),"private Vulkan plaintext rejected before connection");
#ifdef STRATA_TEST_TLS
    TlsFixture tls_fixture;
    bad[0].tls=tls_fixture.config("coordinator","worker.test");
    check(GpuWorkerGroup::validate(bad,err),"private Vulkan explicit mTLS configuration accepted");
#endif
    bad[0].tls.key_file.clear();
    check(!GpuWorkerGroup::validate(bad,err),"partial Vulkan mTLS configuration rejected");
    GpuEndpoint parsed;
    check(parse_gpu_endpoint("vulkan@10.0.0.5:7432",parsed)&&parsed.vulkan,"Vulkan endpoint parser");
    check(parse_gpu_endpoint("cuda@10.0.0.5:7432",parsed)&&!parsed.vulkan,"explicit CUDA endpoint parser");
    check(parse_gpu_endpoint("127.0.0.1:7432",parsed)&&parsed.port==7432,"numeric endpoint parser");
    check(!parse_gpu_endpoint("localhost:7432",parsed)&&!parse_gpu_endpoint("127.0.0.1:0",parsed)&&
          !parse_gpu_endpoint("127.0.0.1:65536",parsed)&&!parse_gpu_endpoint("127.0.0.1:12x",parsed),"bad endpoint syntax refused");
}
void exchange(bool failure,bool rows=false) {
    Network n(true);for(auto& o:n.options)o.entry_rows=rows;configuration(n);GpuWorkerGroup group;std::string err;
    require(group.open(n.options,err),err);
    // Ready worker 0 survives worker 2's 800ms prepare and another 650ms idle,
    // both longer than the service's 300ms idle receive budget.
    std::this_thread::sleep_for(std::chrono::milliseconds(650));
    check(!group.dead()&&n.fetches==3,"ready peers kept alive during serial setup and idle");
    float x[2560]{};int32_t ids[]={0,1,2,3};float weights[]={.1f,.2f,.3f,.4f};uint8_t owned[]={1,1,1,0};
    std::thread cpu;
    struct Cleanup {Gate& gate;std::thread& cpu;~Cleanup(){gate.allow_all();if(cpu.joinable())cpu.join();}} cleanup{n.gate,cpu};
    n.gate.fail=failure;
    const bool submitted=strata::core::cpu_first_remote_post([&] {
        cpu=std::thread([&] {
            std::unique_lock<std::mutex> lock(n.gate.mutex);n.gate.cpu_started=true;n.gate.wake.notify_all();
            n.gate.wake.wait_for(lock,std::chrono::seconds(4),[&]{return n.gate.release_cpu;});
        });
        std::unique_lock<std::mutex> lock(n.gate.mutex);
        return n.gate.wake.wait_for(lock,std::chrono::seconds(1),[&]{return n.gate.cpu_started;});
    },[&]{return group.post(0,x,ids,weights,owned,1,4,err);},[&]{n.gate.allow_all();if(cpu.joinable())cpu.join();});
    require(submitted,err);
    // Mutating caller buffers cannot affect the already-owned requests.
    ids[0]=3;weights[0]=99;x[0]=99;
    require(n.gate.wait_entered(3),"all three real SGW1 exchanges entered before any reply/CPU release");
    for(int i:{2,0,1}) {
        n.gate.allow(i);
        if(!(failure&&i==1)) {
            std::unique_lock<std::mutex> lock(n.gate.mutex);
            require(n.gate.wake.wait_for(lock,std::chrono::seconds(1),[&]{
                return std::find(n.gate.completed.begin(),n.gate.completed.end(),i)!=n.gate.completed.end();}),"controlled completion");
        }
    }
    n.gate.allow_all();if(cpu.joinable())cpu.join();
    const bool finished=group.finish(err);
    if(failure) {
        check(!finished&&group.dead()&&group.size()==0,"failed peer drains/closes every owner and exposes no partials");
        ids[0]=0;check(!group.post(0,x,ids,weights,owned,1,4,err),"failed group cannot retry implicitly");
    } else {
        require(finished,err);
        check(n.gate.completed==std::vector<int>({2,0,1}),"replies deliberately complete out of configuration order");
        check(group.mask(0)[0]&&!group.mask(0)[3]&&group.mask(2)[2],"copied disjoint masks exclude CPU share");
        if(rows) {
            std::vector<float> dense(4*2560), expected(2560), actual(2560);
            const float values[]={1e20f,-1e20f,1.f};
            for(size_t p=0;p<3;++p) {
                const auto& reply=group.entry_rows(p);
                require(reply.indices==std::vector<uint16_t>{(uint16_t)p},"original sparse entry index");
                std::vector<float> reference(2560,values[p]);reference[1]=-0.f;
                check(reply.rows.size()==reference.size() && !std::memcmp(reply.rows.data(),reference.data(),2560*4),
                      "backend rows byte-identical including negative zero");
                std::copy(reply.rows.begin(),reply.rows.end(),dense.begin()+p*2560);
            }
            const float original_weights[]={.1f,.2f,.3f,.4f};
            ExpertWorkerService::accumulate_partial(1,4,owned,original_weights,dense.data(),2560,actual.data());
            for(int h=0;h<2560;++h) {
                float acc=0.f;
                for(int e=0;e<3;++e) {const float product=original_weights[e]*(h==1?-0.f:values[e]);acc=acc+product;}
                expected[h]=acc;
            }
            check(!std::memcmp(actual.data(),expected.data(),2560*4),"ordered reducer equals independent original-order reference");
            std::vector<float> cpu(4*2560,0.f),mixed;
            std::fill(cpu.begin()+3*2560,cpu.end(),.125f);
            std::vector<uint8_t> cpu_mask{0,0,0,1},full{1,1,1,1};
            require(group.reduce_rows(cpu,cpu_mask,full,original_weights,1,4,mixed,err),err);
            for(int h=0;h<2560;++h)expected[h]=expected[h]+original_weights[3]*.125f;
            check(!std::memcmp(mixed.data(),expected.data(),2560*4),"mixed CPU/GPU union reduced once in original order");
            auto gap=cpu_mask;gap[3]=0;
            check(!group.reduce_rows(cpu,gap,full,original_weights,1,4,mixed,err)&&mixed.empty(),"mixed gap refuses before commit");
            auto overlap=cpu_mask;overlap[0]=1;
            check(!group.reduce_rows(cpu,overlap,full,original_weights,1,4,mixed,err)&&mixed.empty(),"mixed duplicate ownership refuses before commit");
            auto foreign=full;foreign[0]=0;
            check(!group.reduce_rows(cpu,cpu_mask,foreign,original_weights,1,4,mixed,err),"mixed foreign GPU row refuses");
            cpu[3*2560]=std::numeric_limits<float>::infinity();
            check(!group.reduce_rows(cpu,cpu_mask,full,original_weights,1,4,mixed,err),"mixed nonfinite CPU row refuses");
            int32_t cpu_ids[]={3,3,3,3};uint8_t no_gpu[]={0,0,0,0};x[0]=0;
            require(group.post(0,x,cpu_ids,original_weights,no_gpu,1,4,err)&&group.finish(err),err);
            cpu.assign(4*2560,.125f);cpu_mask=full;
            require(group.reduce_rows(cpu,cpu_mask,full,original_weights,1,4,mixed,err),err);
            check(mixed.size()==2560,"CPU-only window clears prior GPU rows before mixed assembly");
        } else check(group.partials(0)[0]==1e20f&&group.partials(1)[0]==-1e20f&&group.partials(2)[0]==1.f,
              "configuration-indexed results independent of arrival order");
        group.close();
    }
    for(auto& s:n.servers) {s->stop();check(s->service.counters().gpu_allocated_bytes==0,"owned executor released after group close/failure");}
}
void single_and_setup_failure() {
    Network n;std::string err;GpuWorkerGroup group;auto one=n.options;one.resize(1);
    require(group.open(one,err),err);
    {std::lock_guard<std::mutex> lock(n.gate.mutex);n.gate.cpu_started=true;n.gate.release[0]=true;}
    float x[2560]{};int32_t ids[]={0,3};float weights[]={1,1};uint8_t mask[]={1,0};
    require(group.post(0,x,ids,weights,mask,1,2,err)&&group.finish(err),err);
    check(group.size()==1&&group.partials()[0]==1e20f,"single-worker request/partial behavior");group.close();
    // Worker 0 is healthy but a later connection cannot complete. Its owner must
    // be joined/closed too, rather than leaking keepalives after failed open.
    GpuWorkerGroup broken;auto opts=n.options;opts[1].cancelled=[]{return true;};
    check(!broken.open(opts,err)&&broken.dead()&&broken.size()==0,"partial setup failure closes previously ready workers");
}
// The same bounded synthetic inputs can exercise three real worker processes
// over approved relays. Plans cannot name real weights: compare to our generated
// metadata before uploading anything. This mode starts no local worker/engine.
void external(int argc,char** argv) {
    Fixture f;const auto m=metadata(f);std::string err;
    std::vector<GpuPlan> plans(3);
    for(int i=0;i<3;++i)require(make_gpu_plan(m,{{0,(uint16_t)i}},plans[i],err),err);
    if(argc==3 && std::string(argv[1])=="--emit-plans") {
        const fs::path dir=argv[2];require(fs::create_directory(dir),"plan output must be a new owned directory");
        for(int i=0;i<3;++i)require(write_gpu_plan((dir/("peer-"+std::to_string(i)+".plan")).string(),plans[i],err),err);
        std::printf("GPU_WORKER_GROUP_PLANS_V1 peers=3 bytes_per_peer=%llu synthetic=1\n",
                    (unsigned long long)plans[0].assigned_bytes());return;
    }
    require(argc==8 && std::string(argv[1])=="--peers","usage: --emit-plans NEW_DIR | --peers HOST:PORT PLAN HOST:PORT PLAN HOST:PORT PLAN");
    std::vector<GpuWorkerTier::Open> options(3);
    for(int i=0;i<3;++i) {
        GpuEndpoint endpoint;require(parse_gpu_endpoint(argv[2+i*2],endpoint),"invalid numeric endpoint");
        auto& o=options[i];o.host=endpoint.host;o.port=endpoint.port;o.deadline_ms=10000;o.setup_ms=30000;
        require(read_gpu_plan(argv[3+i*2],o.plan,err),err);
        std::vector<uint8_t> got,expected;
        require(encode_gpu_plan(o.plan,got,err)&&encode_gpu_plan(plans[i],expected,err)&&got==expected,
                "external peer plan must match this synthetic fixture exactly");
        o.fetch=[&f](int64_t l,int64_t e,uint8_t* dst,uint32_t n) {
            if(l!=0||e<0||e>=3||n!=f.blobs[(size_t)e].size())return false;
            std::memcpy(dst,f.blobs[(size_t)e].data(),n);return true;
        };
    }
    GpuWorkerGroup group;require(group.open(options,err),err);
    std::vector<float> x(2560);for(size_t i=0;i<x.size();++i)x[i]=.01f*(int(i%17)-8);
    int32_t ids[]={0,1,2,3};float weights[]={.1f,.2f,.3f,.4f};uint8_t owned[]={1,1,1,0};
    const auto began=std::chrono::steady_clock::now();
    for(int r=0;r<3;++r) {
        require(group.post(0,x.data(),ids,weights,owned,1,4,err)&&group.finish(err),err);
        for(size_t p=0;p<3;++p) {
            require(group.active(p)&&group.mask(p)[p],"each peer served its exclusive assignment");
            bool nonzero=false;
            for(int h=0;h<2560;++h) {require(std::isfinite(group.partials(p)[h]),"finite CUDA partial");nonzero|=group.partials(p)[h]!=0;}
            require(nonzero,"nonzero CUDA contribution from each peer");
        }
    }
    const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count();
    std::string report;group.report(report);std::printf("%s\nGPU_WORKER_GROUP_REMOTE_V1 peers=3 windows=3 elapsed_ms=%.3f models=0 GEN=0\n",report.c_str(),ms);
    group.close();
}
}
int main(int argc,char** argv) {
    const rlimit memory{768ull<<20,768ull<<20},cpu{20,20},file{128ull<<20,128ull<<20};
    if(setrlimit(RLIMIT_AS,&memory)||setrlimit(RLIMIT_CPU,&cpu)||setrlimit(RLIMIT_FSIZE,&file))return 2;
    alarm(argc==1?40:120);
    try {if(argc>1)external(argc,argv);else {exchange(false);exchange(true);exchange(false,true);exchange(true,true);single_and_setup_failure();}}
    catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());++failures;}
    if(argc==1)std::printf("GPU_WORKER_GROUP_V1 failures=%d endpoints=3 models=0 cuda=0\n",failures);
    else std::printf("GPU_WORKER_GROUP_EXTERNAL_V1 failures=%d client_cuda=0 remote_cuda=%d models=0 GEN=0\n",
                     failures,argc==8?1:0);
    return failures?1:0;
}
