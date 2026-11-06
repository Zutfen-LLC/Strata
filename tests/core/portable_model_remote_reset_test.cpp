#include "portable_model_fixture.hpp"
#include "strata/worker/gpu_worker.hpp"
#include <thread>
using namespace strata::core;
using namespace strata::worker;
using model_fixture::check;
namespace {
struct FailExecutor final : GpuExecutor {
    bool native_q8() const override { return true; }
    bool supports_rows() const override { return true; }
    bool prepare(const GpuPlan& p,uint64_t,GpuCache& cache,GpuCounters& c,std::string& e) override {
        std::vector<uint8_t> bytes;
        for (const auto& a : p.assets) if (!cache.read(a,bytes,e)) return false;
        c.gpu_weight_bytes=c.gpu_allocated_bytes=p.assigned_bytes(); c.gpu_allocation_count=1; return true;
    }
    bool infer(const GpuPlan&,const InferRequest&,InferReply&,std::string&) override { return false; }
    bool infer_rows(const GpuPlan&,const InferRequest&,EntryRowsReply&,std::string& e) override { e="injected execution failure"; return false; }
    void release(GpuCounters& c) override { c.gpu_weight_bytes=c.gpu_allocated_bytes=c.gpu_allocation_count=0; }
};
struct Server {
    GpuWorkerService worker; std::thread thread; uint16_t port=0;
    explicit Server(GpuWorkerOptions o):worker(std::move(o),std::make_unique<FailExecutor>()) {
        std::string e; check(worker.listen(port,e),e); thread=std::thread([this]{ worker.run(); });
    }
    ~Server() { worker.stop(); if(thread.joinable()) thread.join(); }
};
}
int main() {
 try {
    model_fixture::SparseModel fixture;
    std::string e; GpuPlan plan; LocalManifest manifest;
    check(read_local_manifest((fixture.root/"manifest").string(),manifest,e),e);
    check(make_gpu_plan(manifest,{{0,0}},plan,e),e);
    GpuWorkerOptions worker; worker.plan=plan; worker.vulkan=worker.native_q8=true;
    worker.runtime_reserve=0; worker.cache_budget=64ull<<20;
    worker.cache_root=(fixture.root/"worker-cache").string();
    std::filesystem::create_directory(worker.cache_root);
    Server server(worker);
    PortableModelOptions options; options.context=1;
    GpuWorkerTier::Open remote; remote.port=server.port; remote.plan=plan;
    remote.vulkan=remote.native_q8=remote.entry_rows=true;
    options.remote_experts=remote;
    auto session=PortableModelSession::open(fixture.paths,(fixture.root/"manifest").string(),options,e);
    check(bool(session),e);
    const auto single_host_bound=session->report().host_state_bound;
    check(session->close_remote_experts(e),e);
    check(session->poisoned(),"closed remote must poison model");
    check(!session->reset(e) && session->poisoned() && e.find("reopen")!=std::string::npos,
          "reset of explicitly closed remote must require reopening");
    session.reset();
    session=PortableModelSession::open(fixture.paths,(fixture.root/"manifest").string(),options,e);
    check(bool(session),e);
    std::vector<float> logits;
    check(!session->step(5,logits,e) && session->poisoned() && logits.empty(),"remote execution error poisons model");
    check(session->remote_expert_stats().failures>0,"failure reached the remote tier");
    check(!session->reset(e) && session->poisoned() && e.find("reopen")!=std::string::npos,
          "reset of terminal remote must require reopening");
    session.reset();
    GpuPlan plan2;check(make_gpu_plan(manifest,{{0,1}},plan2,e),e);
    auto worker2=worker;worker2.plan=plan2;worker2.cache_root=(fixture.root/"worker-cache-2").string();
    std::filesystem::create_directory(worker2.cache_root);Server server2(worker2);
    auto peer2=remote;peer2.port=server2.port;peer2.plan=plan2;
    options.remote_experts.reset();options.remote_expert_peers={remote,peer2};
    session=PortableModelSession::open(fixture.paths,(fixture.root/"manifest").string(),options,e);
    check(bool(session),e);
    check(session->report().host_state_bound==single_host_bound+(64ull<<20),"second peer has separate host admission charge");
    check(!session->step(5,logits,e) && session->poisoned() && logits.empty(),"multi-peer execution failure poisons model atomically");
    check(!session->reset(e) && session->poisoned() && e.find("reopen")!=std::string::npos,
          "reset of terminal multi-peer group requires reopening");
    const auto peers=session->remote_expert_peer_stats();
    check(peers.size()==2 && peers[0].port==remote.port && peers[1].port==peer2.port,
          "failed group preserves per-peer identities");
    std::puts("portable_model_remote_reset_test: closed and failed remote sessions require reopen");
    return 0;
 } catch(const std::exception& e) { std::fprintf(stderr,"ERROR %s\n",e.what()); return 1; }
}
