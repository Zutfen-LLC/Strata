#include "strata/worker/gpu_group.hpp"
#include "mixed_rows.hpp"
#include "strata/worker/expert_worker.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>

namespace strata::worker {
struct GpuWorkerGroup::Impl {
    struct Request {
        int64_t layer=0; int T=0, K=0;
        std::vector<float> x, weights;
        std::vector<int32_t> ids;
        std::vector<uint8_t> mask;
    };
    struct Peer {
        GpuWorkerTier tier;
        std::mutex mutex;
        std::condition_variable wake;
        bool stopping=false, pending=false, busy=false;
        int interval=50;
        Request request;
        std::vector<float> output;
        EntryRowsReply rows;
        std::string error, report;
        std::thread thread;
    };
    std::vector<std::unique_ptr<Peer>> peers;
    std::vector<int> owners;
    std::vector<std::vector<uint8_t>> masks;
    std::vector<std::vector<float>> outputs;
    std::vector<uint8_t> active;
    std::vector<EntryRowsReply> rows;
    bool entry_rows=false;
    std::atomic<bool> failed{false};
    bool ready=false, posted=false, finished=false;
    int T=0,K=0;
    int layers=0, experts=0, H=0;
    std::string reason;

    int owner(int64_t l, int32_t e) const {
        return l>=0 && l<layers && e>=0 && e<experts ? owners[(size_t)l*experts+e] : -1;
    }
    std::string error() const {
        if (!reason.empty()) return reason;
        for (size_t i=0;i<peers.size();++i) {
            std::lock_guard<std::mutex> lock(peers[i]->mutex);
            if (!peers[i]->error.empty()) return "CUDA worker "+std::to_string(i)+": "+peers[i]->error;
        }
        return "CUDA group is not ready or a peer failed";
    }
    void run(Peer& p) {
        std::unique_lock<std::mutex> lock(p.mutex);
        for (;;) {
            if (!p.pending && !p.stopping && !failed.load())
                p.wake.wait_for(lock,std::chrono::milliseconds(p.interval),[&]{return p.pending || p.stopping || failed.load();});
            if (p.stopping || failed.load()) break;
            const bool job=p.pending;
            p.pending=false;
            lock.unlock();
            bool ok=false; std::string why, line; std::vector<float> output; EntryRowsReply rows;
            try {
                if (job) {
                    const auto& r=p.request;
                    ok=p.tier.post(r.layer,r.x.data(),r.ids.data(),r.weights.data(),r.mask.data(),r.T,r.K,why);
                    if (ok) ok=p.tier.finish(why);
                    if (ok && p.tier.served()) {
                        if(entry_rows) rows=p.tier.entry_rows();
                        else output.assign(p.tier.partials(),p.tier.partials()+p.tier.partial_count());
                    }
                    else if (ok) {ok=false;why="posted CUDA share was not served";}
                } else {
                    GpuCounters status; ok=p.tier.status(status,why);
                }
                p.tier.report(line);
            } catch (const std::exception& e) {ok=false;why=e.what();}
              catch (...) {ok=false;why="exception in CUDA communication thread";}
            lock.lock();
            p.report=std::move(line);
            if (!ok) {p.error=std::move(why);failed.store(true);}
            if (job) {p.output=std::move(output);p.rows=std::move(rows);p.busy=false;p.wake.notify_all();}
            if (!ok) break;
        }
        // Failed peers stop every socket owner. An exchange already in progress
        // completes/fails under its existing deadline before this point.
        p.pending=p.busy=false; p.wake.notify_all();
        lock.unlock(); p.tier.close();
    }
};

GpuWorkerGroup::GpuWorkerGroup():impl_(std::make_unique<Impl>()) {}
GpuWorkerGroup::~GpuWorkerGroup(){close();}
bool GpuWorkerGroup::validate(const std::vector<GpuWorkerTier::Open>& opts,std::string& err) {
    if (opts.empty() || opts.size()>kMaxGpuWorkers) {err="CUDA worker count must be 1..8";return false;}
    std::set<std::pair<std::string,uint16_t>> endpoints;
    std::set<std::pair<uint16_t,uint16_t>> owned;
    const auto& first=opts.front().plan;
    for (const auto& o:opts) {
        if (!gpu_numeric_ipv4(o.host) || !o.port || o.deadline_ms<1 || o.setup_ms<1 ||
            !endpoints.emplace(o.host,o.port).second || !validate_gpu_plan(o.plan,err)) {
            if(err.empty())err="invalid/duplicate CUDA endpoint or deadline";return false;
        }
        if(o.vulkan && (!o.entry_rows || (o.host!="127.0.0.1" && !o.tls.enabled()) ||
            o.plan.n_embd!=2560 || o.plan.n_ff!=640)) {
            err="Vulkan requires indexed rows, H2560/FF640 and mTLS outside loopback";return false;
        }
        if(!o.tls.validate(err))return false;
        const auto& p=o.plan;
        if (o.entry_rows!=opts.front().entry_rows) {err="CUDA group reply modes differ";return false;}
        if (p.model_id!=first.model_id || p.layout_id!=first.layout_id || p.n_layers!=first.n_layers ||
            p.n_expert!=first.n_expert || p.n_embd!=first.n_embd || p.n_ff!=first.n_ff ||
            p.gu_type!=first.gu_type || p.d_type!=first.d_type) {
            err="CUDA workers must share trusted model/layout/geometry";return false;
        }
        for (const auto& a:p.assets) if(!owned.emplace(a.layer,a.expert).second) {
            err="CUDA worker assignments overlap";return false;
        }
    }
    return true;
}
bool GpuWorkerGroup::open(const std::vector<GpuWorkerTier::Open>& opts,std::string& err) {
    auto& d=*impl_;
    if(d.failed.load()) {err=d.error();return false;}
    if(d.ready || !validate(opts,err)) {if(err.empty())err="CUDA group already open";return false;}
    d.entry_rows=opts[0].entry_rows;d.rows.resize(opts.size());
    d.layers=opts[0].plan.n_layers;d.experts=opts[0].plan.n_expert;d.H=opts[0].plan.n_embd;
    d.owners.assign((size_t)d.layers*d.experts,-1);
    d.masks.resize(opts.size());d.outputs.resize(opts.size());d.active.assign(opts.size(),0);
    try {
        for(size_t i=0;i<opts.size();++i) {
            auto p=std::make_unique<Impl::Peer>();
            auto option=opts[i];const auto caller_cancel=option.cancelled;
            option.cancelled=[&d,caller_cancel]{return d.failed.load() || (caller_cancel && caller_cancel());};
            if(!p->tier.open(option,err)) {
                d.reason="CUDA worker "+std::to_string(i)+" setup: "+err;d.failed.store(true);close();err=d.reason;return false;
            }
            if(d.failed.load()) {err=d.error();close();return false;}
            p->interval=std::min(50,std::max(1,opts[i].deadline_ms/4));
            p->tier.report(p->report);
            for(const auto& a:opts[i].plan.assets)d.owners[(size_t)a.layer*d.experts+a.expert]=(int)i;
            d.peers.push_back(std::move(p));
            auto* peer=d.peers.back().get();
            peer->thread=std::thread([&d,peer]{d.run(*peer);});
        }
    } catch (...) {d.failed.store(true);close();throw;}
    if(d.failed.load()){err=d.error();close();return false;}
    d.ready=true;return true;
}
void GpuWorkerGroup::close() {
    auto& d=*impl_;
    for(auto& p:d.peers) {std::lock_guard<std::mutex> lock(p->mutex);p->stopping=true;p->wake.notify_all();}
    for(auto& p:d.peers) {if(p->thread.joinable())p->thread.join();p->tier.close();}
    d.peers.clear();d.ready=d.posted=d.finished=false;d.outputs.clear();d.rows.clear();d.masks.clear();d.active.clear();
}
bool GpuWorkerGroup::holds(int64_t l,int32_t e)const{return impl_->owner(l,e)>=0;}
bool GpuWorkerGroup::dead()const{return impl_->failed.load();}
bool GpuWorkerGroup::ready()const{return impl_->ready && !dead();}
size_t GpuWorkerGroup::size()const{return impl_->peers.size();}
bool GpuWorkerGroup::active(size_t i)const{return i<impl_->active.size() && impl_->active[i];}
const uint8_t* GpuWorkerGroup::mask(size_t i)const{return impl_->masks.at(i).data();}
const EntryRowsReply& GpuWorkerGroup::entry_rows(size_t i)const{return impl_->rows.at(i);}
const float* GpuWorkerGroup::partials(size_t i)const{return impl_->outputs.at(i).data();}
bool GpuWorkerGroup::post(int64_t l,const float* x,const int32_t* ids,const float* weights,
                           const uint8_t* owned,int T,int K,std::string& err) {
    auto& d=*impl_;
    auto refuse=[&](const char* why){d.reason=why;d.failed.store(true);err=d.reason;close();return false;};
    if(d.failed.load()){err=d.error();close();return false;}
    if(!d.ready || d.posted || !x || !ids || !weights || !owned || l<0 || l>=d.layers || T<1 || T>kMaxT || K<1 || K>kMaxK)
        return refuse("invalid CUDA group post state or geometry");
    d.finished=false;d.T=T;d.K=K;
    const size_t n=(size_t)T*K;
    for(size_t i=0;i<n;++i)if(ids[i]<0 || ids[i]>=d.experts || !std::isfinite(weights[i]) ||
        owned[i]!=(d.owner(l,ids[i])>=0))return refuse("CUDA group ownership/ids/weights mismatch");
    for(int i=0;i<T*d.H;++i)if(!std::isfinite(x[i]))return refuse("CUDA group activation nonfinite");
    // Allocate/copy every request before publishing any to a communication thread.
    std::vector<Impl::Request> jobs(d.peers.size());
    for(size_t p=0;p<d.peers.size();++p) {
        auto& r=jobs[p];r.layer=l;r.T=T;r.K=K;r.mask.assign(n,0);
        for(size_t i=0;i<n;++i)r.mask[i]=d.owner(l,ids[i])==(int)p;
        d.masks[p]=r.mask;d.outputs[p].clear();d.rows[p]={};
        d.active[p]=std::find(r.mask.begin(),r.mask.end(),1)!=r.mask.end();
        if(d.active[p]) {r.x.assign(x,x+T*d.H);r.ids.assign(ids,ids+n);r.weights.assign(weights,weights+n);}
    }
    d.posted=std::find(d.active.begin(),d.active.end(),1)!=d.active.end();
    for(size_t p=0;p<d.peers.size();++p)if(d.active[p]) {
        auto& peer=*d.peers[p];std::lock_guard<std::mutex> lock(peer.mutex);
        if(d.failed.load()) break; // an owner may already have exited after another peer failed
        peer.request=std::move(jobs[p]);peer.output.clear();peer.busy=peer.pending=true;peer.wake.notify_all();
    }
    if(d.failed.load()) {err=d.error();finish(err);return false;}
    return true;
}
bool GpuWorkerGroup::finish(std::string& err) {
    auto& d=*impl_;
    for(auto& p:d.peers) {std::unique_lock<std::mutex> lock(p->mutex);p->wake.wait(lock,[&]{return !p->busy;});}
    d.posted=false;
    if(d.failed.load()) {err=d.error();d.reason=err;close();return false;}
    for(size_t i=0;i<d.peers.size();++i)if(d.active[i]) {
        std::lock_guard<std::mutex> lock(d.peers[i]->mutex);d.outputs[i]=d.peers[i]->output;d.rows[i]=d.peers[i]->rows;
    }
    d.finished=true;return true;
}
bool GpuWorkerGroup::row_mode()const{return impl_->entry_rows;}
bool GpuWorkerGroup::reduce_rows(const std::vector<float>& cpu_rows,const std::vector<uint8_t>& cpu_mask,
                                 const std::vector<uint8_t>& expected,const float* weights,int T,int K,
                                 std::vector<float>& partials,std::string& err)const {
    const auto& d=*impl_;partials.clear();
    auto refuse=[&](const char* why){err=why;return false;};
    const size_t n=(size_t)T*K;
    if(!ready() || !d.entry_rows || !d.finished || T<1 || T>kMaxT || K<1 || K>kMaxK || T!=d.T || K!=d.K || !weights ||
       expected.size()!=n || cpu_mask.size()!=n || cpu_rows.size()!=n*d.H)
        return refuse("ordered mixed row state/shape mismatch");
    std::vector<MixedRowsPeer> peers;
    for(size_t p=0;p<d.rows.size();++p)if(d.active[p])peers.push_back({d.rows[p],d.masks[p]});
    return reduce_mixed_rows(d.H,T,K,cpu_rows,cpu_mask,expected,weights,peers,partials,err);
}
void GpuWorkerGroup::report(std::string& out)const {
    std::ostringstream text;text<<"CUDA group workers="<<size()<<" "<<(dead()?"DEAD":"READY");
    for(size_t i=0;i<impl_->peers.size();++i) {
        std::lock_guard<std::mutex> lock(impl_->peers[i]->mutex);
        text<<"\nworker["<<i<<"] "<<impl_->peers[i]->report;
    }
    out=text.str();
}
} // namespace strata::worker
