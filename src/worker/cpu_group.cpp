#include "strata/worker/cpu_group.hpp"
#include "strata/worker/setup_keepalive.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>
#include <sstream>

namespace strata::worker {
namespace {
uint64_t now_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
}
ExpertWorkerGroup::~ExpertWorkerGroup() { close(); }
bool ExpertWorkerGroup::validate(const std::vector<Open>& options, const std::vector<uint8_t>& expected,
                                std::string& err) {
    auto refuse = [&](const char* why) { err = why; return false; };
    if (options.empty() || options.size() > kMaxCpuWorkers) return refuse("CPU group needs 1..2 endpoints");
    for(const auto& option:options)if((option.diagnostic_regroup || option.diagnostic_ordered) && options.size()!=1)
        return refuse("regroup diagnostic needs exactly one full resident endpoint");
    const auto& first = options.front();
    if (!first.local || !first.n_layers || first.n_layers > kMaxLayers || !first.n_expert ||
        first.n_expert > kMaxExperts || !first.n_embd || first.n_embd > kMaxNEmbd ||
        !first.n_ff || first.n_ff > kMaxNFf) return refuse("CPU group requires trusted bounded geometry");
    const size_t stride = (first.n_expert + 7u) / 8u;
    if (expected.size() != first.n_layers * stride) return refuse("CPU complement bitmap size mismatch");
    std::set<std::pair<std::string,uint16_t>> endpoints;
    std::vector<uint8_t> covered(expected.size(), 0);
    for (const auto& o : options) {
        CpuEndpoint endpoint;
        if (!parse_cpu_endpoint(o.host + ":" + std::to_string(o.port), endpoint) ||
            !endpoints.emplace(o.host,o.port).second || o.deadline_ms < 1 || o.setup_ms < 1)
            return refuse("invalid/duplicate CPU endpoint or deadline");
        if (!o.local || o.local->identity != first.local->identity || o.gu_type != first.gu_type ||
            o.d_type != first.d_type || o.n_layers != first.n_layers || o.n_expert != first.n_expert ||
            o.n_embd != first.n_embd || o.n_ff != first.n_ff || o.partition.size() != expected.size())
            return refuse("CPU group trusted model/layout/geometry mismatch");
        SetupManifest m; m.model_id=o.local->identity.a; m.gu_type=o.gu_type; m.d_type=o.d_type;
        m.n_layers=o.n_layers; m.n_expert=o.n_expert; m.n_embd=o.n_embd; m.n_ff=o.n_ff;
        std::strcpy(m.quant_name,"Q8_0"); m.partition=o.partition; m.partition_id=blob_digest(m.partition).a;
        if (!o.local->matches(m)) return refuse("CPU group local manifest mismatch");
        bool any = false;
        for (size_t i=0;i<expected.size();++i) {
            if ((o.partition[i] & ~expected[i]) || (covered[i] & o.partition[i]))
                return refuse("CPU group ownership overlaps or escapes the CPU complement");
            any = any || o.partition[i]; covered[i] |= o.partition[i];
        }
        if (!any) return refuse("CPU group endpoint owns no experts");
    }
    if (covered != expected) return refuse("CPU group leaves missing CPU experts");
    for (int l=0;l<first.n_layers;++l) if (first.n_expert % 8 &&
        (expected[l*stride+stride-1] & (uint8_t)~((1u<<(first.n_expert%8))-1)))
        return refuse("CPU complement has nonzero padding bits");
    return true;
}
bool ExpertWorkerGroup::open(const std::vector<Open>& options, const std::vector<uint8_t>& expected,
                            std::string& err) {
    if (dead()) { err=error(); return false; }
    if (ready_ || !validate(options,expected,err)) { if(err.empty())err="CPU group already open";return false; }
    layers_=options[0].n_layers;experts_=options[0].n_expert;H_=options[0].n_embd;
    owners_.assign((size_t)layers_*experts_,-1);stats_={};stopping_=window_=false;
    const size_t stride=(experts_+7u)/8u;
    try {
        for (size_t i=0;i<options.size();++i) {
            // The first ready session must not expire while the second peer reads its cold subset.
            std::unique_ptr<CpuSetupKeepalive> acquired;
            if (!peers_.empty()) acquired=std::make_unique<CpuSetupKeepalive>(*peers_.front(),
                std::max(1,std::min(50,options[0].deadline_ms/4)));
            auto peer=std::make_unique<ExpertWorkerTier>();
            auto option=options[i]; const auto cancelled=option.cancelled;
            option.cancelled=[&,cancelled] { return (acquired && acquired->failed()) || (cancelled && cancelled()); };
            const bool opened=peer->open(option,err);
            std::string prior;
            const bool alive=!acquired || acquired->finish(prior);
            if (!opened || !alive) { if(!alive)err=prior;fail(err);close();return false; }
            for (int l=0;l<layers_;++l) for(int e=0;e<experts_;++e)
                if(options[i].partition[l*stride+e/8]&(1u<<(e%8)))owners_[(size_t)l*experts_+e]=(int)i;
            stats_.setup_blobs+=peer->stats().setup_blobs;
            stats_.setup_weight_bytes+=peer->stats().setup_weight_bytes;
            stats_.setup_ms+=peer->stats().setup_ms; // serial setup scopes; per-peer reports retained
            peers_.push_back(std::move(peer));
            endpoints_.push_back({options[i].host,options[i].port});
            interval_ms_=std::min(interval_ms_,std::max(1,options[i].deadline_ms/4));
        }
        masks_.resize(peers_.size());active_.assign(peers_.size(),0);
        ready_=true;
        keeper_=std::thread([this] { keepalive(); });
        return true;
    } catch (const std::exception& e) { err=e.what();fail(err);close();return false; }
}
void ExpertWorkerGroup::fail(const std::string& why) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!failed_.load())reason_=why;
    failed_.store(true,std::memory_order_release);wake_.notify_all();
}
std::string ExpertWorkerGroup::error() const { std::lock_guard<std::mutex> lock(mutex_);return reason_; }
void ExpertWorkerGroup::keepalive() {
    try {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!stopping_ && !failed_.load()) {
        wake_.wait_for(lock,std::chrono::milliseconds(interval_ms_));
        if(stopping_ || failed_.load())break;
        for(size_t i=0;i<peers_.size();++i) {
            if(window_ && active_[i])continue;
            std::string why;
            if(!peers_[i]->keepalive_if_idle(why)) {
                reason_="CPU peer "+std::to_string(i)+" idle: "+why;failed_.store(true);break;
            }
        }
    }
    } catch (...) {
        try { fail("CPU group idle keeper exception"); }
        catch (...) { failed_.store(true,std::memory_order_release); }
    }
}
void ExpertWorkerGroup::close() {
    { std::lock_guard<std::mutex> lock(mutex_);stopping_=true;wake_.notify_all(); }
    if(keeper_.joinable())keeper_.join();
    for(auto& peer:peers_)peer->close(); // each joins its existing request thread before socket close
    peers_.clear();endpoints_.clear();masks_.clear();active_.clear();combined_.clear();owners_.clear();
    ready_=posted_=served_=false;
}
bool ExpertWorkerGroup::holds(int64_t l,int32_t e) const {
    return l>=0 && l<layers_ && e>=0 && e<experts_ && !owners_.empty() && owners_[(size_t)l*experts_+e]>=0;
}
bool ExpertWorkerGroup::post(int64_t l,const float* x,const int32_t* ids,const float* weights,
                             const uint8_t* owned,int T,int K,std::string& err) {
    served_=false;combined_.clear();
    if(posted_) {err="CPU group already has an unfinished window";fail(err);std::string drain;finish(drain);return false;}
    if(!ready() || posted_ || !x || !ids || !weights || !owned || l<0 || l>=layers_ ||
        T<1 || T>kMaxT || K<1 || K>kMaxK) {err="invalid CPU group post";fail(err);return false;}
    for(int i=0;i<T*K;++i) if(ids[i]<0 || ids[i]>=experts_ || !std::isfinite(weights[i]) ||
        owned[i]!=(holds(l,ids[i])?1:0)) {err="CPU group ownership/ids/weights mismatch";fail(err);return false;}
    for(int i=0;i<T*H_;++i)if(!std::isfinite(x[i])) {err="CPU group activation nonfinite";fail(err);return false;}
    // The keeper reads active_ only while window_ is true. Finish/transition synchronization ensures
    // its previous window has ended before these masks are changed.
    for(size_t p=0;p<peers_.size();++p) {
        masks_[p].assign((size_t)T*K,0);active_[p]=0;
        for(int i=0;i<T*K;++i) if(owned[i] && owners_[(size_t)l*experts_+ids[i]]==(int)p)
            masks_[p][i]=active_[p]=1;
    }
    bool ok=true;
    { std::lock_guard<std::mutex> lock(mutex_);
      if(failed_.load()) {err=reason_;return false;} window_=true;
      T_=T;posted_ns_=now_ns();posted_=true;
      try {
        for(size_t p=0;p<peers_.size();++p) {
            // post() copies the input and wakes that peer; never wait for a reply in this loop.
            // Keep the group lock through publication: an inactive peer's post resets its state
            // before taking the tier lock, so its idle keeper must not probe concurrently.
            if(!peers_[p]->post(l,x,ids,weights,masks_[p].data(),T,K,err)) {ok=false;break;}
        }
      } catch(const std::exception& e) {err=e.what();ok=false;}
    }
    if(!ok) {fail(err);std::string drained;finish(drained);return false;}
    return true;
}
bool ExpertWorkerGroup::finish(std::string& err) {
    if(!posted_) {if(dead()){err=error();return false;}return true;}
    const auto wait_start=now_ns();
    bool ok=!dead();
    for(size_t p=0;p<peers_.size();++p) {
        std::string why;
        if(!peers_[p]->finish(why)) {if(err.empty())err="CPU peer "+std::to_string(p)+": "+why;ok=false;}
    }
    ok=ok && !dead();
    // Every exchange has drained before exposing any partial, including a failure on another peer.
    try { if(ok) {
        bool first=true;
        for(size_t p=0;p<peers_.size();++p) if(active_[p]) {
            if(!peers_[p]->served()) {err="CPU peer did not serve its share";ok=false;break;}
            const float* row=peers_[p]->partials();
            if(first) {combined_.assign(row,row+(size_t)T_*H_);first=false;}
            else for(size_t i=0;i<combined_.size();++i)combined_[i]=combined_[i]+row[i];
            for(size_t i=0;i<combined_.size();++i)if(!std::isfinite(row[i]) || !std::isfinite(combined_[i]))
                {err="CPU grouped partial is nonfinite";ok=false;break;}
            if(!ok)break;
        }
        if(first)combined_.assign((size_t)T_*H_,0.f);
    } } catch(const std::exception& failure) {err=failure.what();ok=false;}
    stats_.request_ms+=(double)(now_ns()-posted_ns_)/1e6;
    stats_.wait_ms+=(double)(now_ns()-wait_start)/1e6;
    stats_.requests=stats_.entries=0;
    for(const auto& peer:peers_) {stats_.requests+=peer->stats().requests;stats_.entries+=peer->stats().entries;}
    {std::lock_guard<std::mutex> lock(mutex_);
     ok=ok && !failed_.load();posted_=false;
     served_=ok && std::find(active_.begin(),active_.end(),1)!=active_.end();window_=false;wake_.notify_all();}
    if(!ok) {combined_.clear();if(err.empty())err=error();fail(err);return false;}
    return true;
}
bool ExpertWorkerGroup::peer_status(size_t i,WorkerStatusInfo& out,std::string& err) {
    return i<peers_.size() && peers_[i]->status(out,err);
}
void ExpertWorkerGroup::report(std::string& out) const {
    std::ostringstream text;
    text<<"CPU group peers="<<size()<<" "<<(dead()?"DEAD":ready()?"READY":"CLOSED")<<" group_request_wall_ms="<<stats_.request_ms
        <<" caller_wait_ms="<<stats_.wait_ms<<" (peer timings overlap; do not sum as wall time)";
    for(size_t i=0;i<peers_.size();++i) {std::string row;peers_[i]->report(row);
        text<<"\nCPU peer["<<i<<"] "<<endpoints_[i].host<<":"<<endpoints_[i].port<<" "<<row;}
    out=text.str();
}
} // namespace strata::worker
