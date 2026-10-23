#include "strata/worker/cpu_group.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <sstream>

namespace strata::worker {
namespace {
bool bit(const std::vector<uint8_t>& p, size_t stride, int l, int e) {
    return (p[(size_t)l * stride + (size_t)e / 8] & (uint8_t)(1u << (e % 8))) != 0;
}
void set_bit(std::vector<uint8_t>& p, size_t stride, int l, int e) {
    p[(size_t)l * stride + (size_t)e / 8] |= (uint8_t)(1u << (e % 8));
}
}

bool place_cpu_partitions(const std::vector<uint8_t>& complement, uint16_t layers, uint16_t experts,
                          const std::vector<std::pair<int32_t, int32_t>>& profile,
                          const std::vector<CpuEndpoint>& endpoints,
                          std::vector<std::vector<uint8_t>>& partitions, std::string& err) {
    if (!layers || !experts || endpoints.empty() || endpoints.size() > kMaxCpuWorkers) {
        err = "CPU placement needs valid geometry and 1..8 workers"; return false;
    }
    const size_t stride = ((size_t)experts + 7) / 8, bytes = (size_t)layers * stride;
    if (complement.size() != bytes) {err = "CPU complement bitmap does not match geometry";return false;}
    if (experts & 7) for(int l=0;l<layers;++l)
        if(complement[(size_t)l*stride+stride-1] & (uint8_t)~((1u << (experts & 7)) - 1u)) {
            err = "CPU complement has out-of-range ownership bits"; return false;
        }
    std::set<std::pair<std::string,uint16_t>> addresses;
    for (const auto& e:endpoints) if(e.host.empty() || !e.port || !std::isfinite(e.speed) || e.speed<=0 ||
        !addresses.emplace(e.host,e.port).second) {err="CPU endpoints, ports and speeds must be valid and unique";return false;}
    uint64_t total=0; for(int l=0;l<layers;++l)for(int e=0;e<experts;++e)total+=bit(complement,stride,l,e);
    if (!total) {err="CPU complement is empty";return false;}
    if(endpoints.size()>total){err="CPU placement gives at least one expert to every configured worker";return false;}
    uint64_t finite_capacity=0; bool unbounded=false;
    for(const auto& e:endpoints) {if(e.capacity)finite_capacity+=e.capacity;else unbounded=true;}
    if(!unbounded && finite_capacity<total) {err="CPU worker capacities do not cover the complement";return false;}

    // Deterministic blob quotas: bounded workers receive up to their proportional share, with any residual
    // assigned in endpoint order. Zero capacities share what remains equally. No quota exceeds a hard cap.
    std::vector<uint64_t> quota(endpoints.size(),0), count(endpoints.size(),0);
    long double denom=0; for(const auto& e:endpoints)denom += e.capacity ? e.capacity : total;
    uint64_t used=0;
    for(size_t i=0;i<endpoints.size();++i) {
        const uint64_t basis=endpoints[i].capacity?endpoints[i].capacity:total;
        quota[i]=std::min<uint64_t>(endpoints[i].capacity?endpoints[i].capacity:total,
                    (uint64_t)((long double)total*basis/denom)); used+=quota[i];
    }
    while(used<total) {
        bool advanced=false;
        for(size_t i=0;i<endpoints.size() && used<total;++i)
            if(!endpoints[i].capacity || quota[i]<endpoints[i].capacity) {++quota[i];++used;advanced=true;}
        if(!advanced) {err="CPU worker capacities do not cover the complement";return false;}
    }

    partitions.assign(endpoints.size(),std::vector<uint8_t>(bytes,0));
    std::vector<std::vector<uint64_t>> layer_count(endpoints.size(),std::vector<uint64_t>(layers,0));
    std::vector<long double> heat(endpoints.size(),0);
    std::vector<uint8_t> placed((size_t)layers*experts,0);
    auto assign=[&](int l,int e,long double cost,bool ranked)->bool {
        if(l<0 || l>=layers || e<0 || e>=experts || !bit(complement,stride,l,e)) return true;
        const size_t slot=(size_t)l*experts+e;if(placed[slot])return true;
        size_t best=endpoints.size();
        long double best_heat=0,best_layer=0;
        for(size_t i=0;i<endpoints.size();++i) if(count[i]<quota[i]) {
            const long double hs=(heat[i]+(ranked?cost:0))/endpoints[i].speed;
            const long double ls=(long double)(layer_count[i][l]+1)/(long double)std::max<uint64_t>(quota[i],1);
            if(best==endpoints.size() || hs<best_heat || (hs==best_heat && (ls<best_layer || (ls==best_layer && i<best)))) {
                best=i;best_heat=hs;best_layer=ls;
            }
        }
        if(best==endpoints.size())return false;
        set_bit(partitions[best],stride,l,e);placed[slot]=1;++count[best];++layer_count[best][l];
        if(ranked)heat[best]+=cost;return true;
    };
    const long double ranks=(long double)profile.size();
    for(size_t r=0;r<profile.size();++r)
        if(!assign(profile[r].first,profile[r].second,ranks-(long double)r,true)) {err="CPU placement exhausted capacity";return false;}
    // Unranked complement is traversed layer-round-robin, avoiding contiguous whole-layer ownership.
    for(int e=0;e<experts;++e)for(int l=0;l<layers;++l)
        if(!assign(l,e,0,false)) {err="CPU placement exhausted capacity";return false;}
    for(uint8_t v:placed)(void)v;
    return true;
}

struct CpuWorkerGroup::Impl {
    std::vector<std::unique_ptr<ExpertWorkerTier>> peers;
    std::vector<int> owners;
    std::vector<std::vector<uint8_t>> masks;
    std::vector<uint8_t> active;
    bool ready=false, posted=false, failed=false;
    int layers=0,experts=0;
    std::string reason;
    int owner(int64_t l,int32_t e)const{return l>=0&&l<layers&&e>=0&&e<experts?owners[(size_t)l*experts+e]:-1;}
};

CpuWorkerGroup::CpuWorkerGroup():impl_(std::make_unique<Impl>()){}
CpuWorkerGroup::~CpuWorkerGroup(){close();}

bool CpuWorkerGroup::validate(const std::vector<ExpertWorkerTier::Open>& o,const std::vector<uint8_t>& complement,std::string& err){
    if(o.empty()||o.size()>kMaxCpuWorkers){err="CPU worker count must be 1..8";return false;}
    const auto& f=o.front();const size_t stride=((size_t)f.n_expert+7)/8,bytes=(size_t)f.n_layers*stride;
    if(!f.n_layers||!f.n_expert||complement.size()!=bytes){err="CPU group complement/geometry mismatch";return false;}
    std::set<std::pair<std::string,uint16_t>> endpoints;std::vector<uint8_t> seen(bytes,0);
    for(const auto& x:o){
        if(x.host.empty()||!x.port||!endpoints.emplace(x.host,x.port).second||x.n_layers!=f.n_layers||
           x.n_expert!=f.n_expert||x.n_embd!=f.n_embd||x.n_ff!=f.n_ff||x.gu_type!=f.gu_type||x.d_type!=f.d_type||
           std::string(x.model_name?x.model_name:"")!=std::string(f.model_name?f.model_name:"")||x.partition.size()!=bytes){
            err="CPU workers must have unique endpoints and identical model/layout/geometry";return false;
        }
        for(size_t i=0;i<bytes;++i){if(seen[i]&x.partition[i]){err="CPU worker assignments overlap";return false;}seen[i]|=x.partition[i];}
    }
    if(f.n_expert&7)for(int l=0;l<f.n_layers;++l)
        if(seen[(size_t)l*stride+stride-1]&(uint8_t)~((1u<<(f.n_expert&7))-1u)){
            err="CPU worker assignment is out of range";return false;
        }
    if(seen!=complement){err="CPU worker assignments are not an exhaustive copy of the frozen complement";return false;}
    return true;
}

bool CpuWorkerGroup::open(const std::vector<ExpertWorkerTier::Open>& o,const std::vector<uint8_t>& complement,std::string& err){
    auto& d=*impl_;if(d.ready||d.failed||!validate(o,complement,err)){if(err.empty())err="CPU group already open or failed";return false;}
    d.layers=o[0].n_layers;d.experts=o[0].n_expert;d.owners.assign((size_t)d.layers*d.experts,-1);
    d.masks.resize(o.size());d.active.assign(o.size(),0);
    for(size_t i=0;i<o.size();++i){
        auto p=std::make_unique<ExpertWorkerTier>();
        if(!p->open(o[i],err)){d.reason="CPU worker "+std::to_string(i)+" setup: "+err;d.failed=true;close();err=d.reason;return false;}
        for(int l=0;l<d.layers;++l)for(int e=0;e<d.experts;++e)if(p->holds(l,e))d.owners[(size_t)l*d.experts+e]=(int)i;
        d.peers.push_back(std::move(p));
    }
    d.ready=true;return true;
}

void CpuWorkerGroup::close(){auto& d=*impl_;for(auto& p:d.peers)p->close();d.peers.clear();d.ready=d.posted=false;d.masks.clear();d.active.clear();}
bool CpuWorkerGroup::holds(int64_t l,int32_t e)const{return impl_->owner(l,e)>=0;}
bool CpuWorkerGroup::ready()const{return impl_->ready&&!dead();}
bool CpuWorkerGroup::dead()const{if(impl_->failed)return true;for(const auto& p:impl_->peers)if(p->dead())return true;return false;}
size_t CpuWorkerGroup::size()const{return impl_->peers.size();}
bool CpuWorkerGroup::posted()const{return impl_->posted;}
bool CpuWorkerGroup::active(size_t i)const{return i<impl_->active.size()&&impl_->active[i];}
bool CpuWorkerGroup::served(size_t i)const{return i<impl_->peers.size()&&impl_->peers[i]->served();}
const uint8_t* CpuWorkerGroup::mask(size_t i)const{return impl_->masks.at(i).data();}
const float* CpuWorkerGroup::partials(size_t i)const{return impl_->peers.at(i)->partials();}
const ExpertWorkerTier::Stats& CpuWorkerGroup::stats(size_t i)const{return impl_->peers.at(i)->stats();}
bool CpuWorkerGroup::status(size_t i,WorkerStatusInfo& out,std::string& err){if(i>=size()){err="CPU worker index out of range";return false;}return impl_->peers[i]->status(out,err);}

bool CpuWorkerGroup::post(int64_t l,const float* x,const int32_t* ids,const float* weights,const uint8_t* owned,int T,int K,std::string& err){
    auto& d=*impl_;if(!d.ready||d.posted||dead()||!x||!ids||!weights||!owned||l<0||l>=d.layers||T<1||T>kMaxT||K<1||K>kMaxK){err="invalid CPU group post state or geometry";d.failed=true;return false;}
    const size_t n=(size_t)T*K;std::vector<std::vector<uint8_t>> masks(d.peers.size(),std::vector<uint8_t>(n,0));
    d.active.assign(d.peers.size(),0);
    for(size_t i=0;i<n;++i){const int owner=d.owner(l,ids[i]);if(owned[i]&&owner<0){err="CPU group ownership mask names an expert outside its frozen partitions";d.failed=true;return false;}if(owned[i]){masks[(size_t)owner][i]=1;d.active[(size_t)owner]=1;}}
    d.masks=masks;d.posted=std::find(d.active.begin(),d.active.end(),1)!=d.active.end();
    for(size_t i=0;i<d.peers.size();++i)if(d.active[i]){
        if(!d.peers[i]->post(l,x,ids,weights,masks[i].data(),T,K,err)){
            d.reason="CPU worker "+std::to_string(i)+" post: "+err;d.failed=true;
            std::string ignored;for(size_t j=0;j<i;++j)if(d.active[j])d.peers[j]->finish(ignored);
            d.posted=false;err=d.reason;return false;
        }
    }
    return true;
}

bool CpuWorkerGroup::finish(std::string& err){
    auto& d=*impl_;bool ok=true;std::string first;
    // Drain every published request even after one fails: no tier retains an in-flight request and no partial commits.
    for(size_t i=0;i<d.peers.size();++i)if(d.active[i]){std::string e;if(!d.peers[i]->finish(e)){if(first.empty())first="CPU worker "+std::to_string(i)+": "+e;ok=false;}}
    d.posted=false;if(d.failed&&first.empty()){first=d.reason.empty()?"CPU worker group failed":d.reason;ok=false;}
    if(!ok){d.failed=true;d.reason=first;err=first;return false;}return true;
}

void CpuWorkerGroup::report(std::string& out)const{std::ostringstream s;s<<"CPU group workers="<<size()<<" "<<(dead()?"DEAD":"READY");for(size_t i=0;i<size();++i){std::string line;impl_->peers[i]->report(line);s<<"\nworker["<<i<<"] "<<line;}out=s.str();}

} // namespace strata::worker
