#pragma once
// Saved-capture local discriminator; only diagnostic code, no engine/worker implementation changes.
#include "diagnostic_child.hpp"
#include "strata/worker/cpu_capture.hpp"
#include "strata/worker/cpu_group.hpp"
#include <charconv>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <sys/stat.h>
#include <sys/resource.h>
namespace strata::worker::diagnostic {
struct SubsetConfig {
    std::string binary,local_root,output_root;
    uint64_t expected_bytes=0,reserve_bytes=8ull<<30,max_owned_rss=60ull<<30;
    int workers=12,loaders=4,frame_ms=10000,setup_ms=600000,total_ms=1500000,caller=0;
    std::vector<int> pool;
};
inline uint64_t number(const std::string& s){uint64_t n=0;const auto got=std::from_chars(s.data(),s.data()+s.size(),n);
    need(!s.empty() && got.ec==std::errc{} && got.ptr==s.data()+s.size(),"invalid unsigned config value");return n;}
inline uint64_t temporary_bytes(const CpuCaptureFile& f){
    strata::kernels::cpu::NativeFmt fmt;std::string err;
    need(strata::kernels::cpu::native_fmt(8,8,f.setup.n_embd,f.setup.n_ff,fmt,err),err);
    return ((f.setup.owned_count()+1)/2)*(uint64_t)fmt.bytes; // peak: only the larger half is live
}
inline SubsetConfig read_subset_config(const std::string& path,const CpuCaptureFile& f,const LocalManifest& m){
    need(std::filesystem::path(path).is_absolute(),"subset config must be absolute");
    std::ifstream input(path,std::ios::ate);const auto bytes=input.tellg();
    need(input && bytes>=0 && bytes<=16384,"subset config unreadable/oversize");input.seekg(0);
    std::map<std::string,std::string> fields;std::string key,value;
    while(input>>key){need((bool)(input>>std::quoted(value)),"missing subset config value");
        need(fields.emplace(key,value).second,"duplicate subset config key");}
    need(input.eof() && fields.size()==13,"subset config requires exactly 13 known fields");
    SubsetConfig c;c.binary=fields.at("worker_binary");c.local_root=fields.at("local_root");c.output_root=fields.at("output_root");
    c.expected_bytes=number(fields.at("temporary_arena_bytes"));c.reserve_bytes=number(fields.at("reserve_bytes"));
    c.max_owned_rss=number(fields.at("max_owned_rss_bytes"));
    auto bounded=[&](const char* k,uint64_t lo,uint64_t hi){auto n=number(fields.at(k));need(n>=lo && n<=hi,"subset config value outside bounds");return (int)n;};
    c.workers=bounded("workers",1,64);c.loaders=bounded("loader_threads",1,4);c.frame_ms=bounded("frame_ms",100,10000);
    c.setup_ms=bounded("setup_ms",1000,600000);c.total_ms=bounded("total_ms",1000,1800000);c.caller=bounded("caller_cpu",0,1023);
    std::istringstream cpus(fields.at("pool_cpus"));std::string cpu;
    while(std::getline(cpus,cpu,',')){auto n=number(cpu);need(n<1024,"pool CPU outside bounds");c.pool.push_back((int)n);}
    need(c.pool.size()==(size_t)c.workers,"one pool CPU per configured worker required");
    need(f.setup.owned_count()>=2 && c.expected_bytes==temporary_bytes(f) && c.expected_bytes>0 && c.reserve_bytes>=8ull<<30 &&
         c.reserve_bytes<=64ull<<30 && c.max_owned_rss<=60ull<<30 &&
         c.max_owned_rss>=c.expected_bytes+kCpuCaptureLimit,"subset arena/RSS/reserve bounds mismatch");
    rlimit address_space{};need(::getrlimit(RLIMIT_AS,&address_space)==0,"read inherited address-space limit");
    need(address_space.rlim_cur==RLIM_INFINITY || address_space.rlim_cur>=c.expected_bytes+(1ull<<30),
         "inherited address-space cap too small for a subset worker");
    for(const auto& p:{c.binary,c.local_root,c.output_root})need(std::filesystem::path(p).is_absolute(),"subset paths must be absolute");
    need(::access(c.binary.c_str(),X_OK)==0 && std::filesystem::is_regular_file(c.binary),"subset worker not executable");
    need(std::filesystem::is_directory(std::filesystem::path(c.output_root).parent_path()) &&
         !std::filesystem::exists(c.output_root),"subset output root must be new beneath an existing directory");
    LocalSource source;std::string err;need(source.open(c.local_root,m,err),err);
    for(uint32_t i=0;i<m.files.size();++i)need(source.file(i,err)>=0,err); // metadata only, no duplicate weights
#ifdef __linux__
    cpu_set_t allowed;CPU_ZERO(&allowed);need(::sched_getaffinity(0,sizeof allowed,&allowed)==0,"read allowed CPUs");
    auto planned=c.pool;planned.push_back(c.caller);
    for(int p:planned)need(CPU_ISSET(p,&allowed),"configured CPU outside launcher's allowed mask");
#else
    throw std::runtime_error("local subset diagnostic requires Linux");
#endif
    return c;
}
inline std::map<std::string,uint64_t> proc_numbers(const std::string& path){
    std::ifstream input(path);need((bool)input,"resource coverage missing");std::map<std::string,uint64_t> result;std::string line;
    while(std::getline(input,line)){std::istringstream row(line);std::string key,unit;uint64_t n;
        if(row>>key>>n){if(!key.empty() && key.back()==':')key.pop_back();row>>unit;
            if(unit.empty() || unit=="kB")result[key]=n*(unit=="kB"?1024:1);}}
    return result;
}
inline bool cgroup_room(const std::filesystem::path& directory,uint64_t required){
    for(const char* key:{"memory.max","memory.high"})if(std::filesystem::exists(directory/key)){
        std::ifstream input(directory/key);std::string limit;need((bool)std::getline(input,limit),"cgroup limit coverage missing");
        if(limit=="max")continue;
        std::ifstream usage(directory/"memory.current");std::string value;
        need((bool)std::getline(usage,value),"cgroup current coverage missing");
        const auto cap=number(limit),current=number(value);
        if(current>cap || required>cap-current)return false;
    }
    return true;
}
struct SubsetGuard {
    const SubsetConfig& cfg;uint64_t start=millis(),peak=0,current_rss=0,min_available=UINT64_MAX,oom=0;
    std::map<std::string,std::map<std::string,uint64_t>> cgroup_oom;
    Child* child=nullptr; // one live temporary process; never a handle for the full resident
    explicit SubsetGuard(const SubsetConfig& c):cfg(c){
        oom=proc_numbers("/proc/vmstat").at("oom_kill");
        std::ifstream input("/proc/self/cgroup");std::string line;bool found=false;
        while(std::getline(input,line))if(line.rfind("0::",0)==0){
            auto p=std::filesystem::path("/sys/fs/cgroup"+line.substr(3)).lexically_normal();
            while(p=="/sys/fs/cgroup" || p.string().rfind("/sys/fs/cgroup/",0)==0){const auto file=p/"memory.events";
                if(std::filesystem::exists(file)){const auto counters=proc_numbers(file.string());std::map<std::string,uint64_t> baseline;
                    for(const char* key:{"oom","oom_kill","oom_group_kill"})if(counters.count(key))baseline[key]=counters.at(key);
                    cgroup_oom.emplace(file.string(),std::move(baseline));}
                if(p=="/sys/fs/cgroup")break;p=p.parent_path();}
            found=true;break;}
        need(found && !cgroup_oom.empty(),"cgroup-v2 OOM coverage required");
    }
    void check(bool before_load=false){
        need(millis()-start<(uint64_t)cfg.total_ms,"local subset total deadline");
        const auto available=proc_numbers("/proc/meminfo").at("MemAvailable");min_available=std::min(min_available,available);
        need(available>=cfg.reserve_bytes+(before_load?cfg.expected_bytes:0),"MemAvailable below configured arena/reserve guard");
        uint64_t rss=0;
        for(pid_t p:{::getpid(),child?child->pid:-1})if(p>0){
            const auto status=proc_numbers("/proc/"+std::to_string(p)+"/status");
            need(status.at("VmSwap")==0,"owned VmSwap must remain zero");rss+=status.at("VmRSS");}
        current_rss=rss;peak=std::max(peak,rss);need(rss<=cfg.max_owned_rss,"aggregate additional owned RSS exceeds configured bound");
        need(proc_numbers("/proc/vmstat").at("oom_kill")==oom,"new host OOM kill");
        for(const auto& entry:cgroup_oom){
            need(cgroup_room(std::filesystem::path(entry.first).parent_path(),cfg.reserve_bytes+(before_load?cfg.expected_bytes:0)),
                 "finite cgroup memory.max/high headroom insufficient");
            const auto current=proc_numbers(entry.first);
            for(const auto& counter:entry.second)need(current.at(counter.first)==counter.second,"new cgroup OOM event");}
        if(child){need(child->pid>0,"owned worker already exited");child->pump();}
    }
};
// Two references occupy <=96MiB: every captured reply has an equally sized activation block in the file.
struct MaskedReferences {
    std::vector<size_t> offsets;std::vector<float> peer[2];
    explicit MaskedReferences(const CpuCaptureFile& f){size_t count=0;
        for(const auto& r:f.records){offsets.push_back(count);count+=r.partials.size();}
        need(count<=kCpuCaptureLimit/8,"masked reference storage exceeds 96 MiB");
        peer[0].resize(count);peer[1].resize(count);
    }
    void save(size_t record,int p,const std::vector<float>& v){
        need(p>=0 && p<2,"masked reference peer index");
        const size_t end=record+1<offsets.size()?offsets.at(record+1):peer[p].size();
        need(end-offsets.at(record)==v.size(),"masked reference shape");
        std::copy(v.begin(),v.end(),peer[p].begin()+offsets[record]);
    }
};
struct SubsetReport {
    uint32_t peers_checked=0,completed_peers=0,released_peers=0,max_live_children=0;
    uint64_t peak_rss=0,min_available=0,manual_ms=0;
    pid_t children[2]{-1,-1};bool reaped=false;
};
inline std::vector<uint8_t> peer_mask(const CpuCaptureFile& f,const InferRequest& r,const std::vector<int8_t>& owners,int peer){
    auto mask=r.mask;for(size_t j=0;j<mask.size();++j)mask[j]=mask[j] && owners[(size_t)r.layer*f.setup.n_expert+r.ids[j]]==peer;return mask;
}
inline std::vector<float> tier_partial(const ExpertWorkerTier& t,size_t n){
    return t.served()?std::vector<float>(t.partials(),t.partials()+n):std::vector<float>(n,0.f);
}
inline bool run_local_subsets(const CpuCaptureFile& f,const LocalManifest& local,const std::string& manifest_path,
    const MaskedReferences& reference,const SubsetConfig& cfg,SubsetReport& report,std::string& err,
    const std::function<void()>& before_manual = {}){
    report={};std::unique_ptr<SubsetGuard> monitored;
    try {
        monitored=std::make_unique<SubsetGuard>(cfg);auto& guard=*monitored;
        size_t values=0;need(reference.offsets.size()==f.records.size(),"masked reference count mismatch");
        for(size_t i=0;i<f.records.size();++i){need(reference.offsets[i]==values,"masked reference offset mismatch");values+=f.records[i].partials.size();}
        need(values<=kCpuCaptureLimit/8 && reference.peer[0].size()==values && reference.peer[1].size()==values,"masked reference layout mismatch");
        for(const auto& peer:reference.peer)need(std::all_of(peer.begin(),peer.end(),[](float v){return std::isfinite(v);}),"reference nonfinite");
        guard.check(true);need(cfg.expected_bytes==temporary_bytes(f),"peak temporary arena mismatch");
        need(::mkdir(cfg.output_root.c_str(),0700)==0,"exclusive local subset output root");
        const auto owners=cpu_capture_owners(f.setup);const size_t stride=(f.setup.n_expert+7)/8;
        std::vector<ExpertWorkerTier::Open> options(2);
        strata::kernels::cpu::NativeFmt fmt;need(strata::kernels::cpu::native_fmt(8,8,f.setup.n_embd,f.setup.n_ff,fmt,err),err);
        // Pure existing ownership validation. Ports are placeholders only; no group is opened/run.
        for(int p=0;p<2;++p){auto& o=options[p];o.host="127.0.0.1";o.port=(uint16_t)(p+1);o.local=&local;o.resident=true;
            o.n_layers=f.setup.n_layers;o.n_expert=f.setup.n_expert;o.n_embd=f.setup.n_embd;o.n_ff=f.setup.n_ff;
            o.gu_type=o.d_type=8;o.setup_ms=cfg.setup_ms;o.deadline_ms=cfg.frame_ms;o.partition.assign(f.setup.partition.size(),0);
            for(int l=0;l<o.n_layers;++l)for(int e=0;e<o.n_expert;++e)if(owners[(size_t)l*o.n_expert+e]==p)
                o.partition[(size_t)l*stride+e/8]|=(uint8_t)(1u<<(e%8));
            o.cancelled=[&]{guard.check();return false;};}
        need(ExpertWorkerGroup::validate(options,f.setup.partition,err),err);
        for(int p=0;p<2;++p){
            need(!guard.child && report.released_peers==(uint32_t)p,"previous temporary subset not released");
            guard.check(true); // fresh one-half headroom before each process/load
            ChildSpec spec;spec.binary=cfg.binary;spec.root=cfg.local_root;spec.manifest=manifest_path;
            spec.workers=cfg.workers;spec.loaders=cfg.loaders;spec.frame_ms=cfg.frame_ms;spec.setup_ms=cfg.setup_ms;spec.setup_cpus=cfg.pool;
            const auto elapsed=millis()-guard.start;need(elapsed<(uint64_t)cfg.total_ms,"local subset startup total deadline");
            spec.start_ms=(int)std::min<uint64_t>(10000,(uint64_t)cfg.total_ms-elapsed);
            spec.setup_cpus.push_back(cfg.caller);spec.log_path=cfg.output_root+"/peer"+std::to_string(p)+".log";
            const auto bytes=(uint64_t)std::count(owners.begin(),owners.end(),(int8_t)p)*fmt.bytes;
            need(bytes && bytes<=cfg.expected_bytes,"one-half arena budget mismatch");
            spec.budget_mib=(bytes+((1u<<20)-1))>>20;
            {
            Child child(spec);report.children[p]=child.pid;guard.child=&child;report.max_live_children=1;
            struct Detach {SubsetGuard& g;~Detach(){g.child=nullptr;}} detach{guard};
            std::printf("LOCAL_SUBSET_OWNED peer=%d pid=%d port=%u live_arena_bytes=%llu\n",p,child.pid,
                (unsigned)child.port,(unsigned long long)bytes);std::fflush(stdout);
            auto option=options[p];option.port=child.port;
            {
                const auto load_start=millis();ExpertWorkerTier tier;need(tier.open(option,err),err);
                std::printf("LOCAL_SUBSET_LOAD peer=%d wall_ms=%llu (separate from replay)\n",p,(unsigned long long)(millis()-load_start));
                child.compute_affinity(cfg.caller,cfg.pool);guard.check();
                const auto manual_start=millis();
                for(size_t i=0;i<f.records.size();++i){guard.check();const auto& r=f.records[i];const auto& q=r.request;
                    need(tier.keepalive_if_idle(err),err);
                    if(before_manual)before_manual(); // bounded synthetic timing seam; CLI never supplies one
                    std::vector<int32_t> ids(q.ids.begin(),q.ids.end());auto mask=peer_mask(f,q,owners,p);
                    bool posted=tier.post(q.layer,q.activations.data(),ids.data(),q.weights.data(),mask.data(),q.T,q.K,err);
                    if(!tier.keepalive_if_idle(err))posted=false;
                    const bool finished=tier.finish(err);need(posted && finished,err);
                    const auto got=tier_partial(tier,r.partials.size());
                    need(!std::memcmp(got.data(),reference.peer[p].data()+reference.offsets[i],got.size()*4),
                         "subset peer "+std::to_string(p)+" bit mismatch at record "+std::to_string(i));++report.peers_checked;
                }
                report.manual_ms+=millis()-manual_start;tier.close();++report.completed_peers;
            }
            guard.check();
            {need(child.owns_listener(),"subset accounting requires owned listener");
                ExpertWorkerClient c;WorkerCacheInfo cache;WorkerStatusInfo status;
                need(c.connect_local("127.0.0.1",child.port,cfg.frame_ms,{},err) && c.cache_status(cache,err) && c.status(status,err),err);
                uint64_t requests=0;
                for(const auto& r:f.records){const auto mask=peer_mask(f,r.request,owners,p);requests+=std::find(mask.begin(),mask.end(),1)!=mask.end();}
                need(cache.complete && !cache.session && cache.resident_bytes==bytes && cache.loaded_bytes==bytes &&
                    cache.verified_bytes==bytes && cache.arena_allocations==1 && !cache.unloads && !cache.expirations &&
                    cache.idle_remaining_ms==kResidencyNoExpiry && !status.failures && !status.setup_weight_bytes && status.requests==requests,
                    "subset load/resident/request accounting mismatch");
                std::printf("LOCAL_SUBSET_CACHE peer=%d bytes=%llu allocations=%llu requests=%llu failures=%llu\n",p,
                    (unsigned long long)bytes,(unsigned long long)cache.arena_allocations,(unsigned long long)status.requests,(unsigned long long)status.failures);
                c.close();}
            child.stop();need(child.pid<0,"owned temporary reap");
            int status=0;errno=0;need(::waitpid(report.children[p],&status,WNOHANG)<0 && errno==ECHILD,"temporary release not confirmed");
            guard.child=nullptr;guard.check();++report.released_peers;
            std::printf("LOCAL_SUBSET_RELEASE peer=%d reaped=1 owned_rss_after=%llu\n",p,(unsigned long long)guard.current_rss);
            }
        }
        report.reaped=true;
        report.peak_rss=guard.peak;report.min_available=guard.min_available;return true;
    }catch(const std::exception& failure){err=failure.what();report.peak_rss=monitored?monitored->peak:0;report.min_available=monitored?monitored->min_available:0;
        report.reaped=true;for(pid_t child:report.children)if(child>0){int status=0;errno=0;
            report.reaped=report.reaped && ::waitpid(child,&status,WNOHANG)<0 && errno==ECHILD;}
        return false;
    }
}
}
