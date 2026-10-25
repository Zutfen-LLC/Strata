// Manual diagnostic. Run only after the engine owner reaps and its resident session disconnects.
#include "strata/worker/cpu_capture.hpp"
#include "strata/worker/expert_client.hpp"
#include "cpu_subset_replay.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

using namespace strata::worker;
namespace {
bool file(const char* path,size_t cap,std::vector<uint8_t>& b){
    if(!std::filesystem::path(path).is_absolute())return false;
    std::ifstream in(path,std::ios::binary|std::ios::ate);const auto n=in.tellg();
    if(!in || n<0 || (uint64_t)n>cap)return false;
    b.resize((size_t)n);in.seekg(0);return (bool)in.read((char*)b.data(),b.size());
}
bool unchanged(const WorkerCacheInfo& a,const WorkerCacheInfo& b){
    return b.complete && !b.session && a.resident_bytes==b.resident_bytes &&
        a.loaded_bytes==b.loaded_bytes && a.verified_bytes==b.verified_bytes &&
        a.arena_allocations==b.arena_allocations && a.peak_arena_bytes==b.peak_arena_bytes &&
        a.expirations==b.expirations && a.unloads==b.unloads;
}
}
int main(int argc,char** argv){
    if(argc!=7 && !(argc==9 && std::string(argv[7])=="--local-subsets")){
        std::fprintf(stderr,"usage: cpu_capture_replay CAPTURE MANIFEST REQUESTS REFERENCE_RAW_F32 NUMERIC_HOST PORT [--local-subsets CONFIG]\n");return 2;}
    try {
        CpuCaptureFile capture;LocalManifest local;std::string err;
        char* end=nullptr;const long port=std::strtol(argv[6],&end,10);
        std::vector<uint8_t> requests,logits;
        if(!*argv[6] || *end || port<1 || port>65535 ||
           !std::filesystem::path(argv[1]).is_absolute() || !std::filesystem::path(argv[2]).is_absolute() ||
           !read_cpu_capture(argv[1],capture,err) || !read_local_manifest(argv[2],local,err) ||
           !local.matches(capture.setup) || local.identity!=capture.identity || local.index_digest!=capture.index ||
           !file(argv[3],2u<<20,requests) || !file(argv[4],4u<<20,logits)) {
            std::fprintf(stderr,"preflight refused: %s\n",err.c_str());return 1;
        }
        // Match index zero of the existing harness's GEN-line filtering, excluding line endings.
        std::istringstream asset(std::string((char*)requests.data(),requests.size()));std::string request_text;
        while(std::getline(asset,request_text)) {
            if(!request_text.empty() && request_text.back()=='\r')request_text.pop_back();
            if(request_text.rfind("GEN ",0)==0)break;
        }
        if(request_text.size()>16384 || request_text.rfind("GEN ",0)!=0 ||
           blob_digest(request_text.data(),request_text.size())!=capture.line || logits.size()!=(size_t)capture.vocab*4 ||
           blob_digest(logits)!=capture.logits) {
            std::fprintf(stderr,"request or retained raw first-logit identity mismatch; no worker contacted\n");return 1;
        }
        for(size_t i=0;i<logits.size();i+=4){float v;std::memcpy(&v,logits.data()+i,4);
            if(!std::isfinite(v)){std::fprintf(stderr,"reference logit nonfinite; no worker contacted\n");return 1;}}
        std::replace(request_text.begin(),request_text.end(),',',' ');
        std::istringstream input(request_text);
        std::string verb;uint64_t max_new=0;input>>verb>>max_new;
        std::vector<uint8_t> tokens;int64_t token=0;
        while(input>>token){
            if(token<0 || token>INT32_MAX || tokens.size()>=128u*4u){err="request token bounds";break;}
            for(int b=0;b<4;++b)tokens.push_back((uint8_t)((uint32_t)token>>(8*b)));
        }
        if(!err.empty() || !input.eof() || max_new!=capture.max_new || tokens.size()!=capture.prompt_tokens*4u ||
           blob_digest(tokens)!=capture.tokens){std::fprintf(stderr,"request token binding mismatch; no worker contacted\n");return 1;}
        std::unique_ptr<diagnostic::SubsetConfig> subset_config;
        std::unique_ptr<diagnostic::MaskedReferences> references;
        if(argc==9){diagnostic::need(std::string(argv[5])=="127.0.0.1","local subset mode requires a loopback full reference");
            subset_config=std::make_unique<diagnostic::SubsetConfig>(diagnostic::read_subset_config(argv[8],capture,local));
            references=std::make_unique<diagnostic::MaskedReferences>(capture);}
        strata::kernels::cpu::NativeFmt fmt;
        if(!strata::kernels::cpu::native_fmt(8,8,capture.setup.n_embd,capture.setup.n_ff,fmt,err))return 1;
        const auto expected=capture.setup.owned_count()*(uint64_t)fmt.bytes;
        ExpertWorkerClient client;WorkerCacheInfo before,after;
        if(!client.connect_local(argv[5],(uint16_t)port,5000,{},err) || !client.cache_status(before,err) ||
           !before.complete || before.session || before.resident_bytes!=expected || before.idle_remaining_ms!=kResidencyNoExpiry) {
            std::fprintf(stderr,"idle resident preflight refused: %s\n",err.c_str());return 1;
        }
        // Existing setup refuses a mismatching retained key without discarding its arena. No load path.
        auto setup=capture.setup;setup.nonce=(uint64_t)std::chrono::steady_clock::now().time_since_epoch().count();
        bool ok=client.setup_local(setup,local.identity,false,err,true) && client.cache_hit() &&
            client.reused_bytes()==expected && client.mark_ready(err);
        CpuReplayReport result;uint64_t sequence=0;
        const auto start=std::chrono::steady_clock::now();
        if(ok)ok=replay_cpu_capture(capture,[&](const InferRequest& saved,std::vector<float>& out,std::string& why){
            if(std::chrono::steady_clock::now()-start>std::chrono::seconds(180)) {why="replay total deadline";return false;}
            auto request=saved;request.session=client.session();request.sequence=++sequence;
            InferReply reply;if(!client.infer(request,reply,why))return false;out=std::move(reply.partials);return true;
        },result,err,references?CpuReplayObserver([&](size_t i,int p,const std::vector<float>& v){references->save(i,p,v);}):CpuReplayObserver{});
        const std::string primary_error=err;
        client.close(); // single-session disconnect; no unload, stop, upload or weight read
        ExpertWorkerClient check;
        const bool retained=check.connect_local(argv[5],(uint16_t)port,5000,{},err) &&
            check.cache_status(after,err) && unchanged(before,after);
        check.close();
        std::printf("CPU_CAPTURE_REPLAY ok=%d resident_unchanged=%d windows=%zu records=%zu full=%u grouped=%u "
            "maxabs=%.9g maxrel=%.9g relativeL2=%.9g first_difference=%u contributions=%u entry_bits=%d peer_bits=%d detail_envelope=%d\n",
            ok,retained,capture.windows.size(),capture.records.size(),result.full_checked,result.grouped_checked,
            result.max_abs,result.max_rel,result.relative_l2,result.first_difference,result.contribution_calls,
            result.entry_order_bits,result.peer_bits,result.detail_envelope);
        if(result.first_difference!=UINT32_MAX) {
            const auto& r=capture.records[result.first_difference];const auto& w=capture.windows[r.window-1];
            std::printf("FIRST_DIFFERENCE record=%u phase=%u position=%llu T=%u slice=%u layer=%u K=%u\n",
                result.first_difference,w.phase,(unsigned long long)w.pos,r.request.T,r.token_begin,r.request.layer,r.request.K);
        }
        if(result.detail_record!=UINT32_MAX) {
            const auto& r=capture.records[result.detail_record];const auto& w=capture.windows[r.window-1];
            std::printf("DETAIL record=%u phase=%u position=%llu T=%u slice=%u layer=%u K=%u maxabs=%.9g\n",
                result.detail_record,w.phase,(unsigned long long)w.pos,r.request.T,r.token_begin,r.request.layer,r.request.K,result.detail_max_abs);
        }
        if(!ok)std::fprintf(stderr,"replay refused/failed: %s\n",primary_error.c_str());
        if(!retained)std::fprintf(stderr,"resident verification failed: %s\n",err.c_str());
        ok=ok && retained && result.entry_order_bits && result.peer_bits && result.detail_envelope;
        if(ok && subset_config){
            diagnostic::SubsetReport subsets;
            const bool matched=diagnostic::run_local_subsets(capture,local,argv[2],*references,*subset_config,subsets,err);
            const auto subset_error=err;WorkerCacheInfo final_cache;ExpertWorkerClient final_check;
            const bool preserved=final_check.connect_local(argv[5],(uint16_t)port,5000,{},err) &&
                final_check.cache_status(final_cache,err) && unchanged(before,final_cache);final_check.close();
            std::printf("CPU_SEQUENTIAL_SUBSETS ok=%d full_cache_unchanged=%d peers_checked=%u completed_peers=%u released_peers=%u max_live_children=%u concurrent_group=NOT_RUN owned_reaped=%d peak_owned_rss=%llu min_available=%llu\n",
                matched,preserved,subsets.peers_checked,subsets.completed_peers,subsets.released_peers,subsets.max_live_children,subsets.reaped,
                (unsigned long long)subsets.peak_rss,(unsigned long long)subsets.min_available);
            if(!matched)std::fprintf(stderr,"local subset failure: %s\n",subset_error.c_str());
            if(!preserved)std::fprintf(stderr,"full resident final verification failure: %s\n",err.c_str());
            ok=matched && preserved && subsets.reaped;
        }
        return ok?0:1;
    } catch(const std::exception& e){std::fprintf(stderr,"replay exception: %s\n",e.what());return 1;}
}
