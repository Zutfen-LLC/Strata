// New trace/attach deltas only; synthetic weights, no engine/CUDA/model.
#include "local_fixture.hpp"
#include "strata/worker/layer_trace.hpp"
#include <sys/resource.h>
namespace L=strata::worker::diagnostic::layer;
namespace {
void trace_cases(const Fixture& f){
    L::Record row{L::Host,0,2,2560,10,2,{L::bytes(f.blobs[0].data(),100)}};
    auto raw=L::encode(row);L::Record out;std::string error;
    check(L::decode(raw,out,error)&&out.fields==row.fields,"trace roundtrip");
    auto short_record=raw;short_record.pop_back();check(!L::decode(short_record,out,error),"truncated trace refuses");
    raw.push_back(0);check(!L::decode(raw,out,error),"trailing bytes refuse");
    row.layers=129;bool refused=false;try{(void)L::encode(row);}catch(...){refused=true;}check(refused,"layer count cap");
    row.layers=2;row.fields={std::vector<uint8_t>(L::record_cap)};refused=false;try{(void)L::encode(row);}catch(...){refused=true;}check(refused,"aggregate record cap");
    row.fields={{1,2,3}};require(L::write(f.root.c_str(),row,error),error);check(!L::write(f.root.c_str(),row,error),"never replace trace evidence");
    const std::vector<float> a{1.f,2.f},tiny{1.f+1e-6f,2.f},large{1.01f,2.f};
    check(!L::difference(a,tiny).bitwise&&!L::difference(a,tiny).material,"tiny floating difference separated from diagnostic drift");
    check(L::difference(a,large).material,"material divergence classified");
    refused=false;try{(void)L::difference(a,{NAN,2.f});}catch(...){refused=true;}check(refused,"nonfinite comparison refuses");
}
void attach_cases(Fixture& f){
    auto options=f.options();options.residency_forever=true;Service service(options);std::string error;
    ExpertWorkerClient client;require(client.connect_local("127.0.0.1",service.port,1000,{},error),error);
    require(client.setup_local(f.setup(),f.m.identity,false,error,true),error);std::vector<uint8_t> hits;
    require(client.load_local({{0,0},{0,1},{1,0},{1,1}},hits,error)&&client.mark_ready(error),error);
    WorkerCacheInfo cold;require(client.cache_status(cold,error),error);client.close();
    ExpertWorkerTier::Open open;open.port=service.port;open.resident=true;open.local=&f.m;
    open.n_layers=2;open.n_expert=2;open.n_embd=2560;open.n_ff=640;open.partition={3,3};
    int fetches=0;open.fetch=[&](int64_t,int64_t,uint8_t*,uint32_t){++fetches;return false;};
    ::setenv("STRATA_REQUIRE_CPU_RESIDENT_HIT","1",1);
    {ExpertWorkerTier tier;require(tier.open(open,error),error);check(tier.stats().cache_hit&&tier.stats().reused_bytes==cold.resident_bytes&&!tier.stats().setup_network_bytes&&!fetches,"attach exact existing partition without load or network");}
    {auto changed=open;changed.partition={1,3};ExpertWorkerTier tier;check(!tier.open(changed,error),"incompatible existing partition refuses");}
    require(client.connect_local("127.0.0.1",service.port,1000,{},error)&&client.unload(error),error);client.close();
    {ExpertWorkerTier tier;check(!tier.open(open,error),"empty resident service refuses cold fallback");}
    require(client.connect_local("127.0.0.1",service.port,1000,{},error),error);WorkerCacheInfo after;require(client.cache_status(after,error),error);
    check(after.loaded_bytes==cold.loaded_bytes&&!after.resident_bytes&&!fetches,"refused attach never reads/uploads weights");
    client.close();::unsetenv("STRATA_REQUIRE_CPU_RESIDENT_HIT");
}
}
int main(){
    const rlimit memory{512ull<<20,512ull<<20},cpu{10,10},file{32ull<<20,32ull<<20};
    if(setrlimit(RLIMIT_AS,&memory)||setrlimit(RLIMIT_CPU,&cpu)||setrlimit(RLIMIT_FSIZE,&file))return 2;alarm(30);
    try{Fixture fixture;trace_cases(fixture);attach_cases(fixture);}
    catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());++failures;}
    std::printf("FIRST_LAYER_FIXTURES_V1 failures=%d models=0 cuda=0\n",failures);return failures?1:0;
}
