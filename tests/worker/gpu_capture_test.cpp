// Capture format/ownership bounds only; real CUDA capture is exercised by the diagnostic run.
#include "local_fixture.hpp"
#include "strata/worker/gpu_capture.hpp"
#include <sys/resource.h>
int main() {
    const rlimit memory{512ull<<20,512ull<<20},cpu{10,10},file{32ull<<20,32ull<<20};
    if (::setrlimit(RLIMIT_AS,&memory) || ::setrlimit(RLIMIT_CPU,&cpu) || ::setrlimit(RLIMIT_FSIZE,&file)) return 2;
    ::alarm(30);
    try {
        Fixture f; GpuPlan p; std::string error;
        require(make_gpu_plan(f.m,{{0,0},{0,1}},p,error),error);
        InferRequest r; r.session=7;r.sequence=1;r.model_id=p.model_id;r.partition_id=p.placement_id.a;
        r.layer=0;r.T=1;r.K=2;r.mask={1,1};r.ids={0,1};r.weights={.3f,.7f};r.activations.assign(p.n_embd,.25f);
        diagnostic::Capture c; require(gpu_codec::encode_request(p,r,c.request),"request encode");
        c.blob=f.blobs[0];c.rows.resize(2*p.n_embd*4);c.partial.resize(p.n_embd*4);
        c.input_q.resize(p.n_embd/32*36);c.hidden_q.resize(p.n_ff/32*36);
        auto encoded=diagnostic::encode(c); diagnostic::Capture decoded;InferRequest request;
        check(diagnostic::decode(p,encoded,decoded,request,error) && decoded.blob==c.blob,"bounded same-input capture roundtrip");
        auto truncated=encoded;truncated.pop_back();check(!diagnostic::decode(p,truncated,decoded,request,error),"truncated capture refused");
        auto changed=c;changed.blob[0]^=1;check(!diagnostic::decode(p,diagnostic::encode(changed),decoded,request,error),"resident weight corruption refused");
        auto wrong=p;wrong.model_id^=1;check(!diagnostic::decode(wrong,encoded,decoded,request,error),"wrong source binding refused");
        changed=c;changed.selected=2;check(!diagnostic::decode(p,diagnostic::encode(changed),decoded,request,error),"selected entry bounds");
        r.T=2;r.mask={1,1,1,1};r.ids={0,1,0,1};r.weights={.3f,.7f,.3f,.7f};r.activations.assign(2*p.n_embd,.25f);
        changed=c;require(gpu_codec::encode_request(p,r,changed.request),"two-token encode");
        check(!diagnostic::decode(p,diagnostic::encode(changed),decoded,request,error),"capture restricted to first one-token boundary");
        require(diagnostic::file(f.root.string(),"cuda.capture",encoded,true,error),error);
        check(!diagnostic::file(f.root.string(),"cuda.capture",encoded,true,error),"existing evidence never replaced");
        std::vector<uint8_t> read;require(diagnostic::file(f.root.string(),"cuda.capture",read,false,error),error);check(read==encoded,"exact capture file readback");
        diagnostic::Merge merge;merge.request=c.request;merge.before=merge.gpu=merge.cpu=merge.after=c.partial;merge.cpu_mask={0,0};
        auto m=diagnostic::encode(merge);diagnostic::Merge md;
        check(diagnostic::decode(m,md,error) && md.request==c.request,"merge request identity retained");
        m.push_back(0);check(!diagnostic::decode(m,md,error),"merge trailing data refused");
        std::vector<uint8_t> oversized(diagnostic::cap+1);
        check(!diagnostic::file(f.root.string(),"oversized",oversized,true,error),"aggregate capture file bound");
        if (failures) return 1;
        std::puts("GPU_CAPTURE_FIXTURES_PASS_V1 models=0 cuda=0"); return 0;
    } catch (const std::exception& e) {std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
