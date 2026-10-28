// Tiny owned Q8 sources/children only. No model, engine, external endpoint or desktop execution.
#include "local_fixture.hpp"
#include "cpu_capture_micro.hpp"
using namespace strata::worker::diagnostic;
namespace {
void readiness_model(){
    auto trace=std::make_unique<strata::kernels::cpu::ExpertPool::NativeReadiness>();
    trace->count=3;trace->down_start_ns=1000;trace->down_done_ns=5000;trace->pool_done_ns=5100;
    trace->tasks[0].done_ns=4000;trace->tasks[1].done_ns=2000;trace->tasks[2].done_ns=3000;
    // Two 4-float rows; task1 straddles both experts. T2 revisits job0 after job1.
    const auto m=reply_readiness(*trace,2,4,{0,-1,1,0},0);
    check(m.bytes==90 && m.first_range==2000 && m.first_prefix==4000 && m.first_row==3000 && m.first_wire_row==4000,
        "readiness distinguishes any completed range from the wire prefix");
    check(m.prefix_finish==4416 && m.row_finish==4432 && m.batch_finish==5820,
        "readiness ascending byte and whole-row serializers include indices and repeated jobs");
    trace->tasks[0].done_ns=6000;bool refused=false;
    try{reply_readiness(*trace,2,4,{0},0);}catch(const std::exception&){refused=true;}
    check(refused,"readiness refuses completion after parking");
}
CpuCaptureFile capture(const Fixture& fixture){
    CpuCaptureFile f;f.setup=fixture.setup();f.setup.partition={165,165}; // CPU IDs0,2,5,7: filter must not renumber owners
    f.setup.partition_id=blob_digest(f.setup.partition).a;f.identity=fixture.m.identity;f.index=fixture.m.index_digest;
    CpuCaptureRecord r;auto& q=r.request;q.layer=0;q.T=2;q.K=10;q.session=q.sequence=1;
    q.model_id=f.setup.model_id;q.partition_id=f.setup.partition_id;
    q.activations.resize(5120);for(size_t i=0;i<5120;++i)q.activations[i]=(int(i%17)-8)*.0625f;
    q.ids.resize(20);q.mask.resize(20);q.weights.resize(20);std::vector<const uint8_t*> blobs(20);
    for(size_t i=0;i<20;++i){q.ids[i]=i%3==0?3:i%3==1?2:5;q.mask[i]=f.setup.owns(0,q.ids[i]);
        q.weights[i]=i%2?-.25f:.5f;blobs[i]=fixture.blobs[q.ids[i]].data();r.sent=r.sent||q.mask[i];}
    std::string err;require(ExpertWorkerService::direct_native_partial(fixture.fmt,2,10,q.mask.data(),q.weights.data(),
        q.activations.data(),blobs.data(),r.partials,err),err);f.records.push_back(std::move(r));return f;
}
void test(const char* binary){
    Fixture fixture(8);const auto f=capture(fixture);const auto p=capture_micro_plan(f,fixture.m);
    check(p.experts==std::vector<uint16_t>({2,5}) && p.owners[2]==1 && p.owners[5]==0 &&
        p.partition[0][0]==32 && p.partition[1][0]==4,"original full-bitmap parity retained before filtering");
    check(p.bytes[0]+p.bytes[1]==2*fixture.fmt.bytes && p.bytes[0]+p.bytes[1]<=kMicroPayloadMax,"selected arena bound");
    auto bad=f;bad.records[0].request.T=1;bool refused=false;
    try{capture_micro_plan(bad,fixture.m);}catch(const std::exception&){refused=true;}
    check(refused,"wrong record shape refused");
    bad=f;bad.records[0].request.mask[0]=1;refused=false;
    try{capture_micro_plan(bad,fixture.m);}catch(const std::exception&){refused=true;}
    check(refused,"GPU expert cannot become CPU work");
    bad=f;for(size_t i=0;i<20;++i){bad.records[0].request.ids[i]=5;bad.records[0].request.mask[i]=1;}
    refused=false;try{capture_micro_plan(bad,fixture.m);}catch(const std::exception&){refused=true;}
    check(refused,"single-active-peer input refused by this two-peer discriminator");
    auto run=[&](const CpuCaptureFile& input,const char* name,const std::string& b,MicroReport& report,std::string& err){
        return run_capture_micro(input,fixture.m,(fixture.root/"local.manifest").string(),fixture.root.string(),
            (fixture.root/name).string(),binary,b,report,err);};
    MicroReport report;std::string err;
    bad=f;bad.records[0].partials[0]+=1.f;
    check(!run(bad,"bad-reference",binary,report,err) && !report.reference && report.children[0]<0 &&
        report.children[1]<0 && report.reaped && !fs::exists(fixture.root/"bad-reference"),
        "full reference mismatch prevents all worker startup");
    const auto& range=fixture.m.layers[0][0];const auto offset=range.offset+2*range.bytes+11;
    const auto name=fixture.m.files[range.file].name;
    fixture.flip(name.c_str(),offset);
    check(!run(f,"bad-digest",binary,report,err) && !report.reference && report.children[0]<0 &&
        err=="micro selected source digest mismatch","corrupt selected source refuses before any worker");
    fixture.flip(name.c_str(),offset);
    check(!run(f,"second-child-fails","/bin/false",report,err) && report.reference && report.children[0]>0 &&
        report.children[1]<0 && report.reaped && micro_reaped(report.children[0]),"second child startup failure reaps positive owned first child");
    bool delayed=false;
    require(run_capture_micro(f,fixture.m,(fixture.root/"local.manifest").string(),fixture.root.string(),
        (fixture.root/"success").string(),binary,binary,report,err,[&]{
            delayed=true;std::this_thread::sleep_for(std::chrono::seconds(11));},true),err);
    check(report.reference && report.peer[0] && report.peer[1] && report.group && report.reaped &&
        report.payload==2*fixture.fmt.bytes && report.children[0]>0 && report.children[1]>0,
        "independent full bits then real selected peers and group/manual bits");
    check(delayed,"first READY peer kept alive beyond its 10s idle deadline during second setup");
    check(micro_reaped(report.children[0]) && micro_reaped(report.children[1]),"successful owned children reaped");
}
}
int main(int argc,char** argv){
    if(argc!=2)return 2;
    try{readiness_model();test(argv[1]);}catch(const std::exception& e){std::fprintf(stderr,"CPU_CAPTURE_MICRO_TEST exception: %s\n",e.what());return 2;}
    std::printf("CPU_CAPTURE_MICRO_TEST failures=%d\n",failures);return failures?1:0;
}
