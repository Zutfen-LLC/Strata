#include "strata/worker/gpu_client.hpp"
#include "strata/worker/gpu_worker.hpp"
#include "strata/worker/cpu_capture.hpp"
#include <filesystem>
#include <fstream>
#include <thread>
#include <atomic>
#include <sys/wait.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#ifdef STRATA_TEST_TLS
#include "tls_fixture.hpp"
#endif
using namespace strata::worker;
namespace fs=std::filesystem;
void check(bool value,const char* text) {if(!value)throw std::runtime_error(text);}
struct Mock final:GpuExecutor {
    std::atomic<int>* calls=nullptr;
    bool change_allocation=false,change_requests=false,change_failures=false;GpuCounters* counters=nullptr;
    bool native_q8()const override{return true;}
    bool supports_rows()const override{return true;}
    bool prepare(const GpuPlan& p,uint64_t,GpuCache& cache,GpuCounters& c,std::string& err) override {
        counters=&c;std::vector<uint8_t> blob;for(const auto& a:p.assets)if(!cache.read(a,blob,err))return false;
        c.gpu_weight_bytes=c.gpu_allocated_bytes=p.assigned_bytes();c.gpu_allocation_count=1;return true;
    }
    bool infer(const GpuPlan&,const InferRequest&,InferReply&,std::string&)override{return false;}
    bool infer_rows(const GpuPlan& p,const InferRequest& q,EntryRowsReply& out,std::string&)override {
        if(calls)++*calls;
        if(change_allocation && counters)++counters->gpu_allocated_bytes;
        if(change_requests && counters)++counters->requests;
        if(change_failures && counters)++counters->failures;
        out.session=q.session;out.sequence=q.sequence;
        for(size_t i=0;i<q.mask.size();++i)if(q.mask[i]){out.indices.push_back(i);out.rows.insert(out.rows.end(),p.n_embd,float(i));}
        return true;
    }
    void release(GpuCounters& c)override{c.gpu_allocated_bytes=c.gpu_weight_bytes=c.gpu_allocation_count=0;}
};
int invoke(const std::vector<std::string>& args) {
    const auto child=fork();check(child>=0,"fork");if(!child){std::vector<char*> a;for(const auto& s:args)a.push_back(const_cast<char*>(s.c_str()));a.push_back(nullptr);execv(a[0],a.data());_exit(127);}
    int status;check(waitpid(child,&status,0)==child && WIFEXITED(status),"wait");return WEXITSTATUS(status);
}
int main(int argc,char** argv){try {
    check(argc==3,"expected replay and fixture binaries");char tmp[]="/tmp/native-q8-worker-test-XXXXXX";const char* root=mkdtemp(tmp);check(root,"mkdtemp");fs::path dir=root,fixture=dir/"fixture";
    check(invoke({argv[2],"--stage-check80",fixture.string()})==0,"stage fixture");
    std::string err;CpuCaptureFile capture;GpuPlan plan;check(read_cpu_capture((fixture/"capture.bin").string(),capture,err),err.c_str());
    check(read_gpu_plan((fixture/"one-layer.plan").string(),plan,err),err.c_str());auto q=capture.records[0].request;
    q.model_id=plan.model_id;q.partition_id=plan.placement_id.a;q.session=q.sequence=1;
    std::vector<uint8_t> req,ref;check(gpu_codec::encode_request(plan,q,req),"request encode");
    Mock maker;EntryRowsReply direct;check(maker.infer_rows(plan,q,direct,err),"mock direct");
    check(gpu_codec::encode_rows_reply(plan,blob_digest(req),q,direct,ref),"reference encode");
    fs::create_directory(dir/"direct");{std::ofstream f(dir/"direct"/"layer.direct.reply",std::ios::binary);f.write((char*)ref.data(),ref.size());}
    {std::ofstream f(dir/"direct"/"DIRECT_PASS");f<<"direct replay complete\n";}
    fs::create_directory(dir/"cache");GpuWorkerOptions options;options.vulkan=options.native_q8=true;options.runtime_reserve=0;
    options.plan=plan;options.cache_root=(dir/"cache").string();options.cache_budget=64ull<<20;
    std::atomic<int> inference_calls{0};auto mock=std::make_unique<Mock>();mock->calls=&inference_calls;
    GpuWorkerService service(options,std::move(mock));uint16_t port=0;check(service.listen(port,err),err.c_str());
    std::thread server([&]{service.run();});struct Stop{GpuWorkerService& s;std::thread& t;~Stop(){s.stop();t.join();}} stop{service,server};
    auto args=std::vector<std::string>{argv[1],"--worker","127.0.0.1:"+std::to_string(port),(fixture/"capture.bin").string(),"0",(fixture/"one-layer.plan").string(),(fixture/"blobs").string(),(dir/"direct"/"layer.direct.reply").string(),(dir/"out").string(),"--deadline-ms","10000"};
    check(invoke(args)==0 && fs::exists(dir/"out"/"PASS"),"mock replay did not pass");
    args[8]=(dir/"dirty-before").string();
    check(invoke(args)!=0 && fs::exists(dir/"dirty-before"/"worker.status") && fs::exists(dir/"dirty-before"/"FAILURE") && !fs::exists(dir/"dirty-before"/"PASS"),"cumulative failure before call accepted");
    { std::ifstream f(dir/"dirty-before"/"worker.status");std::string session,before;std::getline(f,session);std::getline(f,before);
      check(before.find("failures=1")!=std::string::npos,"dirty replay did not record the prior failure"); }
    args[8]=(dir/"bad").string();ref.back()^=1;{std::ofstream f(dir/"direct"/"layer.direct.reply",std::ios::binary|std::ios::trunc);f.write((char*)ref.data(),ref.size());}
    check(invoke(args)!=0 && fs::exists(dir/"bad"/"FAILURE") && !fs::exists(dir/"bad"/"PASS"),"changed reference accepted");
    {std::ofstream f(dir/"direct"/"layer.direct.reply",std::ios::binary|std::ios::trunc);ref.back()^=1;f.write((char*)ref.data(),ref.size());}
    fs::rename(dir/"direct"/"DIRECT_PASS",dir/"direct"/"held-marker");args[8]=(dir/"missing-marker").string();
    check(invoke(args)!=0 && fs::exists(dir/"missing-marker"/"FAILURE"),"missing completion marker accepted");
    fs::rename(dir/"direct"/"held-marker",dir/"direct"/"DIRECT_PASS");
    fs::rename(fixture/"blobs",fixture/"held-blobs");args[8]=(dir/"missing-blob").string();
    check(invoke(args)!=0 && fs::exists(dir/"missing-blob"/"FAILURE"),"missing source blob accepted");
    fs::rename(fixture/"held-blobs",fixture/"blobs");
    for(const std::string tag:{"cuda@","vulkan@","vulkan-native-q8@"}) {
        args[2]=tag+"127.0.0.1:"+std::to_string(port);args[8]=(dir/("tag-"+tag.substr(0,tag.size()-1))).string();
        check(invoke(args)!=0 && fs::exists(fs::path(args[8])/"FAILURE") && !fs::exists(fs::path(args[8])/"PASS"),"backend-tagged endpoint accepted");
    }
    {
        fs::create_directory(dir/"changing-cache");auto changing_options=options;changing_options.cache_root=(dir/"changing-cache").string();
        auto changed=std::make_unique<Mock>();changed->change_allocation=true;
        GpuWorkerService changing(changing_options,std::move(changed));uint16_t changing_port=0;check(changing.listen(changing_port,err),err.c_str());
        std::thread thread([&]{changing.run();});Stop guard{changing,thread};
        args[2]="127.0.0.1:"+std::to_string(changing_port);args[8]=(dir/"changed-allocation").string();
        check(invoke(args)!=0 && fs::exists(dir/"changed-allocation"/"FAILURE") && !fs::exists(dir/"changed-allocation"/"PASS"),"changed allocation accepted");
    }
    {
        fs::create_directory(dir/"changing-request-cache");auto changing_options=options;changing_options.cache_root=(dir/"changing-request-cache").string();
        auto changed=std::make_unique<Mock>();changed->change_requests=true;
        GpuWorkerService changing(changing_options,std::move(changed));uint16_t changing_port=0;check(changing.listen(changing_port,err),err.c_str());
        std::thread thread([&]{changing.run();});Stop guard{changing,thread};
        args[2]="127.0.0.1:"+std::to_string(changing_port);args[8]=(dir/"changed-requests").string();
        check(invoke(args)!=0 && fs::exists(dir/"changed-requests"/"FAILURE") && !fs::exists(dir/"changed-requests"/"PASS"),"changed request count accepted");
    }
    {
        fs::create_directory(dir/"changing-failure-cache");auto changing_options=options;changing_options.cache_root=(dir/"changing-failure-cache").string();
        auto changed=std::make_unique<Mock>();changed->change_failures=true;
        GpuWorkerService changing(changing_options,std::move(changed));uint16_t changing_port=0;check(changing.listen(changing_port,err),err.c_str());
        std::thread thread([&]{changing.run();});Stop guard{changing,thread};
        args[2]="127.0.0.1:"+std::to_string(changing_port);args[8]=(dir/"changed-failures").string();
        check(invoke(args)!=0 && fs::exists(dir/"changed-failures"/"FAILURE") && !fs::exists(dir/"changed-failures"/"PASS"),"nonzero failure count accepted");
        std::ifstream status(dir/"changed-failures"/"worker.status");std::string session,before,after;
        std::getline(status,session);std::getline(status,before);std::getline(status,after);
        check(before.find("failures=0")!=std::string::npos && after.find("failures=1")!=std::string::npos,"changed failure counters not retained");
    }
#ifdef STRATA_TEST_TLS
    {
        TlsFixture tls;fs::create_directory(dir/"tls-cache");auto secure_options=options;
        secure_options.cache_root=(dir/"tls-cache").string();secure_options.tls=tls.config("worker","coordinator.test");
        GpuWorkerService secure(secure_options,std::make_unique<Mock>());uint16_t tls_port=0;check(secure.listen(tls_port,err),err.c_str());
        std::thread thread([&]{secure.run();});Stop guard{secure,thread};
        args[2]="127.0.0.1:"+std::to_string(tls_port);args[8]=(dir/"tls-ok").string();
        const auto credentials=tls.config("coordinator","worker.test");
        args.insert(args.end(),{"--tls-ca",credentials.ca_file,"--tls-cert",credentials.cert_file,"--tls-key",credentials.key_file,"--tls-peer",credentials.peer_dns});
        check(invoke(args)==0 && fs::exists(dir/"tls-ok"/"PASS"),"mTLS replay did not pass");
        auto setup=std::vector<std::string>{argv[1],"--setup-only","127.0.0.1:"+std::to_string(tls_port),
            (fixture/"one-layer.plan").string(),"--deadline-ms","10000","--setup-ms","30000",
            "--tls-ca",credentials.ca_file,"--tls-cert",credentials.cert_file,"--tls-key",credentials.key_file,"--tls-peer",credentials.peer_dns};
        check(invoke(setup)==0,"mTLS cached setup-only failed");setup.back()="wrong.test";
        check(invoke(setup)!=0,"mTLS setup-only accepted wrong SAN");
        args[8]=(dir/"tls-bad-san").string();args.back()="wrong.test";
        check(invoke(args)!=0 && fs::exists(dir/"tls-bad-san"/"FAILURE") && !fs::exists(dir/"tls-bad-san"/"PASS"),"wrong TLS SAN accepted");
    }
#endif
    {
        auto setup=std::vector<std::string>{argv[1],"--setup-only","127.0.0.1:"+std::to_string(port),
            (fixture/"one-layer.plan").string(),"--deadline-ms","10000","--setup-ms","30000"};
        const int before=inference_calls.load();check(invoke(setup)==0,"cached setup-only failed");
        check(inference_calls.load()==before,"setup-only issued inference");
        setup[2]="cuda@"+setup[2];check(invoke(setup)!=0,"setup-only accepted tagged endpoint");
        fs::create_directory(dir/"empty-setup-cache");auto empty_options=options;empty_options.cache_root=(dir/"empty-setup-cache").string();
        auto empty_mock=std::make_unique<Mock>();empty_mock->calls=&inference_calls;
        GpuWorkerService empty(empty_options,std::move(empty_mock));uint16_t empty_port=0;check(empty.listen(empty_port,err),err.c_str());
        std::thread thread([&]{empty.run();});Stop guard{empty,thread};
        setup[2]="127.0.0.1:"+std::to_string(empty_port);
        check(invoke(setup)!=0 && inference_calls.load()==before,"missing cache or unexpected inference accepted");
    }
    std::printf("mock replay rows, lifecycle, tags and refusal checks passed; no GPU\n");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
