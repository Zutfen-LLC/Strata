#include "strata/worker/gpu_client.hpp"
#include "strata/worker/gpu_worker.hpp"
#include "strata/worker/cpu_capture.hpp"
#include <filesystem>
#include <fstream>
#include <thread>
#include <sys/wait.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
using namespace strata::worker;
namespace fs=std::filesystem;
void check(bool value,const char* text) {if(!value)throw std::runtime_error(text);}
struct Mock final:GpuExecutor {
    bool native_q8()const override{return true;}
    bool supports_rows()const override{return true;}
    bool prepare(const GpuPlan& p,uint64_t,GpuCache& cache,GpuCounters& c,std::string& err) override {
        std::vector<uint8_t> blob;for(const auto& a:p.assets)if(!cache.read(a,blob,err))return false;
        c.gpu_weight_bytes=c.gpu_allocated_bytes=p.assigned_bytes();c.gpu_allocation_count=1;return true;
    }
    bool infer(const GpuPlan&,const InferRequest&,InferReply&,std::string&)override{return false;}
    bool infer_rows(const GpuPlan& p,const InferRequest& q,EntryRowsReply& out,std::string&)override {
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
    GpuWorkerService service(options,std::make_unique<Mock>());uint16_t port=0;check(service.listen(port,err),err.c_str());
    std::thread server([&]{service.run();});struct Stop{GpuWorkerService& s;std::thread& t;~Stop(){s.stop();t.join();}} stop{service,server};
    auto args=std::vector<std::string>{argv[1],"--worker","127.0.0.1:"+std::to_string(port),(fixture/"capture.bin").string(),"0",(fixture/"one-layer.plan").string(),(fixture/"blobs").string(),(dir/"direct"/"layer.direct.reply").string(),(dir/"out").string(),"--deadline-ms","10000"};
    check(invoke(args)==0 && fs::exists(dir/"out"/"PASS"),"mock replay did not pass");
    args[8]=(dir/"bad").string();ref.back()^=1;{std::ofstream f(dir/"direct"/"layer.direct.reply",std::ios::binary|std::ios::trunc);f.write((char*)ref.data(),ref.size());}
    check(invoke(args)!=0 && fs::exists(dir/"bad"/"FAILURE") && !fs::exists(dir/"bad"/"PASS"),"changed reference accepted");
    {std::ofstream f(dir/"direct"/"layer.direct.reply",std::ios::binary|std::ios::trunc);ref.back()^=1;f.write((char*)ref.data(),ref.size());}
    fs::rename(dir/"direct"/"DIRECT_PASS",dir/"direct"/"held-marker");args[8]=(dir/"missing-marker").string();
    check(invoke(args)!=0 && fs::exists(dir/"missing-marker"/"FAILURE"),"missing completion marker accepted");
    fs::rename(dir/"direct"/"held-marker",dir/"direct"/"DIRECT_PASS");
    fs::rename(fixture/"blobs",fixture/"held-blobs");args[8]=(dir/"missing-blob").string();
    check(invoke(args)!=0 && fs::exists(dir/"missing-blob"/"FAILURE"),"missing source blob accepted");
    fs::rename(fixture/"held-blobs",fixture/"blobs");
    std::printf("mock replay comparison, reference, marker and blob refusals passed; no GPU\n");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
