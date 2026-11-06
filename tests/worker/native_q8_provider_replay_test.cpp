#include "local_fixture.hpp"
#include "strata/worker/cpu_capture.hpp"
#include "strata/worker/gpu_worker.hpp"
#include <iomanip>
#include <sstream>
#ifdef STRATA_TEST_TLS
#include "tls_fixture.hpp"
#endif
struct MockProviderExecutor final : GpuExecutor {
    bool native_q8() const override { return true; }
    bool supports_rows() const override { return true; }
    bool prepare(const GpuPlan& p,uint64_t,GpuCache& cache,GpuCounters& c,std::string& error) override {
        std::vector<uint8_t> b; for(const auto& a:p.assets)if(!cache.read(a,b,error))return false;
        c.gpu_weight_bytes=c.gpu_allocated_bytes=p.assigned_bytes();c.gpu_allocation_count=1;return true;
    }
    bool infer(const GpuPlan&,const InferRequest&,InferReply&,std::string&) override {return false;}
    bool infer_rows(const GpuPlan& p,const InferRequest& q,EntryRowsReply& out,std::string&) override {
        out.session=q.session;out.sequence=q.sequence;
        for(size_t i=0;i<q.ids.size();++i)if(q.mask[i]){out.indices.push_back(uint16_t(i));out.rows.insert(out.rows.end(),p.n_embd,float(q.ids[i])+q.activations[0]);}
        return true;
    }
    void release(GpuCounters& c)override{c.gpu_weight_bytes=c.gpu_allocated_bytes=c.gpu_allocation_count=0;}
};
int invoke_provider(const std::vector<std::string>& args) {
    auto pid=fork();require(pid>=0,"fork failed");
    if(!pid){std::vector<char*> v;for(auto& s:args)v.push_back(const_cast<char*>(s.c_str()));v.push_back(nullptr);execv(v[0],v.data());_exit(127);}
    int status=0;require(waitpid(pid,&status,0)==pid && WIFEXITED(status),"child wait failed");return WEXITSTATUS(status);
}
int main(int argc,char** argv) {try {
    require(argc==2,"provider replay executable required");Fixture f(12);std::string error;
    auto setup=f.setup();CpuCapture capture;auto path=(f.root/"capture.bin").string();
    require(capture.prepare(setup,f.m.identity,f.m.index_digest,path,error,1u<<20),error);
    require(capture.request("GEN 1 1",{1,1,1},1) && capture.window(0,2,1),capture.error());
    InferRequest full;full.model_id=setup.model_id;full.partition_id=setup.partition_id;full.session=full.sequence=1;
    full.T=2;full.K=10;full.mask.assign(20,1);full.activations.resize(2*2560);full.weights.resize(20);full.ids.resize(20);
    for(int t=0;t<2;++t){std::fill_n(full.activations.begin()+t*2560,2560,0.1f+0.1f*t);for(int k=0;k<10;++k){full.ids[t*10+k]=(k+t)%12;full.weights[t*10+k]=float(k+1)/55;}}
    std::vector<float> partial(2*2560,0.0f);
    for(int layer=0;layer<2;++layer){full.layer=layer;full.sequence=layer+1;require(capture.begin(full)&&capture.finish(partial.data()),"capture record failed");}
    require(capture.window(2,1,2),capture.error());
    InferRequest final=full;final.T=1;final.activations.resize(2560);final.ids.resize(10);final.weights.resize(10);final.mask.resize(10);
    for(int layer=0;layer<2;++layer){final.layer=layer;require(capture.begin(final)&&capture.finish(partial.data()),capture.error());}
    float logit=0;require(capture.seal(&logit,1),"capture seal failed");
    std::vector<std::pair<uint16_t,uint16_t>> pairs;for(uint16_t i=0;i<12;++i)pairs.emplace_back(0,i);
    GpuPlan plan;require(make_gpu_plan(f.m,pairs,plan,error),error);auto plan_path=(f.root/"plan").string();require(write_gpu_plan(plan_path,plan,error),error);
    fs::create_directory(f.root/"blobs");GpuCache staged;require(staged.open((f.root/"blobs").string(),64ull<<20,error),error);
    std::vector<uint8_t> missing;require(staged.prepare(plan,missing,error),error);
    for(const auto& asset:plan.assets){const auto& bytes=f.blobs[asset.expert];require(staged.publish(asset,bytes.data(),bytes.size(),error),error);}
    const auto first_blob=f.root/"blobs"/staged.key(plan.assets[0]);staged.close();
    for(int token=0;token<2;++token) {
        InferRequest q=full;q.layer=0;q.T=1;q.model_id=plan.model_id;q.partition_id=plan.placement_id.a;q.session=q.sequence=1;
        q.activations.assign(full.activations.begin()+token*2560,full.activations.begin()+(token+1)*2560);
        q.ids.assign(full.ids.begin()+token*10,full.ids.begin()+(token+1)*10);q.weights.assign(full.weights.begin()+token*10,full.weights.begin()+(token+1)*10);q.mask.assign(10,1);
        MockProviderExecutor mock;EntryRowsReply reply;require(mock.infer_rows(plan,q,reply,error),error);
        std::vector<uint8_t> request,ref;require(gpu_codec::encode_request(plan,q,request)&&gpu_codec::encode_rows_reply(plan,blob_digest(request),q,reply,ref),"direct mock encode failed");
        auto direct=f.root/("direct-"+std::to_string(token));fs::create_directory(direct);
        {std::ofstream o(direct/"layer.direct.reply",std::ios::binary);o.write((const char*)ref.data(),ref.size());}
        {std::ofstream o(direct/"DIRECT_PASS");o<<"direct replay complete\n";}
        auto cache=f.root/("worker-"+std::to_string(token));fs::create_directory(cache);
        GpuWorkerOptions options;options.vulkan=options.native_q8=true;options.runtime_reserve=0;options.plan=plan;options.cache_root=cache.string();options.cache_budget=64ull<<20;
#ifdef STRATA_TEST_TLS
        TlsFixture tls;if(token==0)options.tls=tls.config("worker","coordinator.test");
#endif
        GpuWorkerService service(options,std::make_unique<MockProviderExecutor>());uint16_t port=0;require(service.listen(port,error),error);
        std::thread server([&]{service.run();});struct Join{GpuWorkerService& s;std::thread& t;~Join(){s.stop();t.join();}} join{service,server};
        std::vector<std::string> args={argv[1],"--provider","127.0.0.1:"+std::to_string(port),path,"0",plan_path,(f.root/"blobs").string(),(direct/"layer.direct.reply").string(),(f.root/("out-"+std::to_string(token))).string(),"--deadline-ms","10000","--token-index",std::to_string(token)};
#ifdef STRATA_TEST_TLS
        if(token==0){const auto c=tls.config("coordinator","worker.test");args.insert(args.end(),{"--tls-ca",c.ca_file,"--tls-cert",c.cert_file,"--tls-key",c.key_file,"--tls-peer",c.peer_dns});}
#endif
        require(invoke_provider(args)==0 && fs::exists(fs::path(args[8])/"PASS"),"provider replay failed");
        args[8]=(f.root/("wrong-reference-"+std::to_string(token))).string();args[12]=std::to_string(1-token);
        require(invoke_provider(args)!=0 && fs::exists(fs::path(args[8])/"FAILURE"),"mismatched token reference accepted");
        args[8]=(f.root/("invalid-token-"+std::to_string(token))).string();args[12]="2";
        require(invoke_provider(args)!=0 && fs::exists(fs::path(args[8])/"FAILURE"),"invalid token accepted");
        args[8]=(f.root/("missing-blob-"+std::to_string(token))).string();args[12]=std::to_string(token);
        fs::rename(first_blob,first_blob.string()+".held");
        const int missing_result=invoke_provider(args);fs::rename(first_blob.string()+".held",first_blob);
        require(missing_result!=0 && fs::exists(fs::path(args[8])/"FAILURE"),"missing blob accepted");
#ifdef STRATA_TEST_TLS
        if(token==0){args[8]=(f.root/"wrong-san").string();args.back()="wrong.test";
            require(invoke_provider(args)!=0 && fs::exists(fs::path(args[8])/"FAILURE") && !fs::exists(fs::path(args[8])/"PASS"),"wrong SAN accepted");}
#endif
    }
    std::puts("T2 capture split into two original T1 provider comparisons; mock CPU executor only");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"ERROR %s\n",e.what());return 1;}}
