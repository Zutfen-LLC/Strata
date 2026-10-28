#include "../../src/core/portable_local_experts.hpp"
#include "strata/worker/vulkan_executor.hpp"
#include "portable_model_fixture.hpp"
#include <cstdio>
using namespace strata::core;
using namespace model_fixture;
namespace {
struct State { int prepares=0,releases=0,inferences=0; bool fail=false,bad=false,nonfinite=false; };
class Fake final:public strata::worker::GpuExecutor {
    State& s;
public:
    explicit Fake(State& state):s(state) {}
    ~Fake() override { strata::worker::GpuCounters c;release(c); }
    bool live=false;
    bool prepare(const strata::worker::GpuPlan&,uint64_t,strata::worker::GpuCache&,
                 strata::worker::GpuCounters&,std::string&) override {return false;}
    bool prepare_source(const strata::worker::GpuPlan& p,const strata::worker::GpuAssetReader& reader,
                        strata::worker::GpuCounters& c,std::string& error) override {
        ++s.prepares;live=true;
        for(const auto& a:p.assets) {
            std::vector<uint8_t> b;
            if(!reader(a,b,error)) return false;
            check(b.size()==a.bytes && strata::worker::blob_digest(b)==a.digest,"direct mapped digest");
        }
        c.gpu_allocated_bytes=p.assigned_bytes()+4096;
        c.gpu_weight_bytes=p.assigned_bytes();return true;
    }
    bool supports_rows()const override{return true;}
    bool infer(const strata::worker::GpuPlan&,const strata::worker::InferRequest&,
               strata::worker::InferReply&,std::string&)override{return false;}
    bool infer_rows(const strata::worker::GpuPlan&,const strata::worker::InferRequest& r,
                    strata::worker::EntryRowsReply& out,std::string& error) override {
        ++s.inferences;if(s.fail){error="controlled GPU failure";return false;}
        out={};out.session=r.session;out.sequence=r.sequence;
        for(size_t i=0;i<r.mask.size();++i)if(r.mask[i]) {
            out.indices.push_back(uint16_t(i));
            out.rows.insert(out.rows.end(),2560,float(r.ids[i]+1));
        }
        if(s.bad)out.indices.back()=out.indices.front();
        if(s.nonfinite)out.rows[0]=NAN;
        return true;
    }
    void release(strata::worker::GpuCounters&)override {if(live){++s.releases;live=false;}}
};
}
int main(){
    try {
        SparseModel f;strata::GgufModel model(f.paths);PortableModelOptions opts;
        portable_detail::CpuExperts cpu(model,(f.root/"manifest").string(),opts);
        auto plan=[&](const std::string& name,std::vector<std::pair<uint16_t,uint16_t>> ids){
            strata::worker::GpuPlan p;std::string e;
            check(strata::worker::make_gpu_plan(cpu.identity(),ids,p,e),e);
            const auto path=(f.root/name).string();check(strata::worker::write_gpu_plan(path,p,e),e);return path;
        };
        opts.accelerators={reinterpret_cast<ggml_backend_t>(uintptr_t(1)),reinterpret_cast<ggml_backend_t>(uintptr_t(2))};
        PortableExpertDevice a,b;a.plan_path=plan("a",{{0,1},{0,3}});b.plan_path=plan("b",{{0,5}});b.accelerator=1;
        opts.expert_devices={a,b};State state;size_t free=7ull<<30;
        auto factory=[&](ggml_backend_t,size_t){return std::make_unique<Fake>(state);};
        auto memory=[&](ggml_backend_t){return std::make_pair(free,size_t(8ull<<30));};
        PortableRoute route;route.activation.assign(2560,.1f);route.weights.assign(10,.1f);
        route.ids={9,3,7,1,5,2,4,8,6,0};std::string error;std::vector<float> ref,rows;
        check(cpu.run(0,route,ref,error),error);
        {
            portable_detail::LocalExperts local(model,cpu,opts,factory,memory);
            check(state.prepares==2,"one executor per die");
            check(local.run(0,route,rows,error),error);
            for(size_t slot=0;slot<10;++slot)for(size_t h=0;h<2560;++h)
                check(rows[slot*2560+h]==((route.ids[slot]==1||route.ids[slot]==3||route.ids[slot]==5)?float(route.ids[slot]+1):ref[slot*2560+h]),"original-slot hot/cold rows");
            check(local.gpu_entries()==3 && local.cpu_entries()==7 && local.report()[0].entries==2 && local.report()[1].entries==1,"offload counters");
            const auto cold=cpu.work();
            check(local.run(0,route,rows,error),error);
            check(state.prepares==2 && cpu.work().verifications==cold.verifications && cpu.work().hits==cold.hits+7,
                  "serial request reuse retains GPU plans and CPU verification");
            check(local.gpu_entries()==6 && local.cpu_entries()==14,"cumulative request entries persist");
            auto bad=route;bad.ids.pop_back();check(!local.run(0,bad,rows,error)&&rows.empty(),"shape refusal");
            state.bad=true;check(!local.run(0,route,rows,error)&&rows.empty(),"duplicate reply slot refusal");state.bad=false;
            state.nonfinite=true;check(!local.run(0,route,rows,error)&&rows.empty(),"nonfinite reply refusal");state.nonfinite=false;
            state.fail=true;check(!local.run(0,route,rows,error)&&rows.empty(),"failure no CPU replay");state.fail=false;
            free=1;check(!local.run(0,route,rows,error)&&rows.empty(),"runtime free guard");free=7ull<<30;
            check(local.run(0,route,rows,error),"provider reused after caller resets sequence state");
            const auto old=std::filesystem::last_write_time(f.paths[0]);
            std::filesystem::last_write_time(f.paths[0],old+std::chrono::seconds(2));
            check(!local.run(0,route,rows,error)&&rows.empty(),"resident source identity stop");
            std::filesystem::last_write_time(f.paths[0],old);
        }
        check(state.releases==2,"both executors released");
        auto rejected=[&](PortableModelOptions bad,const char* why){
            const int before=state.prepares;bool failed=false;
            try{portable_detail::LocalExperts local(model,cpu,bad,factory,memory);}catch(const std::exception&){failed=true;}
            check(failed,why);check(state.prepares==before,"refusal before device allocation");
        };
        auto bad=opts;bad.expert_devices[1].plan_path=a.plan_path;rejected(bad,"cross-die duplicate ownership");
        bad=opts;bad.expert_devices[0].tensor_cap=1;rejected(bad,"tensor cap refusal");
        bad=opts;bad.expert_devices[1].accelerator=0;rejected(bad,"duplicate die refusal");
        free=1;rejected(opts,"per-die memory refusal");free=7ull<<30;
        // Real existing executor, CPU graph substitution: direct loader, no disk GpuCache.
        strata::worker::GpuPlan p;check(strata::worker::read_gpu_plan(a.plan_path,p,error),error);
        auto executor=strata::worker::create_vulkan_executor_cpu_test(32u<<20);
        strata::worker::GpuCounters counters;
        auto reader=[&](const strata::worker::GpuAsset& asset,std::vector<uint8_t>& blob,std::string& e){return cpu.load_blob(asset.layer,asset.expert,blob,e);};
        check(executor->prepare_source(p,reader,counters,error),error);
        check(counters.gpu_weight_bytes==2*5222400 && counters.gpu_allocated_bytes<=32u<<20,"actual tensor accounting");
        executor->release(counters);check(!counters.gpu_allocated_bytes,"direct executor release");
        strata::worker::InferRequest request;
        request.model_id=p.model_id;request.partition_id=p.placement_id.a;request.session=1;request.sequence=1;
        request.layer=0;request.T=1;request.K=3;request.mask={1,0,1};request.ids={3,8,1};
        request.weights={.2f,.3f,.5f};request.activations=route.activation;
        check(executor->prepare_source(p,reader,counters,error),error);
        strata::worker::EntryRowsReply first,second;
        check(executor->infer_rows(p,request,first,error) && executor->infer_rows(p,request,second,error),error);
        check(first.indices==std::vector<uint16_t>({0,2}) && first.rows==second.rows,"same-backend indexed row repeat");
        request.activations.pop_back();check(!executor->infer_rows(p,request,second,error) && second.rows.empty(),"executor input shape refusal");
        executor->release(counters);
        auto corrupt=[&](const strata::worker::GpuAsset& asset,std::vector<uint8_t>& blob,std::string& e){if(!reader(asset,blob,e))return false;if(asset.expert==3)blob[0]^=1;return true;};
        check(!executor->prepare_source(p,corrupt,counters,error)&&!counters.gpu_allocated_bytes,"partial prepare digest failure frees tensors");
        std::puts("portable_local_experts_test passed: mixed rows, two-die ownership, caps, shapes, failures, source identity, direct executor load/release");
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
