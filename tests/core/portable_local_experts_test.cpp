#include "../../src/core/portable_local_experts.hpp"
#include "strata/worker/vulkan_executor.hpp"
#include "../../src/worker/vulkan_expert_graph.hpp"
#include "portable_model_fixture.hpp"
#include <cstdio>
#include <limits>
using namespace strata::core;
using namespace model_fixture;
namespace {
struct State { int prepares=0,releases=0,inferences=0; size_t last_T=0; bool fail=false,bad=false,nonfinite=false; float bad_value=NAN; size_t bad_offset=0; };
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
        ++s.inferences;s.last_T=r.T;
        check(r.ids.size()==size_t(r.T)*r.K&&r.mask.size()==r.ids.size()&&
              r.activations.size()==size_t(r.T)*2560,"flattened executor request");
        if(s.fail){error="controlled GPU failure";return false;}
        out={};out.session=r.session;out.sequence=r.sequence;
        for(size_t i=0;i<r.mask.size();++i)if(r.mask[i]) {
            out.indices.push_back(uint16_t(i));
            out.rows.insert(out.rows.end(),2560,float(r.ids[i]+1));
        }
        if(s.bad)out.indices.back()=out.indices.front();
        if(s.nonfinite)out.rows.at(s.bad_offset)=s.bad_value;
        return true;
    }
    void release(strata::worker::GpuCounters&)override {if(live){++s.releases;live=false;}}
};
}
int main(){
    try {
        const std::vector<float> finite_values{0.f,-0.f,std::numeric_limits<float>::min(),
                                               std::numeric_limits<float>::denorm_min(),std::numeric_limits<float>::max()};
        strata::worker::vulkan_detail::finite({});
        strata::worker::vulkan_detail::finite(finite_values);
        for(float invalid:{NAN,INFINITY,-INFINITY})
            for(size_t at:{size_t(0),finite_values.size()/2,finite_values.size()-1}) {
                auto values=finite_values;values[at]=invalid;bool refused=false;
                try{strata::worker::vulkan_detail::finite(values);}
                catch(const std::runtime_error& e){refused=true;check(std::string(e.what())=="nonfinite expert value","exact executor finite diagnostic");}
                check(refused,"executor finite validator checks every position");
            }
        SparseModel f;strata::GgufModel model(f.paths);PortableModelOptions opts;
        portable_detail::CpuExperts cpu(model,(f.root/"manifest").string(),opts);
        auto plan=[&](const std::string& name,std::vector<std::pair<uint16_t,uint16_t>> ids){
            strata::worker::GpuPlan p;std::string e;
            check(strata::worker::make_gpu_plan(cpu.identity(),ids,p,e),e);
            const auto path=(f.root/name).string();check(strata::worker::write_gpu_plan(path,p,e),e);return path;
        };
        opts.accelerators={reinterpret_cast<ggml_backend_t>(uintptr_t(1)),reinterpret_cast<ggml_backend_t>(uintptr_t(2))};
        PortableExpertDevice a,b;a.plan_path=plan("a",{{0,1},{0,3}});b.plan_path=plan("b",{{0,5}});b.accelerator=1;
        opts.expert_timings=true;
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
            for(const auto& d:local.report()) check(d.infer_rows_calls==2 && std::isfinite(d.infer_rows_ms) && d.infer_rows_ms>0,
                  "one finite executor-call capture per participating device");
            const auto good_rows=rows;
            auto bad=route;bad.ids.pop_back();check(!local.run(0,bad,rows,error)&&rows.empty(),"shape refusal");
            const auto before_inferences=state.inferences;
            for(float invalid:{NAN,INFINITY,-INFINITY}) {
                for(size_t at:{size_t(0),route.activation.size()/2,route.activation.size()-1}) {
                    auto invalid_route=route;invalid_route.activation[at]=invalid;rows={123.f};
                    check(!local.run(0,invalid_route,rows,error)&&rows.empty()&&
                          error=="portable model: nonfinite resident activation","resident activation diagnostic");
                }
                for(size_t at:{size_t(0),route.weights.size()/2,route.weights.size()-1}) {
                    auto invalid_route=route;invalid_route.weights[at]=invalid;rows={123.f};
                    check(!local.run(0,invalid_route,rows,error)&&rows.empty()&&
                          error=="portable model: nonfinite resident routing weights","resident routing diagnostic");
                }
            }
            for(int invalid_id:{-1,512,route.ids[1]}) {
                auto invalid_route=route;invalid_route.ids[0]=invalid_id;
                check(!local.run(0,invalid_route,rows,error)&&rows.empty()&&error=="invalid resident expert IDs",
                      "resident ID diagnostic");
            }
            check(state.inferences==before_inferences,"invalid route stops before GPU dispatch");
            for(size_t T=1;T<=8;++T) {
                std::vector<PortableRoute> routes(T,route);
                for(size_t t=0;t<T;++t) std::rotate(routes[t].ids.begin(),routes[t].ids.begin()+t,routes[t].ids.end());
                const auto before=state.inferences;
                const auto gpu_before=local.gpu_entries(), cpu_before_batch=local.cpu_entries();
                check(local.run_batch(0,routes,rows,error),error);
                check(state.inferences==before+2&&state.last_T==T,"one T-column request per participating die");
                check(local.gpu_entries()==gpu_before+3*T&&local.cpu_entries()==cpu_before_batch+7*T,
                      "batch counts actual flattened hot/cold entries");
                for(size_t t=0;t<T;++t) {
                    std::vector<float> cold_rows;check(cpu.run(0,routes[t],cold_rows,error),error);
                    for(size_t k=0;k<10;++k) for(size_t h=0;h<2560;++h) {
                        const int id=routes[t].ids[k];
                        const float expected=(id==1||id==3||id==5)?float(id+1):cold_rows[k*2560+h];
                        check(rows[(t*10+k)*2560+h]==expected,"batch original token/slot scatter");
                    }
                }
            }
            std::vector<PortableRoute> bad_batch(8,route);bad_batch.back().ids.back()=512;
            const auto calls_before_bad_batch=state.inferences;
            check(!local.run_batch(0,bad_batch,rows,error)&&rows.empty()&&
                  state.inferences==calls_before_bad_batch,"validate all batch routes before GPU");
            state.bad=true;
            check(!local.run_batch(0,std::vector<PortableRoute>(8,route),rows,error)&&rows.empty(),
                  "batch duplicate flattened reply slot rejects publication");state.bad=false;
            state.fail=true;
            check(!local.run_batch(0,std::vector<PortableRoute>(8,route),rows,error)&&rows.empty(),
                  "batch device failure cannot fall back or publish");state.fail=false;
            std::puts("Local expert T1-8 flattened requests and hot/cold original-slot scatter passed; GPU graphs remain per-entry");
            state.bad=true;check(!local.run(0,route,rows,error)&&rows.empty()&&
                error=="resident reply duplicate/unowned slot","duplicate reply slot diagnostic");state.bad=false;
            const auto cpu_before=local.cpu_entries();
            state.nonfinite=true;
            for(float invalid:{NAN,INFINITY,-INFINITY})
                for(size_t at:{size_t(0),size_t(2560),size_t(5119)}) {
                    state.bad_value=invalid;state.bad_offset=at;rows={123.f};
                    check(!local.run(0,route,rows,error)&&rows.empty()&&
                          error=="portable model: nonfinite resident expert output","nonfinite reply diagnostic");
                    check(local.cpu_entries()==cpu_before,"nonfinite reply cannot dispatch CPU fallback");
                }
            state.nonfinite=false;
            check(local.run(0,route,rows,error)&&rows==good_rows,"valid provider replay after finite failures");
            const auto calls_before_failure=local.report()[0].infer_rows_calls;
            state.fail=true;check(!local.run(0,route,rows,error)&&rows.empty(),"failure no CPU replay");state.fail=false;
            check(local.report()[0].infer_rows_calls==calls_before_failure+1,"failed executor attempt captured once");
            free=1;check(!local.run(0,route,rows,error)&&rows.empty(),"runtime free guard");free=7ull<<30;
            check(local.run(0,route,rows,error),"provider reused after caller resets sequence state");
            const auto old=std::filesystem::last_write_time(f.paths[0]);
            std::filesystem::last_write_time(f.paths[0],old+std::chrono::seconds(2));
            check(!local.run(0,route,rows,error)&&rows.empty(),"resident source identity stop");
            std::filesystem::last_write_time(f.paths[0],old);
        }
        check(state.releases==2,"both executors released");
        opts.expert_timings=false;
        {
            portable_detail::LocalExperts untimed(model,cpu,opts,factory,memory);
            check(untimed.run(0,route,rows,error),error);
            for(const auto& d:untimed.report()) check(d.infer_rows_calls==0 && d.infer_rows_ms==0,
                "default executor path has no new timing capture");
        }
        opts.expert_timings=true;
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
        strata::worker::GpuPlan p;check(strata::worker::make_gpu_plan(cpu.identity(),{{0,0},{0,1},{0,3}},p,error),error);
        auto executor=strata::worker::create_vulkan_executor_cpu_test(32u<<20);
        strata::worker::GpuCounters counters;
        auto reader=[&](const strata::worker::GpuAsset& asset,std::vector<uint8_t>& blob,std::string& e){return cpu.load_blob(asset.layer,asset.expert,blob,e);};
        check(executor->prepare_source(p,reader,counters,error),error);
        check(counters.gpu_weight_bytes==3*5222400 && counters.gpu_allocated_bytes<=32u<<20,"actual tensor accounting");
        {
        strata::worker::InferRequest request;
        request.model_id=p.model_id;request.partition_id=p.placement_id.a;
        request.session=1;request.sequence=1;request.layer=0;request.T=8;request.K=10;
        for(size_t t=0;t<8;++t) {
            auto token=route;std::rotate(token.ids.begin(),token.ids.begin()+t,token.ids.end());
            request.ids.insert(request.ids.end(),token.ids.begin(),token.ids.end());
            request.weights.insert(request.weights.end(),token.weights.begin(),token.weights.end());
            for(auto& value:token.activation)value+=float(t)*0.02f;
            request.activations.insert(request.activations.end(),token.activation.begin(),token.activation.end());
            for(auto id:token.ids)request.mask.push_back(p.owns(0,id));
        }
        strata::worker::EntryRowsReply reply;
        check(executor->infer_rows(p,request,reply,error),error);
        check(reply.indices.size()==24&&reply.rows.size()==24*2560,"real executor T8 indexed rows");
        for(size_t t=0;t<8;++t) {
            auto token=route;
            token.ids.assign(request.ids.begin()+t*10,request.ids.begin()+(t+1)*10);
            token.activation.assign(request.activations.begin()+t*2560,request.activations.begin()+(t+1)*2560);
            std::vector<float> expected;check(cpu.run(0,token,expected,error),error);
            for(size_t n=0;n<reply.indices.size();++n) {
                const size_t slot=reply.indices[n];if(slot/10!=t)continue;
                for(size_t h=0;h<2560;++h)
                    check(std::abs(reply.rows[n*2560+h]-expected[(slot%10)*2560+h])<=
                          0.003f*(1+std::abs(expected[(slot%10)*2560+h])),"real executor T8 rows versus native CPU");
            }
        }
        }
        executor->release(counters);check(!counters.gpu_allocated_bytes,"direct executor release");
        strata::worker::InferRequest request;
        request.model_id=p.model_id;request.partition_id=p.placement_id.a;request.session=1;request.sequence=1;
        request.layer=0;request.T=1;request.K=3;request.mask={1,0,1};request.ids={3,8,1};
        request.weights={.2f,.3f,.5f};request.activations=route.activation;
        check(executor->prepare_source(p,reader,counters,error),error);
        strata::worker::EntryRowsReply first,second;
        check(executor->infer_rows(p,request,first,error) && executor->infer_rows(p,request,second,error),error);
        check(first.indices==std::vector<uint16_t>({0,2}) && first.rows==second.rows,"same-backend indexed row repeat");
        for(float invalid:{NAN,INFINITY,-INFINITY}) {
            for(size_t at:{size_t(0),request.activations.size()/2,request.activations.size()-1}) {
                auto bad_request=request;bad_request.activations[at]=invalid;
                check(!executor->infer_rows(p,bad_request,second,error)&&second.rows.empty()&&
                      error=="nonfinite expert value","executor activation failure diagnostic");
            }
            for(size_t at:{size_t(0),size_t(1),size_t(2)}) {
                auto bad_request=request;bad_request.weights[at]=invalid;
                check(!executor->infer_rows(p,bad_request,second,error)&&second.rows.empty()&&
                      error=="invalid Vulkan routing field","all routing slots validated including masked slot");
            }
        }
        for(size_t at:{size_t(0),size_t(1),size_t(2)}) {
            auto bad_request=request;bad_request.mask[at]=2;
            check(!executor->infer_rows(p,bad_request,second,error)&&second.rows.empty()&&
                  error=="invalid Vulkan routing field","mask diagnostic");
        }
        for(size_t at:{size_t(0),size_t(2)}) {
            auto bad_request=request;bad_request.ids[at]=8;
            check(!executor->infer_rows(p,bad_request,second,error)&&second.rows.empty()&&
                  error=="Vulkan adapter request contains unassigned expert","unowned expert diagnostic");
        }
        check(executor->infer_rows(p,request,second,error)&&second.rows==first.rows,
              "valid executor replay after finite/routing failures");
        request.activations.pop_back();check(!executor->infer_rows(p,request,second,error) && second.rows.empty(),"executor input shape refusal");
        executor->release(counters);
        auto corrupt=[&](const strata::worker::GpuAsset& asset,std::vector<uint8_t>& blob,std::string& e){if(!reader(asset,blob,e))return false;if(asset.expert==3)blob[0]^=1;return true;};
        check(!executor->prepare_source(p,corrupt,counters,error)&&!counters.gpu_allocated_bytes,"partial prepare digest failure frees tensors");
        std::puts("portable_local_experts_test passed: mixed rows, two-die ownership, caps, shapes, failures, source identity, direct executor load/release");
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
