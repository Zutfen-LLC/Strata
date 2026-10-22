// First-token comparison and same-input replay through existing native CPU kernels.
#include "strata/worker/layer_trace.hpp"
#include "strata/worker/layer_grouping.hpp"
#include "strata/worker/expert_worker.hpp"
#include <sys/resource.h>
using namespace strata::worker;
namespace L=strata::worker::diagnostic::layer;
namespace G=strata::worker::diagnostic::grouping;
namespace {
void require(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
L::Record read(const char* dir,uint32_t kind,uint32_t layer){L::Record r;std::string error;require(L::read(dir,kind,layer,r,error),error);return r;}
void shape(const L::Record& a,const L::Record& b){require(a.layers==b.layers&&a.H==b.H&&a.K==b.K&&a.experts==b.experts,"trace geometry differs");}
L::Difference compare(const char* label,uint32_t layer,const std::vector<float>& a,const std::vector<float>& b){
    auto d=L::difference(a,b);
    std::printf("LAYER_COMPARE layer=%u field=%s bitwise=%d material=%d maxabs=%.12g relL1=%.12g rmse=%.12g drift_limit=%.12g\n",
        layer,label,d.bitwise,d.material,d.maximum,d.relative,d.rmse,d.limit);return d;
}
struct Boundary {
    L::Record host,device;
    std::vector<float> x,w,parts,shared,before,after,helper,cpu,gpu,sum;
    std::vector<int32_t> ids,kind,mask;
};
Boundary boundary(const char* dir,uint32_t l,const L::Record& meta){
    Boundary b;b.host=read(dir,L::Host,l);b.device=read(dir,L::Device,l);shape(meta,b.host);shape(meta,b.device);
    require(b.host.fields.size()==8&&b.device.fields.size()==5,"boundary field count");
    const auto& h=b.host.fields;const auto& d=b.device.fields;const size_t H=meta.H,K=meta.K;
    b.x=L::finite(h[0],H);b.ids=L::values<int32_t>(h[1],K);b.w=L::finite(h[2],K);b.kind=L::values<int32_t>(h[3],K);
    b.helper=L::finite(h[4],H);b.cpu=L::finite(h[5],H);b.gpu=L::finite(h[6],H);b.sum=L::finite(h[7],H);
    b.parts=L::finite(d[0],K*H);b.shared=L::finite(d[1],H);b.before=L::finite(d[2],H);b.after=L::finite(d[3],H);b.mask=L::values<int32_t>(d[4],K);
    for(size_t i=0;i<K;++i){require(b.ids[i]>=0&&b.ids[i]<(int)meta.experts&&b.kind[i]>=-1&&b.kind[i]<=4,"route/owner bounds");
        require(b.mask[i]==0||b.mask[i]==1,"device ownership mask bounds");}
    return b;
}
void merge_checks(const char* mode,const Boundary& b){
    const size_t H=b.host.H,K=b.host.K;std::vector<float> remote(H),final(H);
    for(size_t h=0;h<H;++h){remote[h]=G::add(G::add(b.helper[h],b.gpu[h]),b.cpu[h]);final[h]=G::add(b.before[h],b.sum[h]);}
    char label[100];
    for(auto model:{G::Local::NativeFma,G::Local::Float64}){
        std::snprintf(label,sizeof label,"SELF_%s_local_%s",mode,G::name(model));
        compare(label,b.host.layer,G::local(b.parts,b.w,b.shared,model),b.before);
    }
    std::snprintf(label,sizeof label,"SELF_%s_host_remote_sum",mode);compare(label,b.host.layer,remote,b.sum);
    std::snprintf(label,sizeof label,"SELF_%s_final_device_add",mode);compare(label,b.host.layer,final,b.after);
    for(size_t i=0;i<K;++i){const bool remote_owner=b.kind[i]==2||b.kind[i]==3||b.kind[i]==4;
        std::printf("OWNER mode=%s layer=%u entry=%zu expert=%d kind=%d device_mask=%d expected_remote=%d\n",mode,b.host.layer,i,b.ids[i],b.kind[i],b.mask[i],remote_owner);}
}
void inspect(const Boundary& b,const Boundary& g){
    for(const auto& field:std::vector<std::pair<const char*,std::pair<const std::vector<float>*,const std::vector<float>*>>>{
            {"shared",{&b.shared,&g.shared}},{"local_rows",{&b.parts,&g.parts}},{"pre_remote",{&b.before,&g.before}},
            {"helper_partial",{&b.helper,&g.helper}},{"CPU_partial",{&b.cpu,&g.cpu}},{"GPU_partial",{&b.gpu,&g.gpu}},
            {"remote_sum",{&b.sum,&g.sum}}}){
        std::string label="CROSS_B_G_"+std::string(field.first);compare(label.c_str(),b.host.layer,*field.second.first,*field.second.second);
    }
    compare("CROSS_B_G_routing_weights",b.host.layer,b.w,g.w);
    for(size_t i=0;i<b.ids.size();++i)
        std::printf("ROUTE_DETAIL layer=%u entry=%zu B_id=%d G_id=%d B_weight=%a G_weight=%a B_owner=%d G_owner=%d B_mask=%d G_mask=%d\n",
            b.host.layer,i,b.ids[i],g.ids[i],b.w[i],g.w[i],b.kind[i],g.kind[i],b.mask[i],g.mask[i]);
}
std::vector<float> actual_gpu_rows(const Boundary& b,const L::Record& gpu,const GpuPlan& frozen){
    shape(b.host,gpu);InferRequest request;
    require(gpu.fields.size()==3&&gpu_codec::decode_request(frozen,gpu.fields[0],request),"actual GPU request binding");
    require(request.T==1&&request.layer==b.host.layer&&request.K==b.host.K,"GPU request geometry association");
    require(G::exact(request.activations,b.x)&&G::exact(request.weights,b.w),"GPU request exact input/weight bytes");
    for(size_t i=0;i<b.ids.size();++i)require(request.ids[i]==b.ids[i]&&request.mask[i]==(b.kind[i]==4),"GPU request routing/ownership association");
    require(gpu.fields[2]==b.host.fields[6],"GPU/engine partial exact bytes");
    return L::finite(gpu.fields[1],(size_t)b.host.K*b.host.H);
}
std::vector<float> native_rows(const Boundary& b,const LocalManifest& manifest,const char* root){
    const size_t H=b.host.H,K=b.host.K;const uint32_t layer=b.host.layer;std::string error;
    std::vector<std::pair<uint16_t,uint16_t>> selected;
    for(int32_t expert:b.ids)selected.emplace_back(layer,expert);
    std::sort(selected.begin(),selected.end());selected.erase(std::unique(selected.begin(),selected.end()),selected.end());
    GpuPlan plan;require(make_gpu_plan(manifest,selected,plan,error),error);GpuAssetSource source;require(source.open(root,manifest,plan,error),error);
    strata::kernels::cpu::NativeFmt fmt;require(strata::kernels::cpu::native_fmt(plan.gu_type,plan.d_type,H,plan.n_ff,fmt,error),error);
    std::vector<float> rows(K*H);
    std::vector<uint8_t> blob;
    for(size_t i=0;i<K;++i){const auto* asset=plan.find(layer,b.ids[i]);require(asset&&source.read(*asset,blob,error),error);
        const uint8_t* pointer=blob.data();const uint8_t mask=1;const float weight=1.f;std::vector<float> row;
        require(ExpertWorkerService::direct_native_partial(fmt,1,1,&mask,&weight,b.x.data(),&pointer,row,error),error);
        std::copy(row.begin(),row.end(),rows.begin()+i*H);
    }
    std::printf("COMMON_INPUT_NATIVE_ROWS layer=%u source_bytes=%llu experts=%zu input=B_bitwise_G kernel_changes=0\n",layer,(unsigned long long)source.stats().source_read_bytes,K);
    return rows;
}
void group_report(const char* experiment,const Boundary& b,const Boundary& g,const G::Groups& pb,const G::Groups& pg){
    char label[120];
    for(const auto& mode:std::vector<std::pair<const char*,std::pair<const Boundary*,const G::Groups*>>>{{"B",{&b,&pb}},{"G",{&g,&pg}}}){
        const auto& actual=*mode.second.first;const auto& predicted=*mode.second.second;
        for(const auto& field:std::vector<std::pair<const char*,std::pair<const std::vector<float>*,const std::vector<float>*>>>{
                {"local",{&predicted.local,&actual.before}},{"CPU",{&predicted.cpu,&actual.cpu}},
                {"GPU",{&predicted.gpu,&actual.gpu}},{"remote",{&predicted.remote,&actual.sum}},{"final",{&predicted.final,&actual.after}}}){
            std::snprintf(label,sizeof label,"%s_%s_prediction_vs_actual_%s",experiment,mode.first,field.first);
            compare(label,b.host.layer,*field.second.first,*field.second.second);
        }
    }
    const auto delta=G::delta(b.after,g.after,pb.final,pg.final);
    std::printf("GROUPING_DELTA experiment=%s layer=%u exact_coordinates=%zu/%zu actual_max=%.12g predicted_max=%.12g residual_max=%.12g helper_per_expert_rows_unobserved=1 causality_not_implied=1\n",
        experiment,b.host.layer,delta.equal,b.after.size(),delta.actual_max,delta.predicted_max,delta.residual_max);
}
void common_replay(const char* row_source,const Boundary& b,const Boundary& g,const std::vector<float>& rows){
    require(G::exact(b.x,g.x)&&b.ids==g.ids,"common-input replay requires exact input and identical routed IDs");
    bool helper_owners_equal=true;
    for(size_t i=0;i<b.kind.size();++i)helper_owners_equal&=(b.kind[i]==2)==(g.kind[i]==2);
    const bool controlled=G::exact(b.w,g.w)&&G::exact(b.shared,g.shared)&&G::exact(b.helper,g.helper)&&helper_owners_equal;
    std::printf("COMMON_GROUPING_CONTROL rows=%s layer=%u input_bitwise=1 ids_equal=1 weights_bitwise=%d shared_bitwise=%d helper_partial_bitwise=%d helper_owners_equal=%d fixed_controls_match_observed=%d\n",
        row_source,b.host.layer,G::exact(b.w,g.w),G::exact(b.shared,g.shared),G::exact(b.helper,g.helper),helper_owners_equal,controlled);
    for(auto model:{G::Local::NativeFma,G::Local::Float64}){
        const auto pb=G::reduce(rows,b.w,b.kind,b.shared,model,&b.helper),pg=G::reduce(rows,g.w,g.kind,g.shared,model,&g.helper);
        std::string label=std::string(row_source)+"_actual_controls_"+G::name(model);
        group_report(label.c_str(),b,g,pb,pg);
        if(helper_owners_equal){
            // Same rows AND weights/shared/helper: isolate only grouping order.
            const auto fixed=G::reduce(rows,b.w,g.kind,b.shared,model,&b.helper);
            label=std::string(row_source)+"_fixed_B_controls_"+G::name(model);group_report(label.c_str(),b,g,pb,fixed);
        }else std::puts("FIXED_HELPER_GROUPING_UNAVAILABLE helper_membership_differs");
    }
}
}
int main(int argc,char** argv){
    const rlimit memory{512ull<<20,512ull<<20},cpu{90,90},output{1ull<<20,1ull<<20};
    if(setrlimit(RLIMIT_AS,&memory)||setrlimit(RLIMIT_CPU,&cpu)||setrlimit(RLIMIT_FSIZE,&output))return 2;alarm(120);
    try{
        require(argc==7,"usage: strata-layer-diagnose B_DIR G_DIR GPU_DIR FROZEN_PLAN LOCAL_ROOT LOCAL_MANIFEST");
        const auto bm=read(argv[1],L::Window,0),gm=read(argv[2],L::Window,0);shape(bm,gm);
        require(bm.fields==gm.fields&&bm.fields.size()==2&&L::values<int32_t>(bm.fields[0],1)[0]==9419&&L::values<int64_t>(bm.fields[1],1)[0]==0,"same fresh first-token window required");
        const auto bp=read(argv[1],L::Partition,0),gp=read(argv[2],L::Partition,0);shape(bm,bp);shape(bm,gp);
        require(bp.fields.size()==1&&bp.fields==gp.fields&&bp.fields[0].size()==(size_t)bm.layers*((bm.experts+7)/8),"exact B/G CPU partition differs");
        size_t count=0;for(uint8_t byte:bp.fields[0])for(int bit=0;bit<8;++bit)count+=(byte>>bit)&1;
        require(count==21369,"frozen CPU partition count differs");
        std::string error;GpuPlan frozen;require(read_gpu_plan(argv[4],frozen,error),error);
        require(frozen.n_layers==bm.layers&&frozen.n_embd==bm.H&&frozen.n_expert==bm.experts,"frozen plan geometry");
        int first_bits=-1,first_material=-1,first_input_bits=-1,first_input_material=-1;
        std::vector<G::Observation> observations;
        for(uint32_t l=0;l<bm.layers;++l){auto b=boundary(argv[1],l,bm),g=boundary(argv[2],l,bm);
            const auto input=compare("B_vs_G_input",l,b.x,g.x),out=compare("B_vs_G_final_device",l,b.after,g.after);
            if(!input.bitwise&&first_input_bits<0)first_input_bits=l;if(input.material&&first_input_material<0)first_input_material=l;
            if(!out.bitwise&&first_bits<0)first_bits=l;if(out.material&&first_material<0)first_material=l;
            observations.push_back({(int)l,input.bitwise,out.bitwise});
            std::printf("ROUTING layer=%u ids_equal=%d weights_bitwise=%d\n",l,b.ids==g.ids,b.host.fields[2]==g.host.fields[2]);
            merge_checks("B",b);merge_checks("G",g);
            if(l>=1&&l<=4)inspect(b,g);
            for(size_t i=0;i<g.ids.size();++i)require((g.kind[i]==4)==frozen.owns(l,g.ids[i]),"G frozen GPU owner differs");
        }
        std::printf("FIRST_DIVERGENCE input_bits=%d input_drift=%d output_bits=%d output_drift=%d diagnostic_drift=1e-5 formal_thresholds_unchanged=1\n",first_input_bits,first_input_material,first_bits,first_material);
        const int selected=G::seed(observations);
        std::printf("SELECTION policy=earliest_BIT_output_difference_with_bitwise_identical_input layer=%d source_layer_indices_zero_based=1 requested_detail_layers=1..4\n",selected);
        if(selected>=0){auto b=boundary(argv[1],selected,bm),g=boundary(argv[2],selected,bm);LocalManifest manifest;require(read_local_manifest(argv[6],manifest,error),error);
            require(selected>=1&&selected<=4,"seed is outside requested saved layers 1..4; no speculative replay");
            require(b.ids==g.ids,"seed routed IDs differ; common expert-row association unavailable");
            bool has_gpu=std::find(g.kind.begin(),g.kind.end(),4)!=g.kind.end();std::vector<float> actual;
            if(has_gpu)actual=actual_gpu_rows(g,read(argv[3],L::Gpu,selected),frozen);
            auto common=native_rows(b,manifest,argv[5]);const size_t H=b.host.H;
            for(size_t i=0;i<b.kind.size();++i){
                const std::vector<float> row(common.begin()+i*H,common.begin()+(i+1)*H);
                char label[90];
                if(b.kind[i]<=1){std::snprintf(label,sizeof label,"COMMON_native_vs_actual_B_local_entry_%zu",i);
                    compare(label,selected,row,std::vector<float>(b.parts.begin()+i*H,b.parts.begin()+(i+1)*H));}
                if(g.kind[i]<=1){std::snprintf(label,sizeof label,"COMMON_native_vs_actual_G_local_entry_%zu",i);
                    compare(label,selected,row,std::vector<float>(g.parts.begin()+i*H,g.parts.begin()+(i+1)*H));}
            }
            common_replay("native",b,g,common);
            if(has_gpu){
                for(size_t i=0;i<g.kind.size();++i)if(g.kind[i]==4){
                    char label[90];std::snprintf(label,sizeof label,"COMMON_native_vs_actual_G_GPU_entry_%zu",i);
                    compare(label,selected,std::vector<float>(common.begin()+i*H,common.begin()+(i+1)*H),std::vector<float>(actual.begin()+i*H,actual.begin()+(i+1)*H));
                    std::copy_n(actual.data()+i*H,H,common.data()+i*H);
                }
                std::puts("HYBRID_ROWS actual_G_GPU_rows_used_for_BOTH_groupings other_rows_native helper_partial_anchored=1");
                common_replay("hybrid_G_GPU",b,g,common);
            }
        }else std::puts("COMMON_REPLAY_UNAVAILABLE no_BIT_different_output_with_identical_input no_common_input_invented");
        std::puts("LAYER_GROUPING_DIAGNOSTIC_COMPLETE_V1 numerical_acceptance_not_implied=1 new_GEN=0");return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"LAYER_DIAGNOSTIC_FAIL %s\n",e.what());return 1;}
}
