// First-token comparison and same-input replay through existing native CPU kernels.
#include "strata/worker/layer_trace.hpp"
#include "strata/worker/layer_grouping.hpp"
#include "strata/worker/expert_worker.hpp"
#include <sys/resource.h>
#include <array>
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
std::vector<float> cpu_rows(const Boundary& b,const LocalManifest& manifest,const char* root){
    const size_t H=b.host.H,K=b.host.K;const uint32_t layer=b.host.layer;std::string error;
    std::vector<std::pair<uint16_t,uint16_t>> selected;
    for(size_t i=0;i<K;++i)if(b.kind[i]==3)selected.emplace_back(layer,b.ids[i]);
    std::sort(selected.begin(),selected.end());selected.erase(std::unique(selected.begin(),selected.end()),selected.end());
    GpuPlan plan;require(make_gpu_plan(manifest,selected,plan,error),error);GpuAssetSource source;require(source.open(root,manifest,plan,error),error);
    strata::kernels::cpu::NativeFmt fmt;require(strata::kernels::cpu::native_fmt(plan.gu_type,plan.d_type,H,plan.n_ff,fmt,error),error);
    std::vector<float> rows(K*H);
    std::vector<uint8_t> blob;
    size_t probes=0;
    for(size_t i=0;i<K;++i){
        if(b.kind[i]!=3)continue; // Captured local/helper anchors need no surrogate expert computation.
        const auto* asset=plan.find(layer,b.ids[i]);require(asset&&source.read(*asset,blob,error),error);
        const uint8_t* pointer=blob.data();const uint8_t mask=1;const float weight=1.f;std::vector<float> row;
        require(ExpertWorkerService::direct_native_partial(fmt,1,1,&mask,&weight,b.x.data(),&pointer,row,error),error);
        std::copy(row.begin(),row.end(),rows.begin()+i*H);
        ++probes;
    }
    std::printf("FOUR_WAY_CPU_ROWS layer=%u source_bytes=%llu experts=%zu input=B_bitwise_G local_helper_probes=0 kernel_changes=0\n",layer,(unsigned long long)source.stats().source_read_bytes,probes);
    return rows;
}
size_t four_way_controls(const Boundary& b,const Boundary& g){
    require(G::exact(b.x,g.x)&&b.ids==g.ids&&G::exact(b.w,g.w),"four-way input/IDs/order/weights must be identical");
    require(G::exact(b.shared,g.shared)&&G::exact(b.parts,g.parts)&&G::exact(b.before,g.before)&&
            G::exact(b.helper,g.helper)&&b.mask==g.mask,"four-way captured local/helper/mask controls differ");
    return G::moved_cpu_to_gpu(b.kind,g.kind);
}
void effect_report(const char* field,const char* effect,const std::vector<double>& values){
    double maximum=0,square=0;size_t nonzero=0;
    for(double value:values){maximum=std::max(maximum,std::fabs(value));square+=value*value;nonzero+=value!=0;}
    std::printf("FOUR_WAY_EFFECT field=%s effect=%s nonzero=%zu/%zu maxabs=%.12g rms=%.12g counterfactual=1\n",
        field,effect,nonzero,values.size(),maximum,std::sqrt(square/values.size()));
}
void four_way(const Boundary& b,const Boundary& g,const std::vector<float>& native,const std::vector<float>& cuda,size_t moved){
    const size_t H=b.host.H;
    std::printf("FOUR_WAY_CONTROLS layer=%u entry=%zu expert=%d weight=%a input_ids_weights_bitwise=1 captured_local_helper_bitwise=1 local_helper_recomputed=0 row_native=unit_weight_worker_probe\n",
        b.host.layer,moved,b.ids[moved],b.w[moved]);
    compare("FOUR_WAY_moved_CPU_vs_captured_CUDA_row",b.host.layer,
        std::vector<float>(native.begin()+moved*H,native.begin()+(moved+1)*H),
        std::vector<float>(cuda.begin()+moved*H,cuda.begin()+(moved+1)*H));
    auto substituted=native;std::copy_n(cuda.data()+moved*H,H,substituted.data()+moved*H);
    const std::array<G::Groups,4> arms{
        G::anchored(native,b.w,b.kind,b.before,b.helper),G::anchored(native,b.w,g.kind,b.before,b.helper),
        G::anchored(substituted,b.w,b.kind,b.before,b.helper),G::anchored(substituted,b.w,g.kind,b.before,b.helper)};
    const char* names[]={"CPU_baseline","CPU_distributed","CUDA_baseline","CUDA_distributed"};
    struct Field {const char* name;const std::vector<float> *observed_b,*observed_g;std::vector<float> G::Groups::* member;};
    const Field fields[]={{"CPU_partial",&b.cpu,&g.cpu,&G::Groups::cpu},{"GPU_partial",&b.gpu,&g.gpu,&G::Groups::gpu},
                          {"remote",&b.sum,&g.sum,&G::Groups::remote},{"final",&b.after,&g.after,&G::Groups::final}};
    for(const auto& field:fields){
        for(size_t arm=0;arm<arms.size();++arm){
            const auto& predicted=arms[arm].*(field.member);char label[120];
            std::snprintf(label,sizeof label,"FOUR_WAY_%s_vs_observed_B_%s",names[arm],field.name);compare(label,b.host.layer,predicted,*field.observed_b);
            std::snprintf(label,sizeof label,"FOUR_WAY_%s_vs_observed_G_%s",names[arm],field.name);compare(label,b.host.layer,predicted,*field.observed_g);
            const auto delta=G::delta(*field.observed_b,*field.observed_g,arms[0].*(field.member),predicted);
            std::printf("FOUR_WAY_DELTA cell=%s field=%s reference_cell=CPU_baseline exact_coordinates=%zu/%zu actual_max=%.12g predicted_max=%.12g residual_max=%.12g\n",
                names[arm],field.name,delta.equal,H,delta.actual_max,delta.predicted_max,delta.residual_max);
        }
        const auto effects=G::effects(arms[0].*(field.member),arms[1].*(field.member),arms[2].*(field.member),arms[3].*(field.member));
        effect_report(field.name,"row_at_baseline",effects.row_baseline);effect_report(field.name,"row_at_distributed",effects.row_distributed);
        effect_report(field.name,"group_at_CPU",effects.group_cpu);effect_report(field.name,"group_at_CUDA",effects.group_cuda);
        effect_report(field.name,"row_main_average",effects.row_main);effect_report(field.name,"group_main_average",effects.group_main);
        effect_report(field.name,"interaction_difference_in_differences",effects.interaction);
    }
    std::puts("FOUR_WAY_LIMITS all_cells_are_predictions CPU_baseline_targets_observed_B CUDA_distributed_targets_observed_G helper_per_expert_rows_not_required_for_fixed_anchor no_ULP_universality no_end_to_end_causal_claim");
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
            for(size_t i=0;i<g.ids.size();++i)require((g.kind[i]==4)==frozen.owns(l,g.ids[i]),"G frozen GPU owner differs");
        }
        std::printf("FIRST_DIVERGENCE input_bits=%d input_drift=%d output_bits=%d output_drift=%d diagnostic_drift=1e-5 formal_thresholds_unchanged=1\n",first_input_bits,first_input_material,first_bits,first_material);
        const int selected=G::seed(observations);
        std::printf("SELECTION policy=earliest_BIT_output_difference_with_bitwise_identical_input layer=%d source_layer_indices_zero_based=1 requested_detail=seed_only\n",selected);
        if(selected>=0){auto b=boundary(argv[1],selected,bm),g=boundary(argv[2],selected,bm);LocalManifest manifest;require(read_local_manifest(argv[6],manifest,error),error);
            require(selected>=1&&selected<=4,"seed is outside requested saved layers 1..4; no speculative replay");
            const size_t moved=four_way_controls(b,g);inspect(b,g);
            const auto actual=actual_gpu_rows(g,read(argv[3],L::Gpu,selected),frozen);
            four_way(b,g,cpu_rows(b,manifest,argv[5]),actual,moved);
        }else std::puts("COMMON_REPLAY_UNAVAILABLE no_BIT_different_output_with_identical_input no_common_input_invented");
        std::puts("LAYER_FOUR_WAY_DIAGNOSTIC_COMPLETE_V1 numerical_acceptance_not_implied=1 new_GEN=0");return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"LAYER_DIAGNOSTIC_FAIL %s\n",e.what());return 1;}
}
