// First-token comparison and same-input replay through existing native CPU kernels.
#include "strata/worker/layer_trace.hpp"
#include "strata/worker/expert_worker.hpp"
#include <sys/resource.h>
using namespace strata::worker;
namespace L=strata::worker::diagnostic::layer;
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
    const size_t H=b.host.H,K=b.host.K;std::vector<float> local=b.shared,remote(H),final(H);
    for(size_t h=0;h<H;++h){for(size_t i=0;i<K;++i)local[h]=local[h]+b.w[i]*b.parts[i*H+h];
        remote[h]=(b.helper[h]+b.gpu[h])+b.cpu[h];final[h]=b.before[h]+b.sum[h];}
    char label[80];std::snprintf(label,sizeof label,"%s_local_device_reduction",mode);compare(label,b.host.layer,local,b.before);
    std::snprintf(label,sizeof label,"%s_host_remote_sum",mode);compare(label,b.host.layer,remote,b.sum);
    std::snprintf(label,sizeof label,"%s_final_device_add",mode);compare(label,b.host.layer,final,b.after);
    for(size_t i=0;i<K;++i){const bool remote_owner=b.kind[i]==2||b.kind[i]==3||b.kind[i]==4;
        std::printf("OWNER mode=%s layer=%u entry=%zu expert=%d kind=%d device_mask=%d expected_remote=%d\n",mode,b.host.layer,i,b.ids[i],b.kind[i],b.mask[i],remote_owner);}
}
void replay(const char* mode,const Boundary& b,const L::Record* gpu,const GpuPlan& frozen,const LocalManifest& manifest,const char* root){
    const size_t H=b.host.H,K=b.host.K;const uint32_t layer=b.host.layer;std::string error;
    std::vector<std::pair<uint16_t,uint16_t>> selected;
    for(int32_t expert:b.ids)selected.emplace_back(layer,expert);
    std::sort(selected.begin(),selected.end());selected.erase(std::unique(selected.begin(),selected.end()),selected.end());
    GpuPlan plan;require(make_gpu_plan(manifest,selected,plan,error),error);GpuAssetSource source;require(source.open(root,manifest,plan,error),error);
    strata::kernels::cpu::NativeFmt fmt;require(strata::kernels::cpu::native_fmt(plan.gu_type,plan.d_type,H,plan.n_ff,fmt,error),error);
    std::vector<float> actual_gpu;InferRequest request;
    if(gpu){shape(b.host,*gpu);require(gpu->fields.size()==3&&gpu_codec::decode_request(frozen,gpu->fields[0],request),"actual GPU request binding");
        require(request.T==1&&request.layer==layer&&request.K==K&&request.activations==b.x,"GPU request input association");
        require(std::memcmp(request.activations.data(),b.x.data(),H*4)==0&&std::memcmp(request.weights.data(),b.w.data(),K*4)==0,"GPU request exact float bytes");
        for(size_t i=0;i<K;++i)require(request.ids[i]==b.ids[i]&&request.mask[i]==(b.kind[i]==4),"GPU request routing/ownership association");
        actual_gpu=L::finite(gpu->fields[1],K*H);require(gpu->fields[2]==b.host.fields[6],"GPU/engine partial exact bytes");}
    std::vector<float> all=b.shared,remote_cpu(H,0.f),remote_gpu(H,0.f),helper(H,0.f),rows(K*H);
    std::vector<uint8_t> blob;
    for(size_t i=0;i<K;++i){const auto* asset=plan.find(layer,b.ids[i]);require(asset&&source.read(*asset,blob,error),error);
        const uint8_t* pointer=blob.data();const uint8_t mask=1;const float weight=1.f;std::vector<float> row;
        require(ExpertWorkerService::direct_native_partial(fmt,1,1,&mask,&weight,b.x.data(),&pointer,row,error),error);
        std::copy(row.begin(),row.end(),rows.begin()+i*H);
        char label[100];
        if(b.kind[i]==4){require(gpu,"GPU row evidence missing at selected boundary");std::snprintf(label,sizeof label,"%s_native_vs_actual_GPU_entry_%zu",mode,i);
            compare(label,layer,row,std::vector<float>(actual_gpu.begin()+i*H,actual_gpu.begin()+(i+1)*H));}
        else if(b.kind[i]==0||b.kind[i]==1||b.kind[i]==-1){std::snprintf(label,sizeof label,"%s_native_vs_actual_device_entry_%zu",mode,i);
            compare(label,layer,row,std::vector<float>(b.parts.begin()+i*H,b.parts.begin()+(i+1)*H));}
        for(size_t h=0;h<H;++h){all[h]=all[h]+b.w[i]*row[h];
            if(b.kind[i]==3)remote_cpu[h]=remote_cpu[h]+b.w[i]*row[h];
            if(b.kind[i]==4)remote_gpu[h]=remote_gpu[h]+b.w[i]*row[h];
            if(b.kind[i]==2)helper[h]=helper[h]+b.w[i]*row[h];}
    }
    char label[80];std::snprintf(label,sizeof label,"%s_native_CPU_subset",mode);compare(label,layer,remote_cpu,b.cpu);
    std::snprintf(label,sizeof label,"%s_native_GPU_subset",mode);compare(label,layer,remote_gpu,b.gpu);
    std::snprintf(label,sizeof label,"%s_native_helper_subset",mode);compare(label,layer,helper,b.helper);
    std::snprintf(label,sizeof label,"%s_all_native_plus_actual_shared",mode);compare(label,layer,all,b.after);
    std::printf("LAYER_REPLAY mode=%s layer=%u source_bytes=%llu actual_input=1 kernel_changes=0\n",mode,layer,(unsigned long long)source.stats().source_read_bytes);
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
        for(uint32_t l=0;l<bm.layers;++l){auto b=boundary(argv[1],l,bm),g=boundary(argv[2],l,bm);
            const auto input=compare("B_vs_G_input",l,b.x,g.x),out=compare("B_vs_G_final_device",l,b.after,g.after);
            if(!input.bitwise&&first_input_bits<0)first_input_bits=l;if(input.material&&first_input_material<0)first_input_material=l;
            if(!out.bitwise&&first_bits<0)first_bits=l;if(out.material&&first_material<0)first_material=l;
            std::printf("ROUTING layer=%u ids_equal=%d weights_bitwise=%d\n",l,b.ids==g.ids,b.host.fields[2]==g.host.fields[2]);
            merge_checks("B",b);merge_checks("G",g);
            for(size_t i=0;i<g.ids.size();++i)require((g.kind[i]==4)==frozen.owns(l,g.ids[i]),"G frozen GPU owner differs");
        }
        std::printf("FIRST_DIVERGENCE input_bits=%d input_drift=%d output_bits=%d output_drift=%d diagnostic_drift=1e-5 formal_thresholds_unchanged=1\n",first_input_bits,first_input_material,first_bits,first_material);
        const auto earliest=[](int a,int b){return a<0?b:b<0?a:std::min(a,b);};
        const int material=earliest(first_material,first_input_material);
        const int selected=material>=0?material:earliest(first_bits,first_input_bits);
        if(selected>=0){auto b=boundary(argv[1],selected,bm),g=boundary(argv[2],selected,bm);LocalManifest manifest;require(read_local_manifest(argv[6],manifest,error),error);
            bool has_gpu=std::find(g.kind.begin(),g.kind.end(),4)!=g.kind.end();L::Record rows;
            if(has_gpu)rows=read(argv[3],L::Gpu,selected);
            replay("B_exact_input",b,nullptr,frozen,manifest,argv[5]);replay("G_exact_input",g,has_gpu?&rows:nullptr,frozen,manifest,argv[5]);
        }else std::puts("REPLAY_NOT_NEEDED observed_MoE_boundaries_equal head_residual_path_not_proven");
        std::puts("FIRST_LAYER_DIAGNOSTIC_COMPLETE_V1 numerical_acceptance_not_implied=1 new_GEN=0");return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"LAYER_DIAGNOSTIC_FAIL %s\n",e.what());return 1;}
}
