// Replay one real captured GPU layer request on the existing CPU-native path.
#include "strata/worker/gpu_capture.hpp"
#include "strata/worker/expert_worker.hpp"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <sys/resource.h>

using namespace strata::worker;
namespace {
void require(bool ok,const std::string& why) {if (!ok) throw std::runtime_error(why);}
std::vector<float> values(const std::vector<uint8_t>& bytes,size_t count) {
    require(bytes.size()==count*4,"float record size"); std::vector<float> out(count);
    std::memcpy(out.data(),bytes.data(),bytes.size());
    for (float x:out) require(std::isfinite(x),"nonfinite capture"); return out;
}
void compare(const char* label,const std::vector<float>& cpu,const std::vector<float>& gpu) {
    require(cpu.size()==gpu.size(),"comparison shape"); double error=0,denom=0,square=0,maximum=0;
    for (size_t i=0;i<cpu.size();++i) {require(std::isfinite(cpu[i]) && std::isfinite(gpu[i]),"nonfinite comparison");const double d=(double)gpu[i]-cpu[i];error+=std::fabs(d);denom+=std::fabs(cpu[i]);square+=d*d;maximum=std::max(maximum,std::fabs(d));}
    std::printf("COMPARE %s n=%zu rel_L1=%.12g maxabs=%.12g rmse=%.12g bitwise=%d\n",label,cpu.size(),error/std::max(denom,1e-12),maximum,std::sqrt(square/cpu.size()),std::memcmp(cpu.data(),gpu.data(),cpu.size()*4)==0);
}
std::vector<float> q81(const std::vector<uint8_t>& raw,size_t count) {
    require(count%32==0 && raw.size()==count/32*36,"q8_1 snapshot size"); std::vector<float> out(count);
    for (size_t block=0;block<count/32;++block) {
        ggml_fp16_t scale; std::memcpy(&scale,raw.data()+block*36,2);
        const float d=ggml_fp16_to_fp32(scale); require(std::isfinite(d),"q8_1 finite scale");
        for (size_t i=0;i<32;++i) out[block*32+i]=d*(int8_t)raw[block*36+4+i];
    }
    return out; // Format decoding only, not an alternative expert implementation.
}
std::vector<float> native_quant(const void* raw,int type,size_t count) {
    const auto* traits=ggml_get_type_traits((ggml_type)type);
    require(traits && traits->to_float,"CPU quantized format decoder"); std::vector<float> out(count);
    traits->to_float(raw,out.data(),(int64_t)count); return out;
}
}
int main(int argc,char** argv) {
    const rlimit memory{512ull<<20,512ull<<20},cpu{60,60},output{1ull<<20,1ull<<20};
    if (::setrlimit(RLIMIT_AS,&memory) || ::setrlimit(RLIMIT_CPU,&cpu) || ::setrlimit(RLIMIT_FSIZE,&output)) return 2;
    ::alarm(90);
    try {
        if (argc!=5) throw std::runtime_error("usage: strata-gpu-capture-replay CAPTURE_DIR PLAN LOCAL_ROOT LOCAL_MANIFEST");
        std::string error; GpuPlan plan; require(read_gpu_plan(argv[2],plan,error),error);
        std::vector<uint8_t> encoded; require(diagnostic::file(argv[1],"cuda.capture",encoded,false,error),error);
        diagnostic::Capture capture; InferRequest request;
        require(diagnostic::decode(plan,encoded,capture,request,error),error);
        diagnostic::Merge merge; encoded.clear(); require(diagnostic::file(argv[1],"merge.capture",encoded,false,error),error);
        require(diagnostic::decode(encoded,merge,error) && merge.request==capture.request,"same GPU/engine request identity");
        const size_t H=plan.n_embd,K=request.K; const auto gpu_rows=values(capture.rows,K*H),gpu_partial=values(capture.partial,H);
        require(merge.gpu==capture.partial && merge.cpu_mask.size()==K,"same reply and CPU mask geometry");
        for (size_t i=0;i<K;++i) require(merge.cpu_mask[i]<=1 && !(merge.cpu_mask[i] && request.mask[i]),"disjoint CPU/GPU ownership");
        LocalManifest manifest; require(read_local_manifest(argv[4],manifest,error),error);
        GpuAssetSource source; require(source.open(argv[3],manifest,plan,error),error);
        strata::kernels::cpu::NativeFmt fmt;
        require(strata::kernels::cpu::native_fmt(plan.gu_type,plan.d_type,H,plan.n_ff,fmt,error),error);
        std::vector<float> cpu_rows(K*H,0.f); std::vector<uint8_t> blob; size_t total=0;
        for (size_t entry=0;entry<K;++entry) if (request.mask[entry]) {
            const auto* asset=plan.find(request.layer,request.ids[entry]); require(asset,"assigned expert");
            require(source.read(*asset,blob,error),error); total+=blob.size(); require(total<=K*kGpuBlobMax,"bounded replay reads");
            const uint8_t* pointer=blob.data(); const uint8_t mask=1; const float weight=1.f;
            std::vector<float> row; require(ExpertWorkerService::direct_native_partial(fmt,1,1,&mask,&weight,request.activations.data(),&pointer,row,error),error);
            std::copy(row.begin(),row.end(),cpu_rows.begin()+entry*H);
            char label[100]; std::snprintf(label,sizeof label,"expert layer=%u entry=%zu expert=%u",request.layer,entry,request.ids[entry]);
            compare(label,row,std::vector<float>(gpu_rows.begin()+entry*H,gpu_rows.begin()+(entry+1)*H));
            if (entry==capture.selected) {
                require(blob==capture.blob,"captured resident GPU bytes equal CPU source bytes");
                std::vector<uint8_t> input(fmt.act_bytes),hidden(fmt.h_bytes); std::vector<float> ff(plan.n_ff),down(H);
                strata::kernels::cpu::native_quant_act(fmt,request.activations.data(),input.data());
                const void* a=input.data(); float* f=ff.data();
                strata::kernels::cpu::native_gu_rows(fmt,blob.data(),&a,1,&f,0,plan.n_ff);
                strata::kernels::cpu::native_quant_h(fmt,ff.data(),hidden.data());
                const void* h=hidden.data(); float* d=down.data();
                strata::kernels::cpu::native_down_rows(fmt,blob.data(),&h,1,&d,0,(int)H);
                require(std::memcmp(row.data(),down.data(),H*4)==0,"native phase calls match direct native path");
                compare("input_quant_decoded",native_quant(input.data(),fmt.gu_act,H),q81(capture.input_q,H));
                compare("hidden_quant_decoded",native_quant(hidden.data(),fmt.d_act,plan.n_ff),q81(capture.hidden_q,plan.n_ff));
                compare("CPU_SwiGLU_vs_CUDA_quantized_hidden",ff,q81(capture.hidden_q,plan.n_ff));
                // Fused CUDA SwiGLU does not retain separate gate/up or float h.
                std::puts("TRACE gate_up=not_retained CUDA_float_SwiGLU=not_retained hidden_quant=actual down=actual");
            }
        }
        std::vector<float> weighted(H),from_gpu_rows(H);
        ExpertWorkerService::accumulate_partial(1,(int)K,request.mask.data(),request.weights.data(),cpu_rows.data(),(int)H,weighted.data());
        ExpertWorkerService::accumulate_partial(1,(int)K,request.mask.data(),request.weights.data(),gpu_rows.data(),(int)H,from_gpu_rows.data());
        compare("CPU_native_GPU_subset_vs_GPU_partial",weighted,gpu_partial);
        compare("GPU_rows_weighted_once_vs_GPU_partial",from_gpu_rows,gpu_partial);
        const auto before=values(merge.before,H),cpu=values(merge.cpu,H),after=values(merge.after,H);
        std::vector<float> sum(H);
        for (size_t i=0;i<H;++i) sum[i]=(before[i]+gpu_partial[i])+cpu[i];
        compare("engine_host_merge",sum,after);
        std::printf("GPU_CAPTURE_REPLAY_COMPLETE_V1 layer=%u selected_entry=%u source_read_bytes=%zu kernel_changes=0 thresholds_changed=0\n",request.layer,capture.selected,total);
        return 0; // Diagnostic deltas are reported, never threshold-tuned into PASS.
    } catch (const std::exception& error) {std::fprintf(stderr,"REPLAY_FAIL %s\n",error.what());return 1;}
}
