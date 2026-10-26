#include "local_fixture.hpp"
#include "gpu_cpu_reference.hpp"
#include "ggml-cpu.h"
#include <cmath>
#ifdef STRATA_CPU_REFERENCE_CUDA
#include "strata/worker/gpu_worker.hpp"
#include <cuda_runtime.h>
#endif
namespace {
// Independent host transcription checks the proposed GPU lane grouping against
// the linked ggml AVX2 dot, rather than merely testing the activation wrapper.
float avx2_dot_reference(const uint8_t* weights,const uint8_t* input,int n) {
    float acc[8]{};
    for(int b=0;b<n/32;++b) {
        ggml_fp16_t dw,dx;std::memcpy(&dw,weights+b*34,2);std::memcpy(&dx,input+b*34,2);
        const float scale=ggml_fp16_to_fp32(dw)*ggml_fp16_to_fp32(dx);
        const auto* w=(const int8_t*)(weights+b*34+2);const auto* x=(const int8_t*)(input+b*34+2);
        for(int lane=0;lane<8;++lane) {
            int pair[2]{};
            for(int j=0;j<4;++j) {
                const int k=4*lane+j,ax=w[k]<0?-int(w[k]):int(w[k]);
                const int sx=w[k]<0?(x[k]==-128?-128:-int(x[k])):(w[k]==0?0:int(x[k]));
                pair[j/2]+=ax*sx;
            }
            const int dot=std::min(32767,std::max(-32768,pair[0]))+std::min(32767,std::max(-32768,pair[1]));
            acc[lane]=std::fma(scale,(float)dot,acc[lane]);
        }
    }
    return ((acc[4]+acc[0])+(acc[6]+acc[2]))+((acc[5]+acc[1])+(acc[7]+acc[3]));
}
}
int main() {
    try {
        ggml_cpu_init();if(!ggml_cpu_has_avx2() || !ggml_cpu_has_fma())return 77;
#ifdef STRATA_CPU_REFERENCE_CUDA
        int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices)return 77;
        const char* opt=std::getenv("STRATA_CUDA_Q8_CPU_REFERENCE");
        require(opt && opt[0]=='1',"CUDA reference test requires explicit diagnostic option");
#endif
        Fixture fixture;std::string err;strata::kernels::cpu::NativeFmt fmt;
        require(q8_cpu_reference_format(fmt,err),err);
        std::vector<float> x(2*2560),gate(2*640),up(gate.size()),hidden(gate.size()),cpu_out(2*2560);
        for(size_t i=0;i<x.size();++i)x[i]=float(int(i%37)-18)*.03125f;
        std::vector<uint8_t> input,packed,expected(2*680);
        q8_cpu_reference_input(fmt,x.data(),2,input);
        const auto dot=ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->vec_dot;
        const auto& blob=fixture.blobs[1];
        for(int t=0;t<2;++t) {
            const void* a=input.data()+t*2720;float* h=hidden.data()+t*640;
            strata::kernels::cpu::native_gu_rows(fmt,blob.data(),&a,1,&h,0,640);
            for(int j=0;j<640;++j) {
                dot(2560,&gate[t*640+j],0,blob.data()+j*fmt.gu_row,0,a,0,1);
                dot(2560,&up[t*640+j],0,blob.data()+fmt.up_off+j*fmt.gu_row,0,a,0,1);
                const float g=avx2_dot_reference(blob.data()+j*fmt.gu_row,(const uint8_t*)a,2560);
                const float u=avx2_dot_reference(blob.data()+fmt.up_off+j*fmt.gu_row,(const uint8_t*)a,2560);
                require(!std::memcmp(&g,&gate[t*640+j],4) && !std::memcmp(&u,&up[t*640+j],4),"AVX2 gate/up reference arithmetic differs from linked ggml");
            }
            strata::kernels::cpu::native_quant_h(fmt,h,expected.data()+t*680);
            const void* hq=expected.data()+t*680;float* out=cpu_out.data()+t*2560;
            strata::kernels::cpu::native_down_rows(fmt,blob.data(),&hq,1,&out,0,2560);
            for(int j=0;j<2560;++j) {
                const float v=avx2_dot_reference(blob.data()+fmt.down_off+j*fmt.d_row,(const uint8_t*)hq,640);
                require(!std::memcmp(&v,&out[j],4),"AVX2 down reference arithmetic differs from linked ggml");
            }
        }
        q8_cpu_reference_hidden(fmt,gate.data(),up.data(),2,packed);
        require(packed==expected,"host SwiGLU/quantization differs from native_gu_rows + native_quant_h");
        bool refused=false;try{q8_cpu_reference_input(fmt,x.data(),0,packed);}catch(...){refused=true;}
        require(refused,"invalid reference token count not refused");
        refused=false;try{q8_cpu_reference_hidden(fmt,gate.data(),up.data(),81,packed);}catch(...){refused=true;}
        require(refused,"invalid reference entry count not refused");
#ifdef STRATA_CPU_REFERENCE_CUDA
        GpuPlan plan;require(make_gpu_plan(fixture.m,{{0,1}},plan,err),err);
        GpuCache cache;std::vector<uint8_t> missing;const auto root=fixture.root/"reference-cache";
        require(fs::create_directory(root)&&cache.open(root.string(),16ull<<20,err)&&cache.prepare(plan,missing,err)&&
                cache.publish(plan.assets[0],blob.data(),blob.size(),err),err);
        auto executor=create_cuda_gpu_executor(0);GpuCounters counters;
        require(executor->prepare(plan,256ull<<20,cache,counters,err),err);
        InferRequest q;q.session=q.sequence=1;q.model_id=plan.model_id;q.partition_id=plan.placement_id.a;
        q.layer=0;q.T=2;q.K=3;q.activations=x;q.ids.assign(6,1);q.weights.assign(6,1.f);q.mask={1,0,1,0,1,1};
        EntryRowsReply rows;require(executor->infer_rows(plan,q,rows,err),err);
        require(rows.indices==std::vector<uint16_t>({0,2,4,5}),"reference original entry mapping");
        for(size_t j=0;j<rows.indices.size();++j) {
            const auto* expected_row=cpu_out.data()+(rows.indices[j]/3)*2560;
            require(!std::memcmp(expected_row,rows.rows.data()+j*2560,2560*4),"CUDA AVX2-reference expert row differs from native CPU");
        }
        executor->release(counters);cache.close();
        std::puts("Q8_CPU_REFERENCE CUDA_rows=4 repeated_expert_T2=1 exact_native_CPU=1 actual_CUDA=1");
#else
        std::puts("Q8_CPU_REFERENCE host_dot_order_vs_ggml_bits=1 host_hidden_quant_bits=1 shape_refusals=1 actual_CUDA=0");
#endif
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"Q8_CPU_REFERENCE_REFUSED %s\n",e.what());return 1;}
}
