// Actual network-worker CUDA executor. Uses Strata's existing grouped expert kernels;
// RemoteExperts is a same-process helper and is deliberately not used as a network service.
#include "strata/worker/gpu_worker.hpp"
#include "gpu_cpu_reference.hpp"
#include <cstdio>
#include <cstdlib>
#include "strata/kernels/iq_kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace strata::worker {
namespace {
constexpr int cap = kMaxT*kMaxK;
bool check(cudaError_t s, const char* what, std::string& err) {
    if(s==cudaSuccess) return true;
    err=std::string("CUDA ")+what+": "+cudaGetErrorString(s); return false;
}
size_t align256(size_t n) { return (n+255)&~size_t(255); }
class CudaExecutor final : public GpuExecutor {
public:
    explicit CudaExecutor(int device): device_(device) {}
    ~CudaExecutor() override { GpuCounters ignored; release(ignored); }
    bool prepare(const GpuPlan& p,uint64_t reserve,GpuCache& cache,GpuCounters& c,std::string& err) override {
        release(c);
        if (!validate_gpu_plan(p,err) || p.gu_type!=8 || p.d_type!=8 ||
            !strata::kernels::native_expert_supported(p.gu_type,p.d_type,p.n_embd,p.n_ff)) {
            err="CUDA worker supports plain Q8_0/Q8_0 only at supported native dimensions"; return false;
        }
        const char* reference=std::getenv("STRATA_CUDA_Q8_CPU_REFERENCE");
        cpu_reference_=reference && reference[0]=='1';
        if(cpu_reference_ && (p.n_embd!=2560 || p.n_ff!=640 || !q8_cpu_reference_format(cpu_fmt_,err))) {
            err="CPU reference requires qualified AVX2/non-VNNI Q8_0/Q8_0 2560x640";return false;
        }
        if(cpu_reference_)std::fprintf(stderr,"strata CUDA diagnostic: Q8 CPU AVX2 dot order + host SwiGLU/quantization; slow reference, not production qualification\n");
        if(!check(cudaSetDevice(device_),"select device",err)) return false;
        layout_=strata::kernels::native_expert_layout(p.gu_type,p.d_type,p.n_embd,p.n_ff);
        for(const auto& asset:p.assets) if(asset.bytes!=layout_.bytes) {err="CUDA blob/layout length mismatch";return false;}
        const size_t xbytes=(size_t)kMaxT*p.n_embd*sizeof(float);
        const size_t qbytes=(size_t)kMaxT*(p.n_embd/32)*36;
        const size_t obytes=(size_t)cap*p.n_embd*sizeof(float);
        const size_t scratch=strata::kernels::native_expert_scratch_bytes(cap,p.n_ff);
        // All temporaries used by grouped execution are accounted before expert allocation.
        const size_t work=align256(xbytes)+align256(qbytes)+align256(obytes)+align256(scratch)+
            align256(cap*sizeof(unsigned long long))+align256((cap+1)*sizeof(int32_t))+
            align256(cap*sizeof(int32_t))*2+align256(sizeof(int32_t));
        const uint64_t weights=p.assigned_bytes();
        size_t free_before=0,total=0;
        if(!check(cudaMemGetInfo(&free_before,&total),"available VRAM",err))return false;
        c.gpu_free_before=free_before; c.gpu_runtime_reserve=reserve;
        if(weights>SIZE_MAX || weights>free_before || reserve>free_before-weights || work>free_before-weights-reserve) {
            err="assigned experts plus exact grouped scratch/workspace and runtime reserve exceed available VRAM";return false;
        }
        if(!check(cudaStreamCreateWithFlags(&stream_,cudaStreamNonBlocking),"create stream",err) ||
           !check(cudaMalloc(&weights_,(size_t)weights),"allocate assigned experts",err)) {release(c);return false;}
        c.gpu_allocated_bytes=weights; c.gpu_weight_bytes=weights;
        ++c.gpu_allocation_count; c.gpu_peak_bytes=std::max(c.gpu_peak_bytes,c.gpu_allocated_bytes);
        if(!check(cudaMalloc(&workspace_,work),"allocate bounded workspace",err)) {release(c);return false;}
        c.gpu_allocated_bytes+=work; c.gpu_scratch_bytes=work;
        ++c.gpu_allocation_count; c.gpu_peak_bytes=std::max(c.gpu_peak_bytes,c.gpu_allocated_bytes);
        auto* next=(uint8_t*)workspace_;
        auto take=[&](size_t bytes)->void* {void* result=next;next+=align256(bytes);return result;};
        dx_=(float*)take(xbytes); dq_=take(qbytes); dout_=(float*)take(obytes); scratch_=take(scratch);
        dptr_=(unsigned long long*)take(cap*sizeof(unsigned long long));
        dstart_=(int32_t*)take((cap+1)*sizeof(int32_t)); ddst_=(int32_t*)take(cap*sizeof(int32_t));
        dtok_=(int32_t*)take(cap*sizeof(int32_t)); dcount_=(int32_t*)take(sizeof(int32_t));
        uint64_t offset=0; blobs_.clear(); blobs_.reserve(p.assets.size());
        std::vector<uint8_t> staging;
        for(const auto& asset:p.assets) {
            if(!cache.read(asset,staging,err) || staging.size()!=asset.bytes ||
               !check(cudaMemcpyAsync((uint8_t*)weights_+offset,staging.data(),staging.size(),cudaMemcpyHostToDevice,stream_),"load expert",err) ||
               !check(cudaStreamSynchronize(stream_),"finish expert load",err)) { release(c);return false; }
            c.host_staging_peak=std::max(c.host_staging_peak,(uint64_t)staging.capacity());
            blobs_.push_back((uint8_t*)weights_+offset); offset+=asset.bytes;
        }
        // cudaMemGetInfo observes allocator granularity/driver effects, beyond requested payload bytes.
        size_t free_after=0;
        if(!check(cudaMemGetInfo(&free_after,&total),"post-allocation available VRAM",err) || free_after<reserve) {
            if(err.empty())err="CUDA allocation left less than explicit runtime reserve";release(c);return false;
        }
        observed_bytes_=free_before>=free_after?free_before-free_after:0;
        c.gpu_free_after=free_after;
        c.gpu_peak_bytes=std::max(c.gpu_peak_bytes,(uint64_t)observed_bytes_);
        loaded_=true;return true;
    }
    bool supports_rows() const override { return true; }
    bool infer_rows(const GpuPlan& p,const InferRequest& r,EntryRowsReply& rows,std::string& err) override {
        InferReply ignored; return infer_impl(p,r,ignored,&rows,err);
    }
    bool infer_rows_diagnostic(const GpuPlan& p,const InferRequest& r,EntryRowsReply& rows,GpuExecutionTrace& trace,std::string& err) override {
        trace={};InferReply ignored;return infer_impl(p,r,ignored,&rows,err,&trace);
    }
    bool infer(const GpuPlan& p,const InferRequest& r,InferReply& reply,std::string& err) override {
        return infer_impl(p,r,reply,nullptr,err);
    }
    bool infer_impl(const GpuPlan& p,const InferRequest& r,InferReply& reply,EntryRowsReply* entry_rows,std::string& err,GpuExecutionTrace* trace=nullptr) {
        if(!loaded_ || !check(cudaSetDevice(device_),"select inference device",err)){err="CUDA executor unavailable: "+err;return false;}
        if(cpu_reference_ && trace){err="CPU reference does not expose grouped-kernel metadata";return false;}
        std::vector<int32_t> expert,group(r.mask.size(),-1),start,dst,tok,original;
        std::vector<unsigned long long> ptr;
        for(size_t i=0;i<r.mask.size();++i) if(r.mask[i]) {
            auto it=std::find(expert.begin(),expert.end(),r.ids[i]);
            if(it==expert.end()) {
                const auto* asset=p.find(r.layer,r.ids[i]);
                if(!asset){err="CUDA request contains unassigned expert";return false;}
                const size_t index=(size_t)(asset-p.assets.data());
                if(index>=blobs_.size()){err="CUDA resident slot missing";return false;}
                expert.push_back(r.ids[i]);ptr.push_back((unsigned long long)blobs_[index]);
                if(trace)trace->asset_slots.push_back((int32_t)index);
                group[i]=(int32_t)expert.size()-1;
            } else group[i]=(int32_t)(it-expert.begin());
        }
        for(size_t g=0;g<expert.size();++g) {
            start.push_back((int32_t)dst.size());
            for(size_t i=0;i<group.size();++i)if(group[i]==(int32_t)g){
                original.push_back((int32_t)i);dst.push_back((int32_t)dst.size());tok.push_back((int32_t)(i/r.K));
            }
        }
        start.push_back((int32_t)dst.size());const int32_t count=(int32_t)expert.size();
        if(!count || dst.size()>cap){err="CUDA group bounds";return false;}
        if(trace){trace->experts=expert;trace->start=start;trace->dst=dst;trace->tok=tok;trace->original=original;}
        auto copy=[&](void* d,const void* h,size_t bytes){return check(cudaMemcpyAsync(d,h,bytes,cudaMemcpyHostToDevice,stream_),"request copy",err);};
        if(!copy(dx_,r.activations.data(),r.activations.size()*sizeof(float)) ||
           !copy(dptr_,ptr.data(),ptr.size()*sizeof(ptr[0])) || !copy(dstart_,start.data(),start.size()*sizeof(start[0])) ||
           !copy(ddst_,dst.data(),dst.size()*sizeof(dst[0])) || !copy(dtok_,tok.data(),tok.size()*sizeof(tok[0])) ||
           !copy(dcount_,&count,sizeof count))return false;
        if(cpu_reference_) {
            // Use the actual raw inputs and original packed-entry mapping. Expert weights
            // remain on the GPU; only gate/up and native packed hidden rows cross the host.
            std::vector<unsigned long long> entry_ptr(original.size());
            for(size_t j=0;j<original.size();++j)entry_ptr[j]=ptr[(size_t)group[(size_t)original[j]]];
            std::vector<uint8_t> packed;
            q8_cpu_reference_input(cpu_fmt_,r.activations.data(),r.T,packed);
            if(!copy(dptr_,entry_ptr.data(),entry_ptr.size()*8) || !copy(dq_,packed.data(),packed.size()))return false;
            const size_t fa=align256((size_t)cap*p.n_ff*4);
            auto* gate=(float*)scratch_;auto* up=(float*)((uint8_t*)scratch_+fa);
            auto* hidden=(uint8_t*)scratch_+3*fa;
            const int entries=(int)original.size();
            if(!q8_avx2_reference_dot(dptr_,dtok_,(const uint8_t*)dq_,2560,640,entries,0,gate,stream_) ||
               !q8_avx2_reference_dot(dptr_,dtok_,(const uint8_t*)dq_,2560,640,entries,layout_.up_off,up,stream_) ||
               !check(cudaGetLastError(),"CPU reference gate/up launch",err))return false;
            std::vector<float> gates((size_t)entries*640),ups(gates.size());
            if(!check(cudaMemcpyAsync(gates.data(),gate,gates.size()*4,cudaMemcpyDeviceToHost,stream_),"CPU reference gate read",err) ||
               !check(cudaMemcpyAsync(ups.data(),up,ups.size()*4,cudaMemcpyDeviceToHost,stream_),"CPU reference up read",err) ||
               !check(cudaStreamSynchronize(stream_),"CPU reference gate/up finish",err))return false;
            q8_cpu_reference_hidden(cpu_fmt_,gates.data(),ups.data(),entries,packed);
            std::vector<int32_t> entry_row(original.size());for(size_t j=0;j<entry_row.size();++j)entry_row[j]=(int32_t)j;
            if(!copy(hidden,packed.data(),packed.size()) || !copy(dtok_,entry_row.data(),entry_row.size()*4))return false;
            if(!q8_avx2_reference_dot(dptr_,dtok_,hidden,640,2560,entries,layout_.down_off,dout_,stream_) ||
               !check(cudaGetLastError(),"CPU reference down launch",err))return false;
        } else {
        strata::kernels::quantize_q8_1_rows(dx_,r.T,p.n_embd,dq_,stream_);
        strata::kernels::native_expert_grouped(layout_,dptr_,dstart_,dcount_,ddst_,dtok_,count,
            (int64_t)dst.size(),dq_,scratch_,dout_,stream_);
        }
        std::vector<float> rows(dst.size()*p.n_embd);
        if(!check(cudaGetLastError(),"grouped expert launch",err) ||
           !check(cudaMemcpyAsync(rows.data(),dout_,rows.size()*sizeof(float),cudaMemcpyDeviceToHost,stream_),"result copy",err) ||
           !check(cudaStreamSynchronize(stream_),"grouped expert completion",err))return false;
        // Restore original entry association and apply routing weights once in entry order.
        std::vector<int32_t> packed(r.mask.size(),-1);
        for(size_t j=0;j<original.size();++j)packed[(size_t)original[j]]=(int32_t)j;
        if(entry_rows) {
            entry_rows->session=r.session;entry_rows->sequence=r.sequence;
            entry_rows->indices.clear();entry_rows->rows.clear();
            for(size_t i=0;i<r.mask.size();++i)if(r.mask[i]) {
                if(packed[i]<0){err="CUDA missing packed result";return false;}
                const float* row=rows.data()+(size_t)packed[i]*p.n_embd;
                for(size_t h=0;h<p.n_embd;++h)if(!std::isfinite(row[h])){err="nonfinite CUDA expert result";return false;}
                entry_rows->indices.push_back((uint16_t)i);
                entry_rows->rows.insert(entry_rows->rows.end(),row,row+p.n_embd);
            }
            return true;
        }
        reply.partials.assign((size_t)r.T*p.n_embd,0.f);
        for(size_t i=0;i<r.mask.size();++i)if(r.mask[i]){
            if(packed[i]<0){err="CUDA missing packed result";return false;}
            const float* row=rows.data()+(size_t)packed[i]*p.n_embd;
            float* out=reply.partials.data()+(i/r.K)*p.n_embd;
            for(size_t h=0;h<p.n_embd;++h) {
                if(!std::isfinite(row[h])){err="nonfinite CUDA expert result";return false;}
                out[h]=out[h]+r.weights[i]*row[h];
            }
        }
        return true;
    }
    void release(GpuCounters& c) override {
        if(weights_ || workspace_ || stream_)cudaSetDevice(device_);
        if(stream_)cudaStreamSynchronize(stream_);
        if(workspace_)cudaFree(workspace_);if(weights_)cudaFree(weights_);if(stream_)cudaStreamDestroy(stream_);
        workspace_=weights_=nullptr;stream_=nullptr;blobs_.clear();loaded_=false;observed_bytes_=0;
        c.gpu_allocated_bytes=0;c.gpu_weight_bytes=0;c.gpu_scratch_bytes=0;
    }
private:
    int device_;bool loaded_=false,cpu_reference_=false;size_t observed_bytes_=0;
    strata::kernels::cpu::NativeFmt cpu_fmt_;
    cudaStream_t stream_=nullptr;void *weights_=nullptr,*workspace_=nullptr,*dq_=nullptr,*scratch_=nullptr;
    float *dx_=nullptr,*dout_=nullptr;unsigned long long* dptr_=nullptr;
    int32_t *dstart_=nullptr,*ddst_=nullptr,*dtok_=nullptr,*dcount_=nullptr;
    strata::kernels::NativeExpertLayout layout_;
    std::vector<uint8_t*> blobs_;
};
}
std::unique_ptr<GpuExecutor> create_cuda_gpu_executor(int device) { return std::make_unique<CudaExecutor>(device); }
} // namespace strata::worker
