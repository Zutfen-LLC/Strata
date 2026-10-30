#pragma once
#include "vulkan_native_q8_layout.hpp"
#include "../core/strata_vk_native_stages.hpp"
#include "../core/strata_vk_q8_gu.hpp"
#include "../core/strata_vk_q8_down.hpp"
#include <fstream>
#include <cstdio>
namespace strata::worker::vulkan_detail {
// Explicit unqualified observation path. Frozen packed slots and one reusable
// tensor workspace; no host intermediate reconstruction or alternate placement.
class NativeQ8Cache {
    static constexpr size_t blob_bytes=5222400, down_offset=3481600;
    Context ctx_{nullptr,ggml_free};Buffer buffer_{nullptr,ggml_backend_buffer_free};
    ggml_backend_t backend_;
    ggml_tensor *w_,*i_,*x_,*g_,*u_,*h_,*q_,*o_;
    size_t slots_,weights_,scratch_,bytes_;
    std::vector<bool> uploaded_;
    std::string capture_;
    bool capture_first_only_;
    using Stages=strata::core::portable_detail::StrataVkNativeStages;
    using Gu=strata::core::portable_detail::StrataVkQ8Gu;
    using Down=strata::core::portable_detail::StrataVkQ8Down;
    std::unique_ptr<Stages,decltype(&strata::core::portable_detail::strata_vk_native_stages_destroy)> activation_{nullptr,strata::core::portable_detail::strata_vk_native_stages_destroy};
    std::unique_ptr<Gu,decltype(&strata::core::portable_detail::strata_vk_q8_gu_destroy)> gu_{nullptr,strata::core::portable_detail::strata_vk_q8_gu_destroy};
    std::unique_ptr<Down,decltype(&strata::core::portable_detail::strata_vk_q8_down_destroy)> down_{nullptr,strata::core::portable_detail::strata_vk_q8_down_destroy};
    void dump(const char* suffix,ggml_tensor* t) {
        if(capture_.empty())return;
        std::vector<uint8_t> b(ggml_nbytes(t));ggml_backend_tensor_get(t,b.data(),0,b.size());
        std::ofstream f(capture_+suffix,std::ios::binary);require(bool(f.write(reinterpret_cast<const char*>(b.data()),b.size())),"native Q8 boundary write failed");
    }
public:
    NativeQ8Cache(ggml_backend_t backend,size_t slots,size_t budget,const std::string& receipt,const std::string& capture,bool capture_first_only):backend_(backend),slots_(slots),uploaded_(slots,false),capture_(capture),capture_first_only_(capture_first_only) {
        using namespace strata::core::portable_detail;
        require(!receipt.empty() && slots>=1 && slots<=24,"native Q8 requires receipt prefix and 1..24 frozen slots");
        weights_=native_q8_weight_bytes(slots,budget);
        ctx_.reset(ggml_init({64*1024,nullptr,true}));require(bool(ctx_),"native Q8 metadata allocation failed");
        w_=ggml_new_tensor_1d(ctx_.get(),GGML_TYPE_I32,weights_/4);
        i_=ggml_new_tensor_1d(ctx_.get(),GGML_TYPE_F32,8*H);
        x_=ggml_new_tensor_1d(ctx_.get(),GGML_TYPE_I32,8*(H/32)*9);
        g_=ggml_new_tensor_1d(ctx_.get(),GGML_TYPE_F32,64*F);
        u_=ggml_new_tensor_1d(ctx_.get(),GGML_TYPE_F32,64*F);
        h_=ggml_new_tensor_1d(ctx_.get(),GGML_TYPE_F32,64*F);
        q_=ggml_new_tensor_1d(ctx_.get(),GGML_TYPE_I32,64*(F/32)*9);
        o_=ggml_new_tensor_1d(ctx_.get(),GGML_TYPE_F32,64*H);
        auto buft=ggml_backend_get_default_buffer_type(backend);
        bytes_=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx_.get(),buft);
        require(bytes_>=weights_ && bytes_<=budget && bytes_<=ggml_backend_buft_get_max_size(buft),"native Q8 tensor budget/backend block exceeded");
        scratch_=bytes_-weights_;
        // Existing component owners allocate metadata via the same backend.
        // Charge an explicit 64 KiB allowance, then verify actual allocation sizes.
        require(scratch_<=SIZE_MAX-65536 && bytes_<=budget && 65536<=budget-bytes_,"native Q8 component metadata budget exceeded");
        activation_.reset(strata_vk_native_stages_create_quant_rn_observation(backend,(receipt+".activation.spv").c_str()));
        gu_.reset(strata_vk_q8_gu_create(backend,(receipt+".gu.spv").c_str()));
        down_.reset(strata_vk_q8_down_create(backend,(receipt+".down.spv").c_str()));
        const size_t component_bytes=strata_vk_native_stages_bytes(activation_.get())+strata_vk_q8_gu_bytes(gu_.get())+strata_vk_q8_down_bytes(down_.get());
        require(component_bytes<=65536,"native Q8 metadata exceeds charged allowance");
        buffer_.reset(ggml_backend_alloc_ctx_tensors(ctx_.get(),backend));require(bool(buffer_),"native Q8 buffer allocation failed");
        require(ggml_backend_buffer_get_size(buffer_.get())==bytes_,"native Q8 allocation accounting changed");
        bytes_+=component_bytes;scratch_+=component_bytes;
        std::fprintf(stderr,"native_q8_provider_observation=true qualified=false quantizer_qualified=false fp32_denorm_mode=unspecified tensor_bytes=%zu scratch_bytes=%zu\n",bytes_,scratch_);
    }
    GpuCounters allocation_snapshot() const {
        using namespace strata::core::portable_detail;
        GpuCounters c;
        c.gpu_allocated_bytes=ggml_backend_buffer_get_size(buffer_.get())+
            strata_vk_native_stages_bytes(activation_.get())+strata_vk_q8_gu_bytes(gu_.get())+strata_vk_q8_down_bytes(down_.get());
        c.gpu_weight_bytes=weights_;c.gpu_scratch_bytes=c.gpu_allocated_bytes-weights_;
        c.gpu_allocation_count=1+bool(activation_)+bool(gu_)+bool(down_);return c;
    }
    size_t size() const{return slots_;}
    size_t buffer_bytes()const{return bytes_;}
    size_t scratch_bytes()const{return scratch_;}
    void upload(size_t slot,const std::vector<uint8_t>& blob) {
        require(slot<slots_ && !uploaded_[slot] && blob.size()==blob_bytes,"native Q8 invalid upload");
        ggml_backend_tensor_set(w_,blob.data(),slot*blob_bytes,blob_bytes);uploaded_[slot]=true;
    }
    EntryRowsReply run(const GpuPlan& p,const InferRequest& r,GpuExecutionTrace* trace=nullptr) {
        using namespace strata::core::portable_detail;
        auto t=native_q8_groups(p,r);const uint32_t n=uint32_t(t.dst.size());
        std::vector<uint32_t> go,doo;
        for(auto slot:t.asset_slots){require(size_t(slot)<slots_ && uploaded_[slot],"native Q8 slot unavailable");go.push_back(uint32_t(slot*blob_bytes));doo.push_back(uint32_t(slot*blob_bytes+down_offset));}
        Context views(ggml_init({16384,nullptr,true}),ggml_free);require(bool(views),"native Q8 view metadata allocation failed");
        auto view=[&](ggml_tensor* base,size_t count){return native_q8_view(views.get(),base,count);};
        auto* i=view(i_,size_t(r.T)*H);auto* x=view(x_,size_t(r.T)*(H/32)*9);
        auto* g=view(g_,size_t(n)*F);auto* u=view(u_,size_t(n)*F);auto* h=view(h_,size_t(n)*F);
        auto* q=view(q_,size_t(n)*(F/32)*9);auto* o=view(o_,size_t(n)*H);
        ggml_backend_tensor_set(i,r.activations.data(),0,r.activations.size()*4);
        const int32_t starts[]={0,r.T};
        strata_vk_native_stages_run(activation_.get(),0,i,i,i,x,H,starts,2);
        strata_vk_q8_gu_run(gu_.get(),w_,x,g,u,{H,F,n,r.T,uint32_t(t.experts.size()),go,t.start,t.tok});
        strata_vk_native_stages_run(activation_.get(),2,g,u,h,q,F,t.start.data(),t.start.size());
        strata_vk_q8_down_run(down_.get(),w_,q,o,{H,F,n,n,uint32_t(t.experts.size()),doo,t.start,t.dst});
        std::vector<float> rows(size_t(n)*H);ggml_backend_tensor_get(o,rows.data(),0,rows.size()*4);finite(rows);
        dump(".input.q8_1",x);dump(".gate.f32",g);dump(".up.f32",u);dump(".hidden.q8_1",q);dump(".output.f32",o);
        if(capture_first_only_)capture_.clear();
        std::vector<int32_t> packed(r.mask.size(),-1);for(size_t j=0;j<t.original.size();++j)packed[t.original[j]]=int32_t(j);
        EntryRowsReply result;result.session=r.session;result.sequence=r.sequence;
        for(size_t j=0;j<r.mask.size();++j)if(r.mask[j]) {
            require(packed[j]>=0,"native Q8 missing packed row");result.indices.push_back(uint16_t(j));
            result.rows.insert(result.rows.end(),rows.begin()+size_t(packed[j])*H,rows.begin()+(size_t(packed[j])+1)*H);
        }
        if(trace)*trace=std::move(t);
        return result;
    }
};
}
