#pragma once
// Standalone pending-boundary batch owner. Not wired into model admission.
#include "strata_vk_native_hc.hpp"
#include "native_hc_boundary.hpp"
#include "strata/core/portable_qsa.hpp"
#include "strata/core/portable_gdn.hpp"
#include "ggml-alloc.h"
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strata::core::portable_detail {
// Immutable snapshot of the already-packed roles. The backend remains borrowed.
// Payload is 13,230,080 bytes before backend alignment. This allocation is
// additional to the source bank while the latter remains resident; callers
// must account for both in any enclosing resident-memory budget.
class NativeHcRoleBank {
    using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
    ggml_backend_t backend_;
    Context ctx_{nullptr, ggml_free};
    Buffer buffer_{nullptr, ggml_backend_buffer_free};
    std::array<ggml_tensor*,4> roles_{};
    size_t bytes_=0;
public:
    NativeHcRoleBank(ggml_backend_t backend, const std::array<const ggml_tensor*,4>& source, size_t budget): backend_(backend) {
        auto need=[](bool ok){if(!ok) throw std::runtime_error("composed HC: invalid role bank or budget");};
        need(backend!=nullptr);
        const size_t sizes[]={40960,81920,6553600,6553600};
        for(size_t i=0;i<4;++i) {
            auto* t=source[i];
            need(t && t->buffer && ggml_is_contiguous(t) && ggml_nbytes(t)==sizes[i] &&
                 t->type==(i ? GGML_TYPE_I32 : GGML_TYPE_F32) &&
                 // GGML Vulkan's default type is &ctx->device->buffer_type;
                 // its allocator uses that type's device for the buffer context
                 // (ggml-vulkan.cpp: ggml_backend_vk_get_default_buffer_type and
                 // ggml_backend_vk_buffer_type_alloc_buffer). This proves device
                 // identity for ordinary Vulkan allocations, not arbitrary backends.
                 ggml_backend_buffer_get_type(t->buffer)==ggml_backend_get_default_buffer_type(backend));
            for(size_t j=0;j<i;++j) if(t->buffer==source[j]->buffer) {
                auto a=reinterpret_cast<uintptr_t>(t->data), b=reinterpret_cast<uintptr_t>(source[j]->data);
                need(a<=b ? b-a>=sizes[i] : a-b>=sizes[j]);
            }
        }
        ctx_.reset(ggml_init({8192,nullptr,true})); need(bool(ctx_));
        for(size_t i=0;i<4;++i) { roles_[i]=ggml_dup_tensor(ctx_.get(),source[i]); need(roles_[i]!=nullptr); }
        need(ggml_backend_alloc_ctx_tensors_from_buft_size(ctx_.get(),ggml_backend_get_default_buffer_type(backend))<=budget);
        buffer_.reset(ggml_backend_alloc_ctx_tensors(ctx_.get(),backend)); need(bool(buffer_));
        bytes_=ggml_backend_buffer_get_size(buffer_.get()); need(bytes_<=budget);
        for(size_t i=0;i<4;++i) ggml_backend_tensor_copy(source[i],roles_[i]);
    }
    const ggml_tensor* role(size_t i) const { return roles_.at(i); }
    size_t bytes() const { return bytes_; }
    ggml_backend_t backend() const { return backend_; }
};
class PortableNativeHcComposed {
    using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
public:
    struct Bank {
        ggml_tensor *input, *block, *previous, *residual, *pending_gate, *pending_exp;
        ggml_tensor *normalized, *rs, *inject, *low_pre, *low, *low_exp, *up_pre, *gates, *up_exp, *mixed;
    };
    struct Dispatch {
        std::function<void(const Bank&)> pending;
        std::function<void(const Bank&, uint32_t)> norm;
        std::function<void(const Bank&)> mix;
    };
private:
    struct Access { const Bank* bank=nullptr; uint32_t rows=0; bool valid=true; };
public:
    // A handle never exposes tensor metadata or storage. Copies can be retained,
    // but every read checks validity and returns an independent host copy.
    // Like the owner, handles are single-threaded and must not race owner operations.
    class View {
        std::shared_ptr<Access> access_;
        explicit View(std::shared_ptr<Access> a): access_(std::move(a)) {}
        friend class PortableNativeHcComposed;
    public:
        enum class Field { residual, normalized, inject, mixed };
        uint32_t rows() const { check(); return access_->rows; }
        std::vector<float> read(Field field) const {
            check(); const Bank& b=*access_->bank; const ggml_tensor* t=nullptr;
            switch(field) {
                case Field::residual: t=b.residual; break;
                case Field::normalized: t=b.normalized; break;
                case Field::inject: t=b.inject; break;
                case Field::mixed: t=b.mixed; break;
                default: throw std::runtime_error("composed HC: invalid field");
            }
            std::vector<float> result(ggml_nbytes(t)/sizeof(float));
            ggml_backend_tensor_get(t,result.data(),0,ggml_nbytes(t)); return result;
        }
    private:
        void check() const { if (!access_ || !access_->valid) throw std::runtime_error("composed HC: stale view"); }
    };
    // External consumer side effects are not rolled back on failure.
    using Consumer = std::function<void(const View&, uint32_t)>;
private:
    // The backend is borrowed and must outlive this owner and its role bank.
    std::shared_ptr<const NativeHcRoleBank> roles_;
    std::shared_ptr<Access> publication_;
    Context ctx_{nullptr, ggml_free};
    Buffer buffer_{nullptr, ggml_backend_buffer_free};
    std::unique_ptr<StrataVkNativeHcPending,decltype(&strata_vk_native_hc_pending_destroy)> pending_{nullptr,strata_vk_native_hc_pending_destroy};
    std::unique_ptr<StrataVkNativeHc,decltype(&strata_vk_native_hc_destroy)> norm_{nullptr,strata_vk_native_hc_destroy};
    std::unique_ptr<StrataVkNativeHcMix,decltype(&strata_vk_native_hc_mix_destroy)> mix_{nullptr,strata_vk_native_hc_mix_destroy};
    std::array<std::array<Bank,2>,2> banks_{}; // publication slot, exact row count minus one
    Dispatch dispatch_;
    int published_=-1; uint32_t rows_=0; size_t bytes_=0;
    bool running_=false;
    static void require(bool b, const char* s) { if (!b) throw std::runtime_error(s); }
public:
    // Source tensors and their storage need only survive synchronous construction.
    // The backend must outlive this owner. Model chunk admission remains separate.
    // Injected dispatch is trusted test infrastructure, not a consumer API: it
    // must not retain bank pointers. Production leaves it empty. The owner must
    // not be destroyed during a run or concurrently with any view operation.
    PortableNativeHcComposed(ggml_backend_t backend, const ggml_tensor* nw, const ggml_tensor* iw,
                             const ggml_tensor* down, const ggml_tensor* up, size_t budget,
                             const char* prefix, Dispatch mock={}) {
        roles_=std::make_shared<NativeHcRoleBank>(backend,std::array<const ggml_tensor*,4>{nw,iw,down,up},budget);
        budget-=roles_->bytes();
        nw=roles_->role(0); iw=roles_->role(1); down=roles_->role(2); up=roles_->role(3);
        ctx_.reset(ggml_init({65536,nullptr,true})); require(bool(ctx_),"composed HC: metadata");
        for(auto& slots:banks_) for(int r=1;r<=2;++r) {
            auto make=[&](int n){auto* t=ggml_new_tensor_2d(ctx_.get(),GGML_TYPE_F32,n,r); require(t!=nullptr,"composed HC: tensor metadata"); return t;};
            slots[r-1]={make(10240),make(2560),make(4),make(10240),make(4),make(4),make(10240),make(4),make(4),make(320),make(320),make(320),make(10240),make(10240),make(10240),make(2560)};
        }
        auto bound=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx_.get(),ggml_backend_get_default_buffer_type(backend));
        require(bound<=budget,"composed HC: budget"); size_t meta=0;
        bool injected=bool(mock.pending)||bool(mock.norm)||bool(mock.mix);
        if(injected) { require(mock.pending&&mock.norm&&mock.mix,"composed HC: incomplete dispatch"); dispatch_=std::move(mock); }
        else {
            require(prefix && *prefix,"composed HC: receipt prefix");
            std::string p(prefix);
            pending_.reset(strata_vk_native_hc_pending_create(backend,(p+".pending.spv").c_str(),budget-bound));
            require(bool(pending_),"composed HC: pending create"); meta=strata_vk_native_hc_pending_bytes(pending_.get()); require(meta<=budget-bound,"composed HC: budget");
            norm_.reset(strata_vk_native_hc_create(backend,(p+".norm.spv").c_str(),budget-bound-meta));
            require(bool(norm_),"composed HC: norm create"); auto n=strata_vk_native_hc_bytes(norm_.get()); require(n<=budget-bound-meta,"composed HC: budget"); meta+=n;
            mix_.reset(strata_vk_native_hc_mix_create(backend,(p+".mix.spv").c_str(),budget-bound-meta));
            require(bool(mix_),"composed HC: mix create"); n=strata_vk_native_hc_mix_bytes(mix_.get()); require(n<=budget-bound-meta,"composed HC: budget"); meta+=n;
            dispatch_.pending=[this](const Bank& b){strata_vk_native_hc_pending_run(pending_.get(),b.input,b.block,b.previous,b.residual,b.pending_gate,b.pending_exp);};
            dispatch_.norm=[this,nw,iw](const Bank& b,uint32_t t){strata_vk_native_hc_run(norm_.get(),b.residual,nw,iw,b.normalized,b.rs,b.inject,t);};
            dispatch_.mix=[this,down,up](const Bank& b){strata_vk_native_hc_mix_run(mix_.get(),b.normalized,down,up,b.low_pre,b.low,b.low_exp,b.up_pre,b.gates,b.up_exp,b.mixed);};
        }
        buffer_.reset(ggml_backend_alloc_ctx_tensors(ctx_.get(),backend)); require(bool(buffer_),"composed HC: allocation");
        auto actual=ggml_backend_buffer_get_size(buffer_.get()); require(actual<=budget-meta,"composed HC: actual budget"); bytes_=actual+meta+roles_->bytes();
    }
    PortableNativeHcComposed(const PortableNativeHcComposed&)=delete;
    PortableNativeHcComposed& operator=(const PortableNativeHcComposed&)=delete;
    ~PortableNativeHcComposed() { if (publication_) publication_->valid=false; }
    size_t bytes() const {return bytes_;}
    std::shared_ptr<const NativeHcRoleBank> role_bank() const { return roles_; }
    uint32_t rows() const {return rows_;}
    // Publication exposes metadata only, never a pointer into reusable banks.
    // Scratch banks are available solely during synchronous dispatch/consumption.
    // Reset invalidates publication; scratch stays allocated.
    void reset() {
        require(!running_, "composed HC: reset during active run");
        if (publication_) publication_->valid=false;
        publication_.reset(); published_=-1; rows_=0;
    }
    // Only genuine, live producer-issued snapshots are accepted. The boundary
    // rule deliberately rejects the materialized layer0/layer1 PLE boundary.
    void run(int layer, uint64_t first, const PortableQsaLayer& producer,
             const std::vector<std::shared_ptr<const PortableQsaSnapshot>>& snapshots,
             const Consumer& a, const Consumer& b) { run_impl(layer, first, producer, snapshots, a, b); }
    void run(int layer, uint64_t first, const PortableGdnLayer& producer,
             const std::vector<std::shared_ptr<const PortableGdnSnapshot>>& snapshots,
             const Consumer& a, const Consumer& b) { run_impl(layer, first, producer, snapshots, a, b); }
private:
    template<class Producer, class Snapshot>
    void run_impl(int layer, uint64_t first, const Producer& producer,
             const std::vector<std::shared_ptr<const Snapshot>>& snapshots,
             const Consumer& first_consumer, const Consumer& second_consumer) {
        require(!running_ && native_hc_boundary(layer).pending_attention && snapshots.size()>=1 && snapshots.size()<=2 &&
                first<=UINT64_MAX-snapshots.size() && first_consumer && second_consumer,"composed HC: boundary/rows/consumers");
        running_=true;
        struct Guard {bool& b; ~Guard(){b=false;}} guard{running_};
        int slot=published_==0?1:0; auto& b=banks_[slot][snapshots.size()-1];
        for(size_t i=0;i<snapshots.size();++i) {
            auto& s=snapshots[i]; require(s && s->valid_for(producer,layer,first+i),"composed HC: snapshot identity");
            auto& t=s->tuple(); require(t.residual.size()==10240 && t.block.size()==2560 && t.injection.size()==4,"composed HC: tuple span");
        }
        // Validate the entire batch before writing even the unpublished bank.
        for(size_t i=0;i<snapshots.size();++i) {
            const auto& t=snapshots[i]->tuple();
            ggml_backend_tensor_set(b.input,t.residual.data(),i*40960,40960);
            ggml_backend_tensor_set(b.block,t.block.data(),i*10240,10240);
            ggml_backend_tensor_set(b.previous,t.injection.data(),i*16,16);
        }
        const uint32_t count=uint32_t(snapshots.size());
        dispatch_.pending(b); dispatch_.norm(b,count); dispatch_.mix(b);
        auto candidate=std::make_shared<Access>(); candidate->bank=&b; candidate->rows=count;
        View view(candidate);
        try { first_consumer(view,count); second_consumer(view,count); }
        catch (...) { candidate->valid=false; throw; }
        if (publication_) publication_->valid=false;
        publication_=std::move(candidate); published_=slot; rows_=count;
    }
};
}
