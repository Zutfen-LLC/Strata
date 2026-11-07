#pragma once
// Standalone pending-boundary batch owner. Not wired into model admission.
#include "strata_vk_native_hc.hpp"
#include "native_hc_boundary.hpp"
#include "portable_native_ple_transaction.hpp"
#include "strata/core/portable_qsa.hpp"
#include "strata/core/portable_gdn.hpp"
#include "ggml-alloc.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strata::core::portable_detail {
class PortableNativeFfnComposition;
class PortableNativeFfnSnapshot;
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
    friend class PortableNativeFfnComposition;
    friend class PortableNativeTerminalComposition;
    friend class PortableNativeLayer0Composition;
    friend class NativeFfnHcFactory;
    int checked_layer_=-1;
    bool checked_attention_=false;
    using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
public:
    struct Bank {
        ggml_tensor *input, *block, *previous, *residual, *pending_gate, *pending_exp;
        ggml_tensor *normalized, *rs, *inject, *low_pre, *low, *low_exp, *up_pre, *gates, *up_exp, *mixed;
    };
    struct ComponentCaps { size_t pending, norm, mix; };
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
            const size_t expected = field==Field::inject ? size_t(access_->rows)*16 :
                                    field==Field::mixed ? size_t(access_->rows)*10240 : size_t(access_->rows)*40960;
            if ((access_->rows!=1 && access_->rows!=2) || !t || ggml_nbytes(t)!=expected)
                throw std::runtime_error("composed HC: view extent");
            std::vector<float> result(expected/sizeof(float));
            ggml_backend_tensor_get(t,result.data(),0,ggml_nbytes(t)); return result;
        }
    private:
        void check() const { if (!access_ || !access_->valid) throw std::runtime_error("composed HC: stale view"); }
    };
    // External consumer side effects are not rolled back on failure. A throwing
    // consumer poisons the owner until reset so callers cannot silently continue
    // after a partially applied downstream mutation.
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
    bool running_=false, poisoned_=false;
    size_t host_budget_ = SIZE_MAX;
    static void require(bool b, const char* s) { if (!b) throw std::runtime_error(s); }
public:
    struct HostBreakdown {
        size_t metadata, state, ple_copies, read_copy, diagnostics, component_runs, total;
    };
    // Payload bound, not RSS: ggml metadata arenas, owner/role objects, two
    // live Access records, PLE token/residual copies and one largest View read.
    // Backend buffers and implementation allocations, allocator metadata,
    // dispatch/consumer captures, producer/snapshot storage, source
    // roles and caller-retained View/read copies are excluded.
    // Consumers must budget their own simultaneous or retained reads and handles.
    // The role bank may outlive the owner through role_bank(); account for it once.
    // prefix_bytes is strlen(prefix) for production; injected dispatch uses zero.
    // Diagnostics covers the prefix and the largest temporary shader path.
    // component_runs conservatively sums serial production preflights: pending
    // input/control vectors, norm input/weights and oracle outputs, and the
    // largest mix scan vector. PLE copies remain live during these calls.
    // Stack arrays, allocator metadata, external backend allocations and caller
    // copies are excluded; injected callbacks are trusted, not sandboxed.
    static HostBreakdown host_breakdown(size_t rows, size_t prefix_bytes=0) {
        if (rows != 1 && rows != 2) throw std::invalid_argument("composed HC: host rows");
        HostBreakdown b{};
        b.metadata = ple_host_add(8192,65536);
        // allocate_shared requests are checked before allocation, including its
        // implementation-defined control blocks. One role and two simultaneous
        // publication records are covered; retained stale views are caller-owned.
        b.state = ple_host_add(sizeof(PortableNativeHcComposed), ple_host_mul(3, size_t(512)));
        b.ple_copies = ple_host_add(ple_host_mul(rows,sizeof(int32_t)),ple_host_mul(ple_host_mul(rows,10240),sizeof(float)));
        b.read_copy = ple_host_mul(ple_host_mul(rows,10240),sizeof(float));
        b.diagnostics = prefix_bytes ? ple_host_add(ple_host_mul(2,prefix_bytes), 2*sizeof(std::string)+32) : 0;
        const size_t pending = ple_host_add(ple_host_mul(2,40960),ple_host_add(10240,ple_host_mul(12,sizeof(float))));
        const size_t norm = ple_host_add(ple_host_add(40960,81920),
            ple_host_add(ple_host_mul(rows,81920),ple_host_mul(rows,32)));
        const size_t mix = 6553600; // one vector at a time; packed down/up each 6553600 bytes
        b.component_runs = ple_host_add(ple_host_add(pending,norm),mix);
        b.total = ple_host_add(ple_host_add(b.metadata,b.state),ple_host_add(ple_host_add(b.ple_copies,b.read_copy),ple_host_add(b.diagnostics,b.component_runs)));
        return b;
    }
    static size_t host_bound(size_t rows, size_t prefix_bytes=0) { return host_breakdown(rows,prefix_bytes).total; }
    // Metadata-only ggml contexts are allocated temporarily; no backend buffer
    // is allocated and no graph or shader is dispatched. Shader/backend internal
    // allocations are not represented by this tensor-buffer bound.
    static size_t tensor_bound(ggml_backend_t backend, const std::array<const ggml_tensor*,4>& source) {
        require(backend, "composed HC: backend");
        Context roles(ggml_init({8192,nullptr,true}),ggml_free);
        Context banks(ggml_init({65536,nullptr,true}),ggml_free);
        require(bool(roles) && bool(banks), "composed HC: estimate metadata");
        const size_t sizes[] = {40960,81920,6553600,6553600};
        for (size_t i=0; i<4; ++i) {
            require(source[i] && ggml_is_contiguous(source[i]) && ggml_nbytes(source[i]) == sizes[i] &&
                    source[i]->type == (i ? GGML_TYPE_I32 : GGML_TYPE_F32), "composed HC: estimate role descriptor");
            require(ggml_dup_tensor(roles.get(),source[i]), "composed HC: estimate role");
        }
        for (int slot=0; slot<2; ++slot) for (int r=1; r<=2; ++r)
            for (int n : {10240,2560,4,10240,4,4,10240,4,4,320,320,320,10240,10240,10240,2560})
                require(ggml_new_tensor_2d(banks.get(),GGML_TYPE_F32,n,r), "composed HC: estimate bank");
        auto type=ggml_backend_get_default_buffer_type(backend);
        return ple_host_add(ggml_backend_alloc_ctx_tensors_from_buft_size(roles.get(),type),
                            ggml_backend_alloc_ctx_tensors_from_buft_size(banks.get(),type));
    }
    // Source tensors and their storage need only survive synchronous construction.
    // The backend must outlive this owner. Model chunk admission remains separate.
    // Injected dispatch is trusted test infrastructure, not a consumer API: it
    // must not retain bank pointers. Production leaves it empty. The owner must
    // not be destroyed during a run or concurrently with any view operation.
    // Caps bound only the components' reported bytes, not hidden backend or
    // pipeline resources; callers must budget those separately.
    PortableNativeHcComposed(ggml_backend_t backend, const ggml_tensor* nw, const ggml_tensor* iw,
                             const ggml_tensor* down, const ggml_tensor* up, size_t budget,
                             const char* prefix, Dispatch mock={}, size_t host_budget=SIZE_MAX,
                             std::optional<ComponentCaps> caps=std::nullopt) : host_budget_(host_budget) {
        const bool injected=bool(mock.pending)||bool(mock.norm)||bool(mock.mix);
        const size_t prefix_bytes=(!injected && prefix) ? std::strlen(prefix) : 0;
        require(host_bound(1,prefix_bytes)<=host_budget_, "composed HC: host budget");
        roles_=std::allocate_shared<NativeHcRoleBank>(PleOwnerAllocator<NativeHcRoleBank>{},backend,std::array<const ggml_tensor*,4>{nw,iw,down,up},budget);
        budget-=roles_->bytes();
        nw=roles_->role(0); iw=roles_->role(1); down=roles_->role(2); up=roles_->role(3);
        ctx_.reset(ggml_init({65536,nullptr,true})); require(bool(ctx_),"composed HC: metadata");
        for(auto& slots:banks_) for(int r=1;r<=2;++r) {
            auto make=[&](int n){auto* t=ggml_new_tensor_2d(ctx_.get(),GGML_TYPE_F32,n,r); require(t!=nullptr,"composed HC: tensor metadata"); return t;};
            slots[r-1]={make(10240),make(2560),make(4),make(10240),make(4),make(4),make(10240),make(4),make(4),make(320),make(320),make(320),make(10240),make(10240),make(10240),make(2560)};
        }
        auto bound=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx_.get(),ggml_backend_get_default_buffer_type(backend));
        require(bound<=budget,"composed HC: budget"); size_t meta=0;
        if (caps) {
            const size_t reserved=ple_host_add(ple_host_add(ple_host_add(ple_host_add(roles_->bytes(),bound),caps->pending),caps->norm),caps->mix);
            require(reserved-roles_->bytes()<=budget,"composed HC: component caps budget");
        }
        if(injected) { require(mock.pending&&mock.norm&&mock.mix,"composed HC: incomplete dispatch"); dispatch_=std::move(mock); }
        else {
            require(prefix && *prefix,"composed HC: receipt prefix");
            std::string p(prefix);
            size_t limit=caps ? caps->pending : budget-bound;
            pending_.reset(strata_vk_native_hc_pending_create(backend,(p+".pending.spv").c_str(),limit));
            require(bool(pending_),"composed HC: pending create"); meta=strata_vk_native_hc_pending_bytes(pending_.get()); require(meta<=limit,"composed HC: pending budget");
            limit=caps ? caps->norm : budget-bound-meta;
            norm_.reset(strata_vk_native_hc_create(backend,(p+".norm.spv").c_str(),limit));
            require(bool(norm_),"composed HC: norm create"); auto n=strata_vk_native_hc_bytes(norm_.get()); require(n<=limit,"composed HC: norm budget"); meta+=n;
            limit=caps ? caps->mix : budget-bound-meta;
            mix_.reset(strata_vk_native_hc_mix_create(backend,(p+".mix.spv").c_str(),limit));
            require(bool(mix_),"composed HC: mix create"); n=strata_vk_native_hc_mix_bytes(mix_.get()); require(n<=limit,"composed HC: mix budget"); meta+=n;
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
        publication_.reset(); published_=-1; rows_=0; poisoned_=false;
    }
    // Only genuine, live producer-issued snapshots are accepted. The boundary
    // rule deliberately rejects the materialized layer0/layer1 PLE boundary.
    void run(int layer, uint64_t first, const PortableQsaLayer& producer,
             const std::vector<std::shared_ptr<const PortableQsaSnapshot>>& snapshots,
             const Consumer& a, const Consumer& b) { run_impl(layer, first, producer, snapshots, a, b); }
    void run(int layer, uint64_t first, const PortableGdnLayer& producer,
             const std::vector<std::shared_ptr<const PortableGdnSnapshot>>& snapshots,
             const Consumer& a, const Consumer& b) { run_impl(layer, first, producer, snapshots, a, b); }
    // Ordinary next-layer consumption of privately issued post-FFN tuples.
    void run(int layer, uint64_t first, const PortableNativeFfnComposition& producer,
             const std::vector<std::shared_ptr<const PortableNativeFfnSnapshot>>& snapshots,
             const Consumer& a, const Consumer& b);
    // Materialized PLE is the sole layer-1 entry; it has no pending HC tuple.
    // The transaction and its handle must remain alive and unmodified throughout
    // synchronous consumption. Revalidation at commit catches reentrant changes.
    void run(int layer, uint64_t first, PortableNativePleTransaction& producer,
             const PortableNativePleTransaction::Handle& handle,
             const Consumer& first_consumer, const Consumer& second_consumer) {
        require(!running_ && !poisoned_ && layer==1 && first_consumer && second_consumer &&
                producer.live(handle), "composed HC: PLE boundary/identity");
        running_=true;
        struct Guard { bool& b; ~Guard(){b=false;} } guard{running_};
        const size_t count=handle.token_count();
        require((count==1 || count==2) && first<=UINT64_MAX-count &&
                handle.position()==first && producer.position()==first &&
                handle.residual_count()==count*10240, "composed HC: PLE position/rows");
        require(host_bound(count)<=host_budget_, "composed HC: host budget");
        const auto tokens=handle.tokens();
        const auto residual=handle.residual();
        // host_bound accounts for exact payload capacities, not allocator slack.
        // Check both copies and their extents before writing the unpublished bank.
        require(tokens.size()==count && tokens.capacity()==count &&
                residual.size()==count*10240 && residual.capacity()==residual.size(),
                "composed HC: PLE copy extent/capacity");
        for(float x:residual) require(std::isfinite(x), "composed HC: PLE nonfinite residual");
        int slot=published_==0?1:0; auto& bank=banks_[slot][count-1];
        ggml_backend_tensor_set(bank.residual,residual.data(),0,residual.size()*sizeof(float));
        dispatch_.norm(bank,uint32_t(count)); dispatch_.mix(bank);
        auto candidate=std::allocate_shared<Access>(PleOwnerAllocator<Access>{}); candidate->bank=&bank; candidate->rows=uint32_t(count);
        View view(candidate);
        try {
            first_consumer(view,uint32_t(count)); second_consumer(view,uint32_t(count));
            producer.commit(handle,true);
        } catch (...) { candidate->valid=false; poisoned_=true; throw; }
        if (publication_) publication_->valid=false;
        publication_=std::move(candidate); published_=slot; rows_=uint32_t(count);
    }
private:
    // Layer zero starts with a materialized embedding, never a pending tuple.
    void run_layer0_embedding(const std::vector<float>& residual, size_t count, const Consumer& consume) {
        require(!running_ && !poisoned_ && (count==1 || count==2) && consume &&
                residual.size()==count*10240 && host_bound(count)<=host_budget_, "composed HC: layer0 entry");
        for(float x:residual) require(std::isfinite(x), "composed HC: nonfinite embedding");
        running_=true; struct Guard { bool& b; ~Guard(){b=false;} } guard{running_};
        int slot=published_==0?1:0; auto& b=banks_[slot][count-1];
        std::shared_ptr<Access> candidate;
        try {
            ggml_backend_tensor_set(b.residual,residual.data(),0,residual.size()*sizeof(float));
            dispatch_.norm(b,uint32_t(count)); dispatch_.mix(b);
            candidate=std::allocate_shared<Access>(PleOwnerAllocator<Access>{});
            candidate->bank=&b; candidate->rows=uint32_t(count);
            consume(View(candidate),uint32_t(count));
        } catch (...) { if(candidate) candidate->valid=false; poisoned_=true; throw; }
        if(publication_) publication_->valid=false;
        publication_=std::move(candidate); published_=slot; rows_=uint32_t(count);
    }
    // Only the connected FFN owner can invoke this same-layer entry, inside a
    // live upstream producer callback. These copies are not an identity API.
    void run_post_attention(const std::vector<float>& residual,const std::vector<float>& output,
                            const std::vector<float>& injection,size_t count,const Consumer& consume) {
        require(!running_&&!poisoned_&&(count==1||count==2)&&consume &&
                residual.size()==count*10240&&output.size()==count*2560&&injection.size()==count*4,
                "composed HC: post-attention tuple");
        require(host_bound(count)<=host_budget_,"composed HC: host budget");
        for(const auto* v:{&residual,&output,&injection})for(float x:*v)require(std::isfinite(x),"composed HC: nonfinite post-attention tuple");
        running_=true;struct Guard{bool& b;~Guard(){b=false;}}guard{running_};
        int slot=published_==0?1:0;auto& b=banks_[slot][count-1];
        ggml_backend_tensor_set(b.input,residual.data(),0,residual.size()*sizeof(float));
        ggml_backend_tensor_set(b.block,output.data(),0,output.size()*sizeof(float));
        ggml_backend_tensor_set(b.previous,injection.data(),0,injection.size()*sizeof(float));
        std::shared_ptr<Access> candidate;
        try {
            dispatch_.pending(b);dispatch_.norm(b,uint32_t(count));dispatch_.mix(b);
            candidate=std::allocate_shared<Access>(PleOwnerAllocator<Access>{});candidate->bank=&b;candidate->rows=uint32_t(count);
            consume(View(candidate),uint32_t(count));
        }catch(...){if(candidate)candidate->valid=false;poisoned_=true;throw;}
        if(publication_)publication_->valid=false;
        publication_=std::move(candidate);published_=slot;rows_=uint32_t(count);
    }
    template<class Producer, class Snapshot>
    void run_impl(int layer, uint64_t first, const Producer& producer,
             const std::vector<std::shared_ptr<const Snapshot>>& snapshots,
             const Consumer& first_consumer, const Consumer& second_consumer) {
        require(!running_ && !poisoned_ && native_hc_boundary(layer).pending_attention && snapshots.size()>=1 && snapshots.size()<=2 &&
                first<=UINT64_MAX-snapshots.size() && first_consumer && second_consumer,"composed HC: boundary/rows/consumers");
        require(host_bound(snapshots.size())<=host_budget_, "composed HC: host budget");
        // Hold the exact issued handles independently of the caller's vector.
        // A synchronous consumer can otherwise reorder or clear that vector.
        const size_t held_count=snapshots.size();
        std::array<std::shared_ptr<const Snapshot>,2> held{};
        for(size_t i=0;i<held_count;++i) held[i]=snapshots[i];
        running_=true;
        struct Guard {bool& b; ~Guard(){b=false;}} guard{running_};
        int slot=published_==0?1:0; auto& b=banks_[slot][held_count-1];
        for(size_t i=0;i<held_count;++i) {
            auto& s=held[i]; require(s && s->valid_for(producer,layer,first+i),"composed HC: snapshot identity");
            auto& t=s->tuple(); require(t.residual.size()==10240 && t.block.size()==2560 && t.injection.size()==4,"composed HC: tuple span");
        }
        // Validate the entire batch before writing even the unpublished bank.
        for(size_t i=0;i<held_count;++i) {
            const auto& t=held[i]->tuple();
            ggml_backend_tensor_set(b.input,t.residual.data(),i*40960,40960);
            ggml_backend_tensor_set(b.block,t.block.data(),i*10240,10240);
            ggml_backend_tensor_set(b.previous,t.injection.data(),i*16,16);
        }
        const uint32_t count=uint32_t(held_count);
        dispatch_.pending(b); dispatch_.norm(b,count); dispatch_.mix(b);
        auto candidate=std::allocate_shared<Access>(PleOwnerAllocator<Access>{}); candidate->bank=&b; candidate->rows=count;
        View view(candidate);
        try {
            first_consumer(view,count); second_consumer(view,count);
            // Consumers are synchronous but may mutate the producer. Do not
            // publish a view derived from snapshots invalidated during use.
            for(size_t i=0;i<held_count;++i)
                require(held[i] && held[i]->valid_for(producer,layer,first+i),
                        "composed HC: snapshot invalidated during consumption");
        }
        catch (...) { candidate->valid=false; poisoned_=true; throw; }
        if (publication_) publication_->valid=false;
        publication_=std::move(candidate); published_=slot; rows_=count;
    }
};
}
