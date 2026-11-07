#pragma once
// Bounded layer 47 only; this is not full-model admission. The borrowed QSA,
// FFN rows, head and backend must outlive this owner and cannot be operated
// independently while attached. Operations are synchronous and single-threaded.
#include "portable_native_ffn_composition.hpp"
#include "strata/core/portable_model.hpp"
#include <atomic>

namespace strata::core::portable_detail {
class PortableNativeTerminalComposition {
    using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
    using Buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>;
    using Write=std::unique_ptr<StrataVkNativeHcPending,decltype(&strata_vk_native_hc_pending_destroy)>;
    struct Bank {ggml_tensor *residual,*block,*inject,*candidate,*gate,*exp; std::array<ggml_tensor*,2> row{};};
    struct Active {
        ggml_tensor* tensor=nullptr; uint64_t position=0,owner_id=0,sequence=0; bool live=false;
        static bool current(const void* p,const PortableFinalHcResidual& v) {
            auto& a=*static_cast<const Active*>(p);
            return a.live && v.owner==p && v.tensor==a.tensor && v.position==a.position &&
                   v.owner_id==a.owner_id && v.write_sequence==a.sequence && v.layer==47;
        }
    };
    PortableNativeQsaComposition& source_; PortableNativeFfnRows& rows_; PortableHead& head_;
    std::unique_ptr<PortableNativeHcComposed> hc_;
    Context ctx_{nullptr,ggml_free}, views_{nullptr,ggml_free};
    Buffer buffer_{nullptr,ggml_backend_buffer_free};
    Write write_{nullptr,strata_vk_native_hc_pending_destroy};
    std::array<Bank,2> banks_{};
    uint64_t position_,generation_,owner_id_,sequence_=0,reset_generation_=0; bool running_=false,poisoned_=false;
    static void need(bool b,const char* m){if(!b)throw std::runtime_error(m);}
    static uint64_t next_owner(){
        static std::atomic<uint64_t> id{1}; auto value=id.load(std::memory_order_relaxed);
        do {need(value!=UINT64_MAX,"terminal: owner identity exhausted");}
        while(!id.compare_exchange_weak(value,value+1,std::memory_order_relaxed)); return value;
    }
public:
    // Result is an immutable copy; it does not assert that upstream provenance
    // remains live after publication. Receipts/counters record paid head work,
    // including rows completed before a later row or preparation failure.
    struct Result {uint64_t first; std::vector<std::vector<float>> logits;};
    // Own metadata and simultaneous T2 QSA/HC consumer copies, final-write
    // staging and two logits vectors. Borrowed owners, allocator overhead and
    // caller-retained results are excluded. Vocabulary comes from the validated head.
    static size_t host_bound(const PortableHead& head) {
        const size_t v=head.vocabulary_size();
        need(v <= (SIZE_MAX-(2u<<20))/(2*sizeof(float)),"terminal: host bound overflow");
        return (2u<<20)+v*2*sizeof(float);
    }
    using Prepare=std::function<void()>;
    PortableNativeTerminalComposition(PortableNativeQsaComposition& source,ggml_backend_t backend,
        const std::vector<PortableTensorView>& roles,PortableNativeFfnRows& rows,PortableHead& head,
        size_t hc_device_budget,size_t hc_host_budget,size_t write_device_budget,const char* hc_prefix,const char* write_receipt,
        std::optional<PortableNativeHcComposed::ComponentCaps> caps=std::nullopt,PortableHcReadSelection selection={})
      :source_(source),rows_(rows),head_(head),position_(source.position()),generation_(source.generation()),owner_id_(next_owner()) {
        need(source.layer()==47&&!source.poisoned()&&source.provenance_live()&&rows.layer()==47&&backend&&write_receipt&&*write_receipt,
             "terminal: source/rows/backend");
        const size_t own_host=host_bound(head_);
        need(hc_host_budget>=own_host,"terminal: host budget");
        if(selection.method==PortableHcReadMethod::legacy_decode&&!caps){selection.validate();hc_=NativeFfnHcFactory::create(47,backend,roles,hc_device_budget,hc_host_budget-own_host,hc_prefix,{},caps,false,true);}
        else hc_=NativeFfnHcFactory::create_selected(47,backend,roles,hc_device_budget,hc_host_budget-own_host,hc_prefix,caps.value_or(PortableNativeHcComposed::ComponentCaps{}),selection,false,true);
        ctx_.reset(ggml_init({16384,nullptr,true}));need(bool(ctx_),"terminal: metadata");
        for(size_t i=0;i<2;++i){auto make=[&](int n){auto*t=ggml_new_tensor_2d(ctx_.get(),GGML_TYPE_F32,n,i+1);need(t,"terminal: tensor");return t;};
            banks_[i]={make(10240),make(2560),make(4),make(10240),make(4),make(4),{}};}
        auto bound=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx_.get(),ggml_backend_get_default_buffer_type(backend));
        need(bound<=write_device_budget,"terminal: write bank budget");
        write_.reset(strata_vk_native_hc_write_create(backend,write_receipt,write_device_budget-bound));
        need(write_&&strata_vk_native_hc_pending_bytes(write_.get())<=write_device_budget-bound,"terminal: write owner budget");
        buffer_.reset(ggml_backend_alloc_ctx_tensors(ctx_.get(),backend));
        need(buffer_&&ggml_backend_buffer_get_size(buffer_.get())<=write_device_budget-strata_vk_native_hc_pending_bytes(write_.get()),"terminal: allocation budget");
        views_.reset(ggml_init({8192,nullptr,true}));need(bool(views_),"terminal: view metadata");
        for(size_t n=1;n<=2;++n)for(size_t r=0;r<n;++r){auto*v=ggml_view_1d(views_.get(),banks_[n-1].candidate,10240,r*10240*sizeof(float));
            need(v&&ggml_backend_view_init(v)==GGML_STATUS_SUCCESS&&v->view_src==banks_[n-1].candidate&&v->view_offs==r*10240*sizeof(float),"terminal: row view");banks_[n-1].row[r]=v;}
    }
    PortableNativeTerminalComposition(const PortableNativeTerminalComposition&)=delete;
    PortableNativeTerminalComposition& operator=(const PortableNativeTerminalComposition&)=delete;
    uint64_t position()const{return position_;} bool poisoned()const{return poisoned_;}
    PortableHcReadMethod hc_read_method()const{return hc_->read_method();}
    const PortableNativeQsaComposition* source_identity()const{return &source_;}
    const void* rows_identity()const{return &rows_;}
    uint64_t rows_generation()const{return rows_.generation();}
    size_t vocabulary_size()const{return head_.vocabulary_size();}
    uint64_t reset_generation()const{return reset_generation_;}
    bool available()const{return !running_&&!poisoned_&&!rows_.poisoned()&&rows_.generation()!=UINT64_MAX&&source_.available()&&source_.position()==position_&&source_.generation()==generation_&&generation_!=UINT64_MAX;}
    bool can_run(size_t count)const{return available()&&(count==1||count==2)&&sequence_<=UINT64_MAX-count;}
    // Reset/replay upstream and provider separately; this does not roll back
    // earlier layer commits or external provider side effects.
    void reset(){need(!running_&&!source_.poisoned()&&source_.provenance_live()&&reset_generation_!=UINT64_MAX,"terminal: reset source");
        ++reset_generation_;poisoned_=true;hc_->reset();rows_.reset();position_=source_.position();generation_=source_.generation();poisoned_=false;}
    std::shared_ptr<const Result> run(uint64_t first,const PortableNativeFfnComposition& producer,
        const PortableNativeFfnComposition::Batch& snapshots,const PortableExpertRowsBatch& provider,const Prepare& prepare={}) {
        const size_t count=snapshots.size();
        need(!running_&&!poisoned_&&!source_.poisoned()&&source_.provenance_live()&&!rows_.poisoned()&&provider&&
             (count==1||count==2)&&first==position_&&first<=UINT64_MAX-count&&source_.position()==first&&
             source_.generation()==generation_&&generation_!=UINT64_MAX&&sequence_<=UINT64_MAX-count,
             "terminal: state/position/rows");
        auto result=std::make_shared<Result>();result->first=first;result->logits.resize(count);
        running_=true;struct Guard{bool& b;~Guard(){b=false;}}guard{running_};
        try {
            source_.run(first,producer,snapshots,[&](const PortableNativeQsaComposition::View& v){
                need(v.first()==first&&v.generation()==generation_&&v.rows()==count,"terminal: QSA candidate");
                auto residual=v.residual(),output=v.output(),injection=v.injection();
                hc_->run_post_attention(residual,output,injection,count,[&](const auto& h,uint32_t n){
                    need(n==count,"terminal: HC rows");auto applied=h.read(PortableNativeHcComposed::View::Field::residual);
                    auto inject=h.read(PortableNativeHcComposed::View::Field::inject);auto mixed=h.read(PortableNativeHcComposed::View::Field::mixed);
                    auto computed=rows_.run(mixed,count,provider);need(applied.size()==count*10240&&inject.size()==count*4&&computed.block.size()==count*2560,"terminal: FFN output");
                    auto& b=banks_[count-1];ggml_backend_tensor_set(b.residual,applied.data(),0,applied.size()*4);
                    ggml_backend_tensor_set(b.block,computed.block.data(),0,computed.block.size()*4);ggml_backend_tensor_set(b.inject,inject.data(),0,inject.size()*4);
                    strata_vk_native_hc_pending_run(write_.get(),b.residual,b.block,b.inject,b.candidate,b.gate,b.exp);
                    for(size_t r=0;r<count;++r){Active active{b.row[r],first+r,owner_id_,++sequence_,true};
                        PortableFinalHcResidual identity{active.tensor,active.position,active.owner_id,active.sequence,47,&active,&Active::current};
                        std::string error;bool ok=head_.run_device(identity,result->logits[r],error);active.live=false;
                        if(!ok)throw std::runtime_error("terminal: head: "+error);
                    }
                });
            },[&](const PortableNativeQsaComposition::View& v){
                need(v.first()==first&&v.generation()==generation_&&v.rows()==count&&result->logits.size()==count,"terminal: incomplete candidate");
                if(prepare)prepare();
                need(source_.position()==first&&source_.generation()==generation_,"terminal: source changed during prepare");
            });
        }catch(...){poisoned_=true;throw;}
        position_+=count;++generation_;return result;
    }
};
}
