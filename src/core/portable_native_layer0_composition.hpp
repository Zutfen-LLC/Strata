#pragma once
// Bounded token entry through layer zero. This owner is not model admission.
// Borrowed embedding, projection owners, FFN rows and backend outlive this owner.
#include "portable_native_hc_composed.hpp"
#include "portable_native_gdn_input_rows.hpp"
#include "portable_native_gdn_core_rows.hpp"
#include "portable_native_gdn_output_rows.hpp"
#include "portable_native_ffn_rows.hpp"
#include "strata/artifact/dequant.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace strata::core::portable_detail {
class PortableNativeLayer0Composition;
class PortableNativeLayer0Rows {
    struct Epoch { bool valid=true; };
    struct Record {
        bool committed=false, valid=true;
        uint64_t position=0;
        std::vector<int32_t> tokens;
        std::vector<float> residual;
    };
    std::weak_ptr<Epoch> epoch_;
    std::shared_ptr<const Record> record_;
    PortableNativeLayer0Rows(const std::shared_ptr<Epoch>& e,const std::shared_ptr<Record>& r):epoch_(e),record_(r){}
    friend class PortableNativeLayer0Composition;
public:
    PortableNativeLayer0Rows(const PortableNativeLayer0Rows&) = delete;
    PortableNativeLayer0Rows& operator=(const PortableNativeLayer0Rows&) = delete;
    PortableNativeLayer0Rows(PortableNativeLayer0Rows&&) = delete;
    PortableNativeLayer0Rows& operator=(PortableNativeLayer0Rows&&) = delete;
    bool live() const { auto e=epoch_.lock(); return e && e->valid && record_ && record_->valid && record_->committed; }
    bool same_source(const PortableNativeLayer0Rows& other) const {
        auto a=epoch_.lock(), b=other.epoch_.lock(); return a && a==b && a->valid;
    }
    uint64_t position() const { check(); return record_->position; }
    std::vector<int32_t> tokens() const { check(); return record_->tokens; }
    std::vector<float> residual() const { check(); return record_->residual; }
private:
    void check() const { if(!live()) throw std::runtime_error("layer0 rows: stale source"); }
};

// Checked Q8_0 token decode, matching portable_model's 80 blocks and four streams.
inline std::vector<float> native_layer0_embedding(const PortableTensorView& table,const std::vector<int32_t>& tokens) {
    constexpr size_t row_bytes=80*34;
    if(table.name!="token_embd.weight" || table.type!=8 || table.shape.size()!=2 ||
       table.shape[0]!=2560 || table.shape[1]<=0 || table.shape[1]>1048576 || !table.data ||
       table.bytes!=size_t(table.shape[1])*row_bytes || (tokens.size()!=1 && tokens.size()!=2))
        throw std::runtime_error("layer0 embedding: descriptor/rows");
    for(auto t:tokens) if(t<0 || int64_t(t)>=table.shape[1]) throw std::runtime_error("layer0 embedding: token");
    std::vector<float> out(tokens.size()*10240);
    for(size_t r=0;r<tokens.size();++r) {
        const auto* row=static_cast<const uint8_t*>(table.data)+size_t(tokens[r])*row_bytes;
        for(size_t b=0;b<80;++b) strata::dequantize_q8_0(row+b*34,out.data()+r*10240+b*32);
        for(size_t i=0;i<2560;++i) if(!std::isfinite(out[r*10240+i])) throw std::runtime_error("layer0 embedding: nonfinite");
        for(size_t s=1;s<4;++s) std::copy_n(out.data()+r*10240,2560,out.data()+r*10240+s*2560);
    }
    return out;
}

class PortableNativeLayer0Composition {
    using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
    using Buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>;
    PortableTensorView embedding_;
    std::unique_ptr<PortableNativeHcComposed> attention_,ffn_hc_;
    PortableNativeGdnInputRows& input_;
    std::unique_ptr<PortableNativeGdnCoreRows> core_;
    PortableNativeGdnOutputRows& output_;
    PortableNativeFfnRows& ffn_;
    Context ctx_{nullptr,ggml_free}; Buffer buffer_{nullptr,ggml_backend_buffer_free};
    std::unique_ptr<StrataVkNativeHcPending,decltype(&strata_vk_native_hc_pending_destroy)> write_{nullptr,strata_vk_native_hc_pending_destroy};
    struct Bank {ggml_tensor *residual,*block,*inject,*candidate,*gate,*exp;};
    std::array<Bank,2> banks_{};
    std::shared_ptr<PortableNativeLayer0Rows::Epoch> epoch_=std::make_shared<PortableNativeLayer0Rows::Epoch>();
    uint64_t position_=0, reset_generation_=0; bool running_=false,poisoned_=false;
    static void need(bool b,const char* m){if(!b)throw std::runtime_error(m);}
public:
    // Excludes borrowed owners, backend internals, allocator metadata and caller-retained rows.
    static constexpr size_t host_bound(){return 8u<<20;}
    PortableNativeLayer0Composition(const PortableTensorView& embedding,std::unique_ptr<PortableNativeHcComposed> attention,
        PortableNativeGdnInputRows& input,std::unique_ptr<PortableNativeGdnCoreRows> core,PortableNativeGdnOutputRows& output,
        std::unique_ptr<PortableNativeHcComposed> ffn_hc,PortableNativeFfnRows& ffn,ggml_backend_t backend,
        size_t device_budget,size_t host_budget,const char* receipt)
      :embedding_(embedding),attention_(std::move(attention)),ffn_hc_(std::move(ffn_hc)),input_(input),core_(std::move(core)),output_(output),ffn_(ffn) {
        need(attention_&&ffn_hc_&&attention_->checked_layer_==0&&attention_->checked_attention_&&
             ffn_hc_->checked_layer_==0&&!ffn_hc_->checked_attention_&&core_&&!core_->candidate()&&!core_->poisoned()&&ffn_.layer()==0&&backend&&receipt&&*receipt&&host_budget>=host_bound(),"layer0: admission");
        // Validate the descriptor before allocating backend buffers.
        (void)native_layer0_embedding(embedding_,std::vector<int32_t>{0});
        ctx_.reset(ggml_init({16384,nullptr,true}));need(bool(ctx_),"layer0: metadata");
        for(size_t i=0;i<2;++i){auto make=[&](int n){auto*t=ggml_new_tensor_2d(ctx_.get(),GGML_TYPE_F32,n,i+1);need(t,"layer0: tensor");return t;};
            banks_[i]={make(10240),make(2560),make(4),make(10240),make(4),make(4)};}
        auto bound=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx_.get(),ggml_backend_get_default_buffer_type(backend));
        need(bound<=device_budget,"layer0: device budget");
        write_.reset(strata_vk_native_hc_write_create(backend,receipt,device_budget-bound));
        need(write_&&strata_vk_native_hc_pending_bytes(write_.get())<=device_budget-bound,"layer0: write owner budget");
        buffer_.reset(ggml_backend_alloc_ctx_tensors(ctx_.get(),backend));
        need(buffer_&&ggml_backend_buffer_get_size(buffer_.get())<=device_budget-strata_vk_native_hc_pending_bytes(write_.get()),"layer0: allocation budget");
        core_->reset();
    }
    PortableNativeLayer0Composition(const PortableNativeLayer0Composition&)=delete;
    PortableNativeLayer0Composition& operator=(const PortableNativeLayer0Composition&)=delete;
    ~PortableNativeLayer0Composition(){epoch_->valid=false;}
    uint64_t position()const{return position_;} bool poisoned()const{return poisoned_;}
    uint64_t reset_generation()const{return reset_generation_;}
    PortableHcReadMethod attention_hc_read_method()const{return attention_->read_method();}
    PortableHcReadMethod ffn_hc_read_method()const{return ffn_hc_->read_method();}
    size_t vocabulary_size()const{return size_t(embedding_.shape[1]);}
    const void* rows_identity()const{return &ffn_;}
    uint64_t rows_generation()const{return ffn_.generation();}
    bool available()const{return !running_&&!poisoned_&&!core_->candidate()&&!core_->poisoned()&&!ffn_.poisoned()&&ffn_.generation()!=UINT64_MAX;}
    PortableGdnAbMethod ab_method()const{return core_->ab_method();}
    std::vector<float> committed_state()const{return core_->committed_state();}
    std::vector<float> committed_history()const{return core_->committed_history();}
    void reset(){
        need(!running_&&reset_generation_!=UINT64_MAX,"layer0: active reset or generation exhausted"); ++reset_generation_; auto next=std::make_shared<PortableNativeLayer0Rows::Epoch>();
        poisoned_=true; epoch_->valid=false; attention_->reset(); ffn_hc_->reset(); ffn_.reset(); core_->reset();
        epoch_.swap(next);position_=0;poisoned_=false;
    }
    std::shared_ptr<const PortableNativeLayer0Rows> run(uint64_t first,const std::vector<int32_t>& tokens,const PortableExpertRowsBatch& provider){
        need(!running_&&!poisoned_&&!core_->candidate()&&!core_->poisoned()&&!ffn_.poisoned()&&provider&&
             first==position_&&(tokens.size()==1||tokens.size()==2)&&first<=UINT64_MAX-tokens.size(),"layer0: state/position/rows");
        const auto held=tokens; auto embedding=native_layer0_embedding(embedding_,held); const size_t count=held.size();
        auto record=std::make_shared<PortableNativeLayer0Rows::Record>();record->position=first;record->tokens=held;
        auto result=std::shared_ptr<const PortableNativeLayer0Rows>(new PortableNativeLayer0Rows(epoch_,record));
        running_=true;struct Guard{bool& b;~Guard(){b=false;}} guard{running_};
        try {
            attention_->run_layer0_embedding(embedding,count,[&](const auto& h,uint32_t n){
                need(n==count,"layer0: attention rows");
                auto mixed=h.read(PortableNativeHcComposed::View::Field::mixed);
                auto residual=h.read(PortableNativeHcComposed::View::Field::residual);
                auto injection=h.read(PortableNativeHcComposed::View::Field::inject);
                input_.run(mixed,count);core_->stage(mixed,input_.output(0,count),input_.output(1,count),uint32_t(count));
                auto y=core_->output();output_.run(y,count);std::vector<float> block(count*2560);
                ggml_backend_tensor_get(output_.output(count),block.data(),0,block.size()*sizeof(float));
                ffn_hc_->run_post_attention(residual,block,injection,count,[&](const auto& f,uint32_t rows){
                    need(rows==count,"layer0: FFN rows");
                    auto applied=f.read(PortableNativeHcComposed::View::Field::residual);
                    auto inject=f.read(PortableNativeHcComposed::View::Field::inject);
                    auto activation=f.read(PortableNativeHcComposed::View::Field::mixed);
                    auto computed=ffn_.run(activation,count,provider);auto& b=banks_[count-1];
                    ggml_backend_tensor_set(b.residual,applied.data(),0,applied.size()*sizeof(float));
                    ggml_backend_tensor_set(b.block,computed.block.data(),0,computed.block.size()*sizeof(float));
                    ggml_backend_tensor_set(b.inject,inject.data(),0,inject.size()*sizeof(float));
                    strata_vk_native_hc_pending_run(write_.get(),b.residual,b.block,b.inject,b.candidate,b.gate,b.exp);
                    record->residual.resize(count*10240);ggml_backend_tensor_get(b.candidate,record->residual.data(),0,record->residual.size()*sizeof(float));
                    for(float x:record->residual)need(std::isfinite(x),"layer0: nonfinite final write");
                });
                need(core_->candidate()&&record->residual.size()==count*10240,"layer0: incomplete candidate");
            });
        }catch(...){record->valid=false;core_->invalidate();poisoned_=true;throw;}
        core_->publish_prepared();record->committed=true;position_+=count;return result;
    }
};
}
