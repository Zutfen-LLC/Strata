#pragma once
// Standalone post-attention -> FFN -> deferred HC tuples. Not model admission.
// All borrowed upstream/row owners and their backends must outlive this owner;
// they must not be operated independently while attached or during callbacks.
// Production construction selects this layer's named FFN HC roles and copies
// them into an independently owned bank. TestOnlyHc is a low-level test seam.
#include "portable_native_ple_gdn_composition.hpp"
#include "portable_native_qsa_composition.hpp"
#include "portable_native_ffn_rows.hpp"
#include "portable_native_ffn_hc_factory.hpp"
namespace strata::core::portable_detail {
class PortableNativeFfnComposition;
class PortableNativeFfnSnapshot {
    struct Epoch {bool valid=true;};
    struct Record {bool valid=true,committed=false; PortableHcPendingWrite tuple;};
    std::shared_ptr<Epoch> epoch_;
    std::shared_ptr<Record> record_;
    const PortableNativeFfnComposition* owner_;
    int layer_;uint64_t position_;
    PortableNativeFfnSnapshot(std::shared_ptr<Epoch> e,std::shared_ptr<Record> r,
                              const PortableNativeFfnComposition* o,int l,uint64_t p)
      :epoch_(std::move(e)),record_(std::move(r)),owner_(o),layer_(l),position_(p){}
    friend class PortableNativeFfnComposition;
public:
    bool valid_for(const PortableNativeFfnComposition& owner,int next_layer,uint64_t pos)const;
    const PortableHcPendingWrite& tuple()const {
        if(!epoch_||!epoch_->valid||!record_||!record_->valid)throw std::runtime_error("FFN snapshot: stale");
        return record_->tuple;
    }
};
class PortableNativeFfnComposition {
public:
    using Snapshot=PortableNativeFfnSnapshot;
    using Batch=std::vector<std::shared_ptr<const Snapshot>>;
    using Consumer=std::function<void(const Batch&)>;
private:
    using Epoch=Snapshot::Epoch;using Record=Snapshot::Record;
    PortableNativePleGdnComposition* gdn_=nullptr;PortableNativeQsaComposition* qsa_=nullptr;
    std::unique_ptr<PortableNativeHcComposed> hc_;
    PortableNativeFfnRows& rows_;
    int layer_;uint64_t position_,upstream_generation_;bool running_=false,poisoned_=false;
    std::shared_ptr<Epoch> epoch_=std::make_shared<Epoch>();
    Batch published_;
    static void need(bool b,const char*m){if(!b)throw std::runtime_error(m);}
    uint64_t upstream_position()const{return gdn_?gdn_->position():qsa_->position();}
    uint64_t upstream_generation()const{return gdn_?gdn_->generation():qsa_->generation();}
    bool upstream_poisoned()const{return gdn_?gdn_->poisoned():qsa_->poisoned();}
    bool upstream_provenance_live()const{return gdn_||qsa_->provenance_live();}
    template<class Invoke>
    void execute(uint64_t first,size_t count,const PortableExpertRowsBatch& provider,const Consumer& consumer,const Consumer& prepare,Invoke invoke){
        need(!running_&&!poisoned_&&!upstream_poisoned()&&upstream_provenance_live()&&!rows_.poisoned()&&consumer&&provider&&
             (count==1||count==2)&&first==position_&&first<=UINT64_MAX-count&&
             upstream_position()==first&&upstream_generation()==upstream_generation_&&upstream_generation_!=UINT64_MAX,
             "FFN composition: unavailable or source identity");
        running_=true;struct Guard{bool& b;~Guard(){b=false;}}guard{running_};
        Batch candidate;std::vector<std::shared_ptr<Record>> records;candidate.reserve(count);records.reserve(count);
        auto stage=[&](uint64_t p,uint64_t gen,const std::vector<float>& residual,const std::vector<float>& output,const std::vector<float>& injection){
            need(p==first&&gen==upstream_generation_&&upstream_position()==first&&upstream_generation()==gen,
                 "FFN composition: upstream candidate identity");
            hc_->run_post_attention(residual,output,injection,count,[&](const PortableNativeHcComposed::View& h,uint32_t n){
                need(n==count,"FFN composition: HC rows");auto applied=h.read(PortableNativeHcComposed::View::Field::residual);
                auto inject=h.read(PortableNativeHcComposed::View::Field::inject);auto mixed=h.read(PortableNativeHcComposed::View::Field::mixed);
                auto result=rows_.run(mixed,count,provider);
                need(applied.size()==count*10240&&inject.size()==count*4&&result.block.size()==count*2560,"FFN composition: output shape");
                for(size_t r=0;r<count;++r){auto rec=std::make_shared<Record>();
                    rec->tuple.residual.assign(applied.begin()+r*10240,applied.begin()+(r+1)*10240);
                    rec->tuple.block.assign(result.block.begin()+r*2560,result.block.begin()+(r+1)*2560);
                    rec->tuple.injection.assign(inject.begin()+r*4,inject.begin()+(r+1)*4);
                    records.push_back(rec);candidate.push_back(std::shared_ptr<const Snapshot>(new Snapshot(epoch_,rec,this,layer_,first+r)));
                }
                consumer(candidate);
            });
        };
        auto ready=[&](uint64_t p,uint64_t gen){
            need(p==first&&gen==upstream_generation_&&candidate.size()==count&&records.size()==count&&
                 upstream_position()==first&&upstream_generation()==gen,"FFN composition: preparation identity");
            if(prepare)prepare(candidate);
            need(upstream_position()==first&&upstream_generation()==gen,"FFN composition: source invalidated");
        };
        try{invoke(stage,ready);}catch(...){for(auto& r:records)r->valid=false;poisoned_=true;throw;}
        // Upstream has published. These operations allocate nothing and cannot throw.
        for(auto& r:records)r->committed=true;
        published_.swap(candidate);position_+=count;++upstream_generation_;
    }
public:
    // Budget covers two simultaneous T2 tuple payloads, record/control-block
    // allowance and transient HC copies. Caller-retained snapshots, row core,
    // upstream, backend buffers, allocator and driver overhead are excluded.
    static constexpr size_t host_bound(){return 1u<<20;}
    // Explicit low-level test seam; production construction must use named roles.
    struct TestOnlyHc { explicit TestOnlyHc()=default; };
    PortableNativeFfnComposition(TestOnlyHc,PortableNativePleGdnComposition& source,std::unique_ptr<PortableNativeHcComposed> hc,PortableNativeFfnRows& rows,size_t host_budget)
      :gdn_(&source),hc_(std::move(hc)),rows_(rows),layer_(1),position_(source.position()),upstream_generation_(source.generation()){
        need(hc_&&rows_.layer()==layer_&&host_budget>=host_bound()&&native_hc_boundary(layer_).defer_ffn&&!source.poisoned(),"FFN composition: admission");}
    PortableNativeFfnComposition(TestOnlyHc,PortableNativeQsaComposition& source,std::unique_ptr<PortableNativeHcComposed> hc,PortableNativeFfnRows& rows,size_t host_budget)
      :qsa_(&source),hc_(std::move(hc)),rows_(rows),layer_(source.layer()),position_(source.position()),upstream_generation_(source.generation()){
        need(hc_&&rows_.layer()==layer_&&host_budget>=host_bound()&&native_hc_boundary(layer_).defer_ffn&&!source.poisoned(),"FFN composition: admission");}
    template<class Source>
    PortableNativeFfnComposition(Source& source,ggml_backend_t backend,const std::vector<PortableTensorView>& views,
        PortableNativeFfnRows& rows,size_t device_budget,size_t host_budget,const char* prefix,
        std::optional<PortableNativeHcComposed::ComponentCaps> caps=std::nullopt)
      :PortableNativeFfnComposition(TestOnlyHc{},source,
        NativeFfnHcFactory::create(source_layer(source),backend,views,device_budget,
            host_budget>=host_bound()?host_budget-host_bound():0,prefix,{},caps),rows,host_budget){}
    template<class Source>
    PortableNativeFfnComposition(Source& source,ggml_backend_t backend,const GgufModel& model,
        PortableNativeFfnRows& rows,size_t device_budget,size_t host_budget,const char* prefix,
        std::optional<PortableNativeHcComposed::ComponentCaps> caps=std::nullopt)
      :PortableNativeFfnComposition(source,backend,NativeFfnHcFactory::from_gguf(model,source_layer(source)),rows,device_budget,host_budget,prefix,caps){}
private:
    static int source_layer(const PortableNativePleGdnComposition&){return 1;}
    static int source_layer(const PortableNativeQsaComposition& source){return source.layer();}
public:
    PortableNativeFfnComposition(const PortableNativeFfnComposition&)=delete;
    PortableNativeFfnComposition& operator=(const PortableNativeFfnComposition&)=delete;
    ~PortableNativeFfnComposition(){epoch_->valid=false;}
    bool poisoned()const{return poisoned_;}uint64_t position()const{return position_;}
    // Caller must reset/replay the borrowed upstream and provider first. This
    // operation does not reset recurrent state or undo provider side effects.
    void reset(){need(!running_&&!upstream_poisoned()&&upstream_provenance_live(),"FFN composition: active or invalid source");auto next=std::make_shared<Epoch>();hc_->reset();rows_.reset();epoch_->valid=false;epoch_.swap(next);published_.clear();position_=upstream_position();upstream_generation_=upstream_generation();poisoned_=false;}
    void run(const PortableNativePleGdnComposition::Handle& handle,const PortableExpertRowsBatch& provider,const Consumer& consumer,const Consumer& prepare={}){
        need(gdn_!=nullptr,"FFN composition: wrong source kind");auto first=handle.position();auto count=handle.tokens().size();
        execute(first,count,provider,consumer,prepare,[&](auto& stage,auto& ready){gdn_->run(handle,[&](const auto& v){stage(v.position(),v.generation(),v.residual(),v.output(),v.injection());},[&](const auto& v){ready(v.position(),v.generation());});});
    }
    void run(uint64_t first,const PortableGdnLayer& producer,const std::vector<std::shared_ptr<const PortableGdnSnapshot>>& snapshots,const PortableExpertRowsBatch& provider,const Consumer& consumer,const Consumer& prepare={}){
        need(qsa_!=nullptr,"FFN composition: wrong source kind");const size_t count=snapshots.size();
        execute(first,count,provider,consumer,prepare,[&](auto& stage,auto& ready){qsa_->run(first,producer,snapshots,[&](const auto& v){need(v.rows()==count,"FFN composition: source rows");stage(v.first(),v.generation(),v.residual(),v.output(),v.injection());},[&](const auto& v){need(v.rows()==count,"FFN composition: source rows");ready(v.first(),v.generation());});});
    }
    friend class PortableNativeFfnSnapshot;
};
inline bool PortableNativeFfnSnapshot::valid_for(const PortableNativeFfnComposition& owner,int next_layer,uint64_t pos)const{
    return owner_==&owner&&!owner.running_&&epoch_&&epoch_->valid&&epoch_==owner.epoch_&&record_&&record_->valid&&record_->committed&&
           owner.upstream_provenance_live()&&owner.upstream_position()==owner.position_&&owner.upstream_generation()==owner.upstream_generation_&&
           next_layer==layer_+1&&pos==position_;
}
inline void PortableNativeHcComposed::run(int layer,uint64_t first,const PortableNativeFfnComposition& producer,
    const std::vector<std::shared_ptr<const PortableNativeFfnSnapshot>>& snapshots,const Consumer& a,const Consumer& b){
    run_impl(layer,first,producer,snapshots,a,b);
}
}
