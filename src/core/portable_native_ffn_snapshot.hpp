#pragma once
// Genuine post-FFN provenance. Only the FFN composition can issue snapshots.
#include "portable_native_hc_composed.hpp"
namespace strata::core::portable_detail {
class PortableNativeFfnComposition;
class PortableNativeFfnSnapshot {
    struct Epoch {bool valid=true; std::function<bool()> source_live;};
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
    bool live()const {
        return epoch_&&epoch_->valid&&record_&&record_->valid&&record_->committed&&
               epoch_->source_live&&epoch_->source_live();
    }
    bool valid_for(const PortableNativeFfnComposition& owner,int next_layer,uint64_t pos)const {
        return owner_==&owner&&next_layer==layer_+1&&pos==position_&&live();
    }
    int layer()const{return layer_;}
    uint64_t position()const{return position_;}
    const PortableHcPendingWrite& tuple()const {
        if(!epoch_||!epoch_->valid||!record_||!record_->valid)throw std::runtime_error("FFN snapshot: stale");
        return record_->tuple;
    }
};
inline void PortableNativeHcComposed::run(int layer,uint64_t first,const PortableNativeFfnComposition& producer,
    const std::vector<std::shared_ptr<const PortableNativeFfnSnapshot>>& snapshots,const Consumer& a,const Consumer& b){
    run_impl(layer,first,producer,snapshots,a,b);
}
}
