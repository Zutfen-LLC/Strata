#pragma once

// Internal borrowed-owner row chain, not a model loader or physical admission.
// Assembly accepts only an unstarted (position zero) chain; it never resets it.
// All owners, their role storage, provider dependencies and backends must
// outlive this object. They are exclusively borrowed: no independent calls,
// concurrent access or destruction while attached, including from callbacks.
// Separately admitted owners, backend allocations, role correctness and provider
// assembly remain the caller's responsibility. Operations are synchronous.
#include "portable_native_terminal_composition.hpp"
#include <array>
#include <limits>

namespace strata::core::portable_detail {
class PortableNativeModelRows {
public:
    struct Layer {
        PortableNativeOrdinaryGdnComposition* gdn = nullptr;
        PortableNativeQsaComposition* qsa = nullptr;
        PortableNativeFfnComposition* ffn = nullptr;
    };
    using Result = PortableNativeTerminalComposition::Result;
private:
    PortableNativeLayer0Composition& entry_;
    PortableNativePleComposition& ple_;
    PortableNativePleGdnComposition& ple_gdn_;
    PortableNativeFfnComposition& ffn1_;
    std::array<Layer, 48> layers_;
    PortableNativeQsaComposition& qsa47_;
    PortableNativeTerminalComposition& terminal_;
    uint64_t position_ = 0, entry_reset_ = 0, ple_generation_ = 0, ple_reset_ = 0;
    uint64_t ple_gdn_generation_ = 0, ffn1_reset_ = 0, terminal_reset_ = 0;
    std::array<uint64_t, 48> attention_generation_{}, ffn_reset_{}, rows_generation_{};
    bool running_ = false, poisoned_ = false;
    static void need(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
    void validate_assembly() const {
        need(ple_gdn_.ple_identity()==&ple_ && ffn1_.layer()==1 && ffn1_.ple_source()==&ple_gdn_ &&
             terminal_.source_identity()==&qsa47_ && qsa47_.layer()==47 &&
             entry_.vocabulary_size()==terminal_.vocabulary_size(), "model rows: entry/terminal binding");
        need(!layers_[0].gdn && !layers_[0].qsa && !layers_[0].ffn &&
             !layers_[1].gdn && !layers_[1].qsa && !layers_[1].ffn &&
             !layers_[47].gdn && !layers_[47].qsa && !layers_[47].ffn,
             "model rows: reserved layers");
        std::array<const void*,48> rows{}; rows[0]=entry_.rows_identity(); rows[1]=ffn1_.rows_identity(); rows[47]=terminal_.rows_identity();
        for (int i=2;i<=46;++i) {
            const auto& l=layers_[i];
            need(l.ffn && l.ffn->layer()==i, "model rows: missing/wrong FFN");
            if (i%4==3) need(l.qsa && !l.gdn && l.qsa->layer()==i && l.ffn->qsa_source()==l.qsa, "model rows: QSA binding");
            else need(l.gdn && !l.qsa && l.gdn->layer()==i && l.ffn->ordinary_source()==l.gdn, "model rows: GDN binding");
            rows[i]=l.ffn->rows_identity();
        }
        for (size_t i=0;i<rows.size();++i) for(size_t j=0;j<i;++j)
            need(rows[i] && rows[i]!=rows[j], "model rows: aliased FFN rows");
    }
    void capture() noexcept {
        entry_reset_=entry_.reset_generation(); ple_generation_=ple_.generation(); ple_reset_=ple_.reset_generation();
        rows_generation_[0]=entry_.rows_generation(); rows_generation_[1]=ffn1_.rows_generation(); rows_generation_[47]=terminal_.rows_generation();
        ple_gdn_generation_=ple_gdn_.generation(); ffn1_reset_=ffn1_.reset_generation(); terminal_reset_=terminal_.reset_generation();
        for(int i=2;i<=46;++i) {const auto& l=layers_[i];attention_generation_[i]=l.qsa?l.qsa->generation():l.gdn->generation();ffn_reset_[i]=l.ffn->reset_generation();rows_generation_[i]=l.ffn->rows_generation();}
        attention_generation_[47]=qsa47_.generation();
    }
    void validate_state(bool compare_epochs, uint64_t expected) const {
        need(entry_.available() && entry_.position()==expected && ple_.available() && ple_.position()==expected &&
             ple_gdn_.available() && ple_gdn_.position()==expected && ffn1_.available() && ffn1_.position()==expected &&
             qsa47_.available() && qsa47_.position()==expected && terminal_.available() && terminal_.position()==expected,
             "model rows: entry/terminal unavailable or position");
        if (!expected) need(ple_.previous()==std::array<int32_t,2>{-1,-1}, "model rows: initial PLE predecessor");
        if(compare_epochs) need(entry_.reset_generation()==entry_reset_ && ple_.generation()==ple_generation_ &&
             ple_.reset_generation()==ple_reset_ && ple_gdn_.generation()==ple_gdn_generation_ &&
             ffn1_.reset_generation()==ffn1_reset_ && terminal_.reset_generation()==terminal_reset_ &&
             qsa47_.generation()==attention_generation_[47] && entry_.rows_generation()==rows_generation_[0] &&
             ffn1_.rows_generation()==rows_generation_[1] && terminal_.rows_generation()==rows_generation_[47], "model rows: entry/terminal epoch changed");
        for(int i=2;i<=46;++i) {
            const auto& l=layers_[i];
            need(l.ffn->available() && l.ffn->position()==expected &&
                 (l.qsa ? l.qsa->available() && l.qsa->position()==expected : l.gdn->available() && l.gdn->position()==expected),
                 "model rows: layer unavailable or position");
            if(compare_epochs) need(l.ffn->reset_generation()==ffn_reset_[i] && l.ffn->rows_generation()==rows_generation_[i] &&
                 (l.qsa?l.qsa->generation():l.gdn->generation())==attention_generation_[i], "model rows: layer epoch changed");
        }
    }
public:
    // Covers this executor's token copy, two transient snapshot vectors and
    // the terminal's result payload allowance using its actual head vocabulary.
    // Component owners, backends, allocator overhead, provider and results
    // retained by callers are separately admitted/excluded.
    static size_t host_bound(const PortableNativeTerminalComposition& terminal) {
        const size_t v=terminal.vocabulary_size(), fixed=64u<<10;
        need(v <= (SIZE_MAX-fixed)/(2*sizeof(float)), "model rows: host bound overflow");
        return fixed+2*v*sizeof(float);
    }
    PortableNativeModelRows(PortableNativeLayer0Composition& entry,PortableNativePleComposition& ple,
        PortableNativePleGdnComposition& ple_gdn,PortableNativeFfnComposition& ffn1,
        const std::array<Layer,48>& layers,PortableNativeQsaComposition& qsa47,
        PortableNativeTerminalComposition& terminal,size_t host_budget)
      :entry_(entry),ple_(ple),ple_gdn_(ple_gdn),ffn1_(ffn1),layers_(layers),qsa47_(qsa47),terminal_(terminal),position_(entry.position()) {
        need(host_budget>=host_bound(terminal_), "model rows: host budget");
        need(position_==0, "model rows: assembly requires an unstarted chain");
        validate_assembly(); validate_state(false,position_); capture();
    }
    PortableNativeModelRows(const PortableNativeModelRows&)=delete;
    PortableNativeModelRows& operator=(const PortableNativeModelRows&)=delete;
    uint64_t position()const{return position_;}
    bool poisoned()const{return poisoned_;}
    std::shared_ptr<const Result> run(uint64_t first,const std::vector<int32_t>& tokens,const PortableExpertRowsBatch& provider) {
        need(!running_&&!poisoned_&&provider&&(tokens.size()==1||tokens.size()==2)&&first==position_&&
             first<=UINT64_MAX-tokens.size(), "model rows: request");
        for(auto token:tokens) need(token>=0 && size_t(token)<entry_.vocabulary_size(), "model rows: token");
        validate_state(true,position_);
        need(terminal_.can_run(tokens.size()), "model rows: terminal sequence exhausted");
        auto fits=[&](const PortableNativeQsaComposition& q){return first<=q.capacity() && tokens.size()<=q.capacity()-first;};
        need(fits(qsa47_), "model rows: terminal capacity");
        for(int i=3;i<=46;i+=4) need(fits(*layers_[i].qsa), "model rows: QSA capacity");
        const auto held=tokens;
        running_=true; struct Guard{bool& b;~Guard(){b=false;}} guard{running_};
        try {
            auto rows=entry_.run(first,held,provider);
            auto handle=ple_.stage(rows);
            PortableNativeFfnComposition::Batch previous;
            ffn1_.run(handle,provider,[&](const auto& batch){previous=batch;});
            auto* producer=&ffn1_;
            for(int i=2;i<=46;++i) {
                PortableNativeFfnComposition::Batch next;
                auto& layer=layers_[i];
                if(layer.qsa) layer.ffn->run(first,*producer,previous,provider,[&](const auto& batch){next=batch;});
                else layer.ffn->run_ordinary(first,*producer,previous,provider,[&](const auto& batch){next=batch;});
                previous=std::move(next); producer=layer.ffn;
            }
            auto result=terminal_.run(first,*producer,previous,provider);
            position_+=held.size(); capture();
            return result;
        } catch(...) {poisoned_=true;throw;}
    }
    // Earlier commits and provider side effects are not rolled back. Replay is
    // required after a failure. A failed reset leaves the whole chain poisoned.
    void reset() {
        need(!running_, "model rows: reentrant reset");
        running_=true; struct Guard{bool& b;~Guard(){b=false;}} guard{running_};
        poisoned_=true;
        entry_.reset(); ple_gdn_.reset(); ffn1_.reset();
        for(int i=2;i<=46;++i) {auto& l=layers_[i];if(l.qsa)l.qsa->reset();else l.gdn->reset();l.ffn->reset();}
        qsa47_.reset(); terminal_.reset();
        validate_assembly(); validate_state(false,0); position_=0; capture(); poisoned_=false;
    }
};
}
