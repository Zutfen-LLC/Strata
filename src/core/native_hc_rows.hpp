#pragma once
#include "native_hc_boundary.hpp"
#include "strata/core/portable_ffn.hpp"
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace strata::core::portable_detail {
// Host-owned staging prerequisite only: no device execution or model admission.
// One instance holds one layer boundary, one sequence, and at most two rows.
// Returned references expire on begin, abort, publish, reset or destruction.
class NativeHcRows {
public:
    struct Row {
        uint64_t position;
        int64_t token;
        PortableHcPendingWrite tuple; // block/injection empty for materialized input
    };
    void begin(int layer, uint64_t first, size_t count) {
        if (active_ || (count != 1 && count != 2) || first > UINT64_MAX-count ||
            (has_next_ && (first != next_ || layer != layer_))) throw std::runtime_error("native HC rows: invalid begin");
        const auto boundary = native_hc_boundary(layer);
        pending_ = boundary.pending_attention;
        layer_ = layer; first_ = first; count_ = count; filled_ = 0; active_ = true;
        candidate_ = {};
    }
    // The caller supplies actual producer identities and values. Copy before any
    // producer scratch reuse; never infer a tuple from a materialized residual.
    void stage(size_t index, uint64_t position, int64_t token, const PortableHcPendingWrite& value) {
        if (!active_ || index != filled_ || index >= count_ || position != first_+index ||
            value.residual.size()!=10240 ||
            (pending_ ? (value.block.size()!=2560 || value.injection.size()!=4)
                      : (!value.block.empty() || !value.injection.empty())))
            throw std::runtime_error("native HC rows: invalid row or boundary tuple");
        Row copy{position, token, value};
        candidate_[index] = std::move(copy);
        ++filled_;
    }
    // Publish only after the caller's entire operation succeeds. Failure must
    // call abort; no partially staged row becomes visible as committed state.
    void publish() {
        if (!active_ || filled_ != count_) throw std::runtime_error("native HC rows: incomplete publication");
        committed_.swap(candidate_);
        committed_count_ = count_; next_ = first_+count_; has_next_ = true;
        candidate_ = {}; active_ = false;
    }
    void abort() noexcept { candidate_ = {}; active_ = false; filled_ = 0; }
    void reset() noexcept { abort(); committed_ = {}; committed_count_ = 0; has_next_ = false; }
    size_t size() const noexcept { return committed_count_; }
    const Row& row(size_t index) const {
        if (index >= committed_count_) throw std::out_of_range("native HC rows: unpublished row");
        return committed_[index];
    }
private:
    std::array<Row,2> candidate_{}, committed_{};
    uint64_t first_=0, next_=0;
    size_t count_=0, filled_=0, committed_count_=0;
    int layer_=-1;
    bool active_=false, pending_=false, has_next_=false;
};
}
