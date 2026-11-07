#pragma once

// Standalone transaction boundary, not connected to model chunk admission. The
// executor must run the native PLE gather/projections/norm/gate/convolution for
// the entire supplied T, and fill bounded host materializations. It must
// not mutate committed model state. The executor is trusted native code: output
// spans prevent resizing, but raw pointer out-of-bounds access cannot be
// mechanically prevented. This class does not substitute arithmetic for those
// operations.
#include "portable_native_ple_host_bound.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace strata::core::portable_detail {
// allocate_shared's implementation-defined control block is bounded by this
// allocator; allocator metadata itself is outside the requested-byte budget.
template<class T> struct PleOwnerAllocator {
    using value_type = T;
    PleOwnerAllocator() = default;
    template<class U> PleOwnerAllocator(const PleOwnerAllocator<U>&) {}
    T* allocate(size_t n) {
        if (n > 512 / sizeof(T)) throw std::bad_alloc();
        return std::allocator<T>{}.allocate(n);
    }
    void deallocate(T* p, size_t n) noexcept { std::allocator<T>{}.deallocate(p, n); }
    template<class U> bool operator==(const PleOwnerAllocator<U>&) const { return true; }
    template<class U> bool operator!=(const PleOwnerAllocator<U>&) const { return false; }
};
class PortableNativePleTransaction {
public:
    struct Request {
        uint64_t position;
        std::vector<int32_t> tokens;
        std::vector<std::array<int32_t, 2>> predecessors; // oldest first
        std::vector<float> history; // channel-major, oldest first, 9/channel
        std::vector<float> residual_input; // T x residual_width, in token order
    };
    class Span {
        float* const data_;
        const size_t size_;
        friend class PortableNativePleTransaction;
        explicit Span(std::vector<float>& v) : data_(v.data()), size_(v.size()) {}
    public:
        size_t size() const { return size_; }
        float* data() const { return data_; }
        float* begin() const { return data_; }
        float* end() const { return data_ + size_; }
        float& operator[](size_t i) const { if (i >= size_) throw std::out_of_range("PLE output span"); return data_[i]; }
    };
    struct OutputView { const Span normalized, history, residual; };
    using Executor = std::function<void(const Request&, const OutputView&)>;
private:
    struct Output { std::vector<float> normalized, history, residual; };
public:
    class Candidate {
        friend class PortableNativePleTransaction;
        uint64_t generation_ = 0, position_ = 0;
        bool valid_ = false;
        const Request& request_;
        const Output& output_;
        public:
        Candidate(const Request& r, const Output& o) : request_(r), output_(o) {}
        Candidate(const Candidate&) = delete;
        Candidate& operator=(const Candidate&) = delete;
        uint64_t generation() const { return generation_; }
        uint64_t position() const { return position_; }
        const std::vector<int32_t>& tokens() const { return request_.tokens; }
        const std::vector<std::array<int32_t, 2>>& predecessors() const { return request_.predecessors; }
        const std::vector<float>& history() const { return output_.history; }
        const std::vector<float>& residual() const { return output_.residual; }
    };
    class Handle {
        friend class PortableNativePleTransaction;
        std::weak_ptr<const Candidate> p_;
        uint64_t generation_;
        explicit Handle(const std::shared_ptr<Candidate>& p) : p_(p), generation_(p->generation_) {}
        std::shared_ptr<const Candidate> checked() const {
            auto p = p_.lock();
            require(p && p->valid_ && p->generation_ == generation_, "PLE expired candidate");
            return p;
        }
    public:
        // Copies are caller-owned; no handle can prolong producer storage.
        std::vector<float> residual() const { auto p=checked(); return p->residual(); }
        std::vector<float> history() const { auto p=checked(); return p->history(); }
        std::vector<int32_t> tokens() const { auto p=checked(); return p->tokens(); }
        std::vector<std::array<int32_t,2>> predecessors() const { auto p=checked(); return p->predecessors(); }
        size_t token_count() const { auto p=checked(); return p->tokens().size(); }
        size_t residual_count() const { auto p=checked(); return p->residual().size(); }
        uint64_t position() const { auto p=checked(); return p->position(); }
        uint64_t generation() const { auto p=checked(); return p->generation(); }
    };
private:
    size_t channels_, width_;
    uint64_t position_ = 0, generation_ = 0;
    std::array<int32_t,2> previous_;
    std::vector<float> history_;
    Executor execute_;
    Request request_{};
    Output output_;
    std::shared_ptr<Candidate> owner_;
    bool pending_ = false;
    bool executing_ = false;
    static void require(bool b, const char* s) { if (!b) throw std::invalid_argument(s); }
    static bool finite(const std::vector<float>& v) { for (float x : v) if (!std::isfinite(x)) return false; return true; }
    void advance_generation() { if (generation_ == UINT64_MAX) throw std::overflow_error("PLE generation exhausted"); ++generation_; }
public:
    // Requested vector payloads at construction: three histories (27*C),
    // normalized output (2*C), two residual banks (4*W), and two tokens
    // and predecessor pairs. T=1 retains the full T=2 banks. Commit swaps histories.
    // Caller-owned handle copies and executor scratch are excluded. This is a
    // requested payload sum, including a 512-byte bounded owner/control block;
    // allocator metadata remains excluded.
    static constexpr size_t host_bound(size_t rows, size_t channels, size_t width) {
        if ((rows != 1 && rows != 2) || !channels || !width)
            throw std::invalid_argument("PLE dimensions/rows");
        // All transaction banks are allocated for two rows even for T=1.
        const size_t floats = ple_host_add(ple_host_mul(29, channels), ple_host_mul(4, width));
        return ple_host_add(512, ple_host_add(ple_host_mul(floats, sizeof(float)),
                            ple_host_mul(2, sizeof(int32_t) + sizeof(std::array<int32_t, 2>))));
    }
    // Upper envelope for transaction payloads plus the native table gather's
    // live embedding, index rows and one temporary decoded row. This deliberately
    // sums the decoded row and output even though their peaks do not coincide.
    // Excludes allocator metadata, caller-owned copies and other executor scratch.
    static constexpr size_t host_bound_with_gather(size_t rows, size_t channels, size_t width) {
        return ple_host_add(host_bound(rows, channels, width),
            ple_host_add(ple_host_mul(ple_host_add(ple_host_mul(rows, 2560), 2560), sizeof(float)),
                         ple_host_mul(rows, sizeof(std::array<uint32_t, 16>))));
    }
    // Admission includes the bounded allocate_shared owner/control block. Vector
    // capacity is checked after allocation; allocator metadata, executor scratch
    // and caller copies are excluded, so this is not a hard process memory limit.
    PortableNativePleTransaction(size_t channels, size_t width, size_t budget, Executor execute,
                                 std::array<int32_t,2> initial_previous = {-1,-1})
        : channels_(channels), width_(width), previous_(initial_previous), execute_(std::move(execute)) {
        require(channels && width && bool(execute_), "PLE dimensions/executor");
        require(host_bound(2, channels, width) <= budget, "PLE budget");
        history_ = std::vector<float>(ple_host_mul(9, channels), 0.0f);
        request_.tokens = std::vector<int32_t>(2);
        request_.predecessors = std::vector<std::array<int32_t,2>>(2);
        request_.history = std::vector<float>(ple_host_mul(9, channels));
        request_.residual_input = std::vector<float>(ple_host_mul(2, width));
        output_.normalized = std::vector<float>(ple_host_mul(2, channels));
        output_.history = std::vector<float>(ple_host_mul(9, channels));
        output_.residual = std::vector<float>(ple_host_mul(2, width));
        require(history_.capacity() == history_.size() && request_.tokens.capacity() == 2 &&
                request_.predecessors.capacity() == 2 && request_.history.capacity() == request_.history.size() &&
                request_.residual_input.capacity() == request_.residual_input.size() &&
                output_.normalized.capacity() == output_.normalized.size() &&
                output_.history.capacity() == output_.history.size() && output_.residual.capacity() == output_.residual.size(),
                "PLE bank capacity");
        owner_ = std::allocate_shared<Candidate>(PleOwnerAllocator<Candidate>{}, request_, output_);
    }
    PortableNativePleTransaction(const PortableNativePleTransaction&) = delete;
    PortableNativePleTransaction& operator=(const PortableNativePleTransaction&) = delete;
    Handle stage(uint64_t position, const std::vector<int32_t>& tokens, const std::vector<float>& residual_input) {
        require(!pending_ && !executing_, "PLE candidate already staged or executing");
        require(position == position_ && (tokens.size()==1 || tokens.size()==2), "PLE position/rows");
        require(position <= UINT64_MAX-tokens.size(), "PLE position overflow");
        for (auto t : tokens) require(t >= 0, "PLE token");
        const size_t residual_elements = ple_host_mul(tokens.size(), width_);
        // Check every output extent before constructing request copies or output spans.
        const size_t normalized_elements = ple_host_mul(tokens.size(), channels_);
        require(residual_input.size()==residual_elements && finite(residual_input), "PLE residual input");
        Request& r = request_;
        r.position = position;
        r.tokens.resize(tokens.size());
        r.predecessors.resize(tokens.size());
        r.residual_input.resize(residual_elements);
        std::copy(tokens.begin(), tokens.end(), r.tokens.begin());
        std::copy(history_.begin(), history_.end(), r.history.begin());
        std::copy(residual_input.begin(), residual_input.end(), r.residual_input.begin());
        auto prev = previous_;
        for (size_t i=0; i<tokens.size(); ++i) { r.predecessors[i]=prev; prev = {prev[1], tokens[i]}; }
        struct ExecutionGuard { bool& flag; explicit ExecutionGuard(bool& f) : flag(f) { flag=true; } ~ExecutionGuard() { flag=false; } } guard(executing_);
        const size_t n = tokens.size();
        Output& o = output_;
        o.normalized.resize(normalized_elements);
        o.residual.resize(residual_elements);
        const float nan = std::numeric_limits<float>::quiet_NaN();
        std::fill(o.normalized.begin(), o.normalized.end(), nan);
        std::fill(o.history.begin(), o.history.end(), nan);
        std::fill(o.residual.begin(), o.residual.end(), nan);
        execute_(r, OutputView{Span(o.normalized), Span(o.history), Span(o.residual)});
        require(finite(o.normalized) && finite(o.history) && finite(o.residual), "PLE nonfinite output");
        for (size_t c=0;c<channels_;++c) {
            for (size_t j=0;j<9-n;++j) require(o.history[9*c+j]==history_[9*c+j+n], "PLE history shift");
            for (size_t j=0;j<n;++j) require(o.history[9*c+9-n+j]==o.normalized[j*channels_+c], "PLE history append");
        }
        owner_->generation_ = generation_;
        owner_->position_ = r.position;
        owner_->valid_ = true;
        pending_ = true;
        return Handle(owner_);
    }
    // Validity must be checked again immediately before consumption.
    bool live(const Handle& h) const { auto p=h.p_.lock(); return p && pending_ && p==owner_ && p->valid_ && h.generation_==generation_ && p->generation_==generation_; }
    void commit(const Handle& h, bool downstream_accepted) {
        require(live(h), "PLE stale or foreign candidate");
        require(downstream_accepted, "PLE downstream did not accept");
        // Swap the already validated output bank into committed state.  This
        // avoids an unbudgeted allocation and cannot fail after generation moves.
        advance_generation();
        history_.swap(output_.history);
        for (auto t : request_.tokens) previous_ = {previous_[1],t};
        position_ += request_.tokens.size();
        pending_ = false; owner_->valid_ = false;
    }
    void abort(const Handle& h) { require(live(h), "PLE stale or foreign candidate"); advance_generation(); pending_ = false; owner_->valid_ = false; }
    void reset(std::array<int32_t,2> previous = {-1,-1}, uint64_t position = 0) {
        require(!executing_, "PLE execution in progress");
        advance_generation(); pending_ = false; owner_->valid_ = false; previous_=previous; position_=position;
        std::fill(history_.begin(), history_.end(), 0.0f);
    }
    uint64_t position() const { return position_; }
    const std::vector<float>& history() const { return history_; }
    std::array<int32_t,2> previous() const { return previous_; }
};
}
