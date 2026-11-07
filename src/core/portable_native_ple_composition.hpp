#pragma once

// Standalone PLE -> HC boundary, not model chunk admission. The backend and
// executor roles (including their storage and table) are borrowed and must
// outlive this object. HC source tensors need only survive construction.
#include "portable_native_ple_admission.hpp"
#include "portable_native_layer0_composition.hpp"
#include <memory>
#include <string>

namespace strata::core::portable_detail {
class PortableNativePleComposition {
public:
    using Handle = PortableNativePleTransaction::Handle;
    using Consumer = PortableNativeHcComposed::Consumer;
    using ComponentCaps = PortableNativeHcComposed::ComponentCaps;

private:
    // Reverse destruction order keeps the closure's executor and the producer
    // alive until after HC and the transaction have been destroyed.
    std::unique_ptr<PortableNativePleExecutor> executor_;
    std::unique_ptr<PortableNativePleTransaction> transaction_;
    std::unique_ptr<PortableNativeHcComposed> hc_;
    bool consuming_ = false;
    uint64_t reset_generation_ = 0;
    std::shared_ptr<const PortableNativeLayer0Rows> staged_source_, predecessor_;
    uint64_t staged_generation_=0;

public:
    // The budget covers the admission payload bounds plus all three explicit
    // component caps and component factory host bounds. As with the underlying bounds it excludes allocator and
    // backend/shader internals, borrowed roles, caller copies and captures.
    PortableNativePleComposition(const PortableNativePleExecutor::Roles& roles,
                                 const std::array<const ggml_tensor*, 4>& hc_roles,
                                 ggml_backend_t backend, ComponentCaps caps,
                                 size_t budget, const std::string& hc_receipt_prefix) {
        PortableNativePleExecutor::validate_roles(roles);
        if (!backend || roles.backend != backend || hc_receipt_prefix.empty())
            throw std::invalid_argument("PLE composition: backend or receipt prefix");
        const auto bounds = PortableNativePleAdmission::preflight(budget, [&] {
            return PortableNativePleAdmission::calculate(backend, hc_roles, roles.directory, hc_receipt_prefix.size(), caps);
        });

        executor_ = std::make_unique<PortableNativePleExecutor>(roles,
            ple_host_add(bounds.executor_device, bounds.executor_host));
        transaction_ = std::make_unique<PortableNativePleTransaction>(10240, 10240, bounds.transaction_host,
            [executor = executor_.get()](const PortableNativePleTransaction::Request& request,
                                         const PortableNativePleTransaction::OutputView& output) {
                (*executor)(request, output);
            });
        hc_ = std::make_unique<PortableNativeHcComposed>(backend, hc_roles[0], hc_roles[1], hc_roles[2], hc_roles[3],
            bounds.hc_device, hc_receipt_prefix.c_str(), PortableNativeHcComposed::Dispatch{}, bounds.hc_host, caps);
    }
    PortableNativePleComposition(const PortableNativePleComposition&) = delete;
    PortableNativePleComposition& operator=(const PortableNativePleComposition&) = delete;

    Handle stage(uint64_t position, const std::vector<int32_t>& tokens, const std::vector<float>& residual) {
        if (consuming_) throw std::runtime_error("PLE composition: reentrant stage");
        auto handle=transaction_->stage(position, tokens, residual);
        staged_source_.reset();
        return handle;
    }
    // Typed entry binds the exact committed layer0 token rows to PLE history.
    Handle stage(const std::shared_ptr<const PortableNativeLayer0Rows>& source) {
        if (consuming_ || !source || !source->live()) throw std::runtime_error("PLE composition: invalid layer0 source");
        const auto first=source->position(); auto tokens=source->tokens(); auto residual=source->residual();
        if (first!=transaction_->position() || (!first && transaction_->previous()!=std::array<int32_t,2>{-1,-1}) ||
            (first && (!predecessor_ || !predecessor_->live() || !source->same_source(*predecessor_) ||
            predecessor_->position()+predecessor_->tokens().size()!=first)))
            throw std::runtime_error("PLE composition: layer0 history provenance");
        auto handle=transaction_->stage(first,tokens,residual);
        if (!source->live() || (first && !predecessor_->live())) {
            transaction_->abort(handle);
            throw std::runtime_error("PLE composition: layer0 invalidated during stage");
        }
        staged_source_=source; staged_generation_=handle.generation();
        return handle;
    }
    void commit_to_hc(const Handle& handle, const Consumer& first, const Consumer& second) {
        if (consuming_) throw std::runtime_error("PLE composition: reentrant consume");
        consuming_ = true;
        struct Guard { bool& flag; ~Guard() { flag = false; } } guard{consuming_};
        auto source=staged_source_;
        if(source && (handle.generation()!=staged_generation_ || !source->live()))
            throw std::runtime_error("PLE composition: stale layer0 source");
        hc_->run(1, handle.position(), *transaction_, handle, first, [&](const auto& view,uint32_t rows) {
            second(view,rows);
            if(source && (!source->live() || source->position()!=handle.position() ||
               source->tokens()!=handle.tokens() || (handle.position() && (!predecessor_ || !predecessor_->live() || !source->same_source(*predecessor_)))))
                throw std::runtime_error("PLE composition: layer0 source invalidated before commit");
        });
        predecessor_=std::move(source); staged_source_.reset();
    }
    void abort(const Handle& handle) {
        if (consuming_) throw std::runtime_error("PLE composition: reentrant abort");
        transaction_->abort(handle);
        staged_source_.reset();
    }
    void reset(std::array<int32_t, 2> previous = {-1, -1}, uint64_t position = 0) {
        // Reject reentrant changes before either participant can be changed.
        if (consuming_ || reset_generation_ == UINT64_MAX) throw std::runtime_error("PLE composition: reentrant reset or generation exhausted");
        ++reset_generation_;
        hc_->reset();
        transaction_->reset(previous, position);
        staged_source_.reset(); predecessor_.reset();
    }
    // Raw/legacy commits have no typed predecessor and retain their historical behavior.
    bool provenance_live() const { return !predecessor_ || predecessor_->live(); }
    uint64_t position() const { return transaction_->position(); }
    uint64_t reset_generation() const { return reset_generation_; }
    uint64_t generation() const { return transaction_->generation(); }
    bool available() const { return !consuming_ && transaction_->available() && provenance_live(); }
    const std::vector<float>& history() const { return transaction_->history(); }
    std::array<int32_t, 2> previous() const { return transaction_->previous(); }
};
}
