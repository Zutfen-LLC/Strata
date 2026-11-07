#pragma once

// Standalone PLE -> HC boundary, not model chunk admission. The backend and
// executor roles (including their storage and table) are borrowed and must
// outlive this object. HC source tensors need only survive construction.
#include "portable_native_ple_admission.hpp"
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
        return transaction_->stage(position, tokens, residual);
    }
    void commit_to_hc(const Handle& handle, const Consumer& first, const Consumer& second) {
        if (consuming_) throw std::runtime_error("PLE composition: reentrant consume");
        consuming_ = true;
        struct Guard { bool& flag; ~Guard() { flag = false; } } guard{consuming_};
        hc_->run(1, handle.position(), *transaction_, handle, first, second);
    }
    void abort(const Handle& handle) {
        if (consuming_) throw std::runtime_error("PLE composition: reentrant abort");
        transaction_->abort(handle);
    }
    void reset(std::array<int32_t, 2> previous = {-1, -1}, uint64_t position = 0) {
        // Reject reentrant changes before either participant can be changed.
        if (consuming_) throw std::runtime_error("PLE composition: reentrant reset");
        hc_->reset();
        transaction_->reset(previous, position);
    }
    uint64_t position() const { return transaction_->position(); }
    const std::vector<float>& history() const { return transaction_->history(); }
    std::array<int32_t, 2> previous() const { return transaction_->previous(); }
};
}
