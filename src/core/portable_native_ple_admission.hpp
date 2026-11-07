#pragma once

// Standalone composition calculator, not model admission.
#include "portable_native_ple_executor.hpp"
#include "portable_native_hc_composed.hpp"
#include "native_component_host_bound.hpp"
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

namespace strata::core::portable_detail {
class PortableNativePleAdmission {
public:
    struct Breakdown {
        const size_t executor_device, executor_host, transaction_host, hc_device, hc_host, component_host, estimate_host;
        const size_t total;

        constexpr Breakdown(size_t ed, size_t eh, size_t th, size_t hd, size_t hh, size_t ch = 0, size_t estimate = 0)
            : executor_device(ed), executor_host(eh), transaction_host(th),
              hc_device(hd), hc_host(hh), component_host(ch), estimate_host(estimate),
              total(ple_host_add(ple_host_add(ple_host_add(ple_host_add(ple_host_add(ple_host_add(ed, eh), th), hd), hh), ch), estimate)) {}
    };

    // WARNING: not a complete production bound: allocator and external backend
    // and shader helper internals remain unbounded. Do not treat it as one.
    // T2 upper envelope: executor's mixed owner estimates already contain
    // metadata, so they are not purely device bytes. Its host total deliberately
    // overcounts metadata and diagnostic lifetimes; transaction gather is already
    // covered by the executor. HC includes both publication slots and role copy.
    // Excludes borrowed source roles/table, caller copies, allocator overhead,
    // backend/shader internals, executor path strings and other caller captures.
    // ggml estimate APIs allocate temporary metadata contexts (not backend
    // buffers); this calculation performs no owner construction or dispatch.
    // Conservative simultaneous sum, including transient factory payloads even
    // where their lifetimes do not overlap. Backend/allocator internals excluded.
    static size_t component_host_bound(const std::string& directory, size_t prefix_bytes) {
        if (directory.size() > PortableNativePleExecutor::max_path_bytes ||
            prefix_bytes > PortableNativePleExecutor::max_path_bytes)
            throw std::length_error("PLE composition: path too long");
        size_t sum = 0;
        const size_t root = ple_host_add(directory.size(),
            !directory.empty() && directory.back() != '/' ? 1 : 0);
        auto add = [&](NativeComponentHostKind kind, size_t path) {
            sum = ple_host_add(sum, native_component_factory_host_bound(kind, path));
        };
        add(NativeComponentHostKind::ple_key_quant, ple_host_add(root, sizeof("key/quant-effective.spv")-1));
        add(NativeComponentHostKind::ple_key_mmvq, ple_host_add(root, sizeof("key/dot-effective.spv")-1));
        for (size_t i = 0; i < 2; ++i) {
            add(NativeComponentHostKind::ple_value, ple_host_add(root, sizeof("t1/value/effective.spv")-1));
            add(NativeComponentHostKind::ple_gate, ple_host_add(root, sizeof("t1/gate/effective.spv")-1));
            add(NativeComponentHostKind::ple_norm, ple_host_add(root, sizeof("t1/norm.spv")-1));
            add(NativeComponentHostKind::ple_conv, ple_host_add(root, sizeof("t1/conv.spv")-1));
        }
        add(NativeComponentHostKind::hc_pending, ple_host_add(prefix_bytes, sizeof(".pending.spv")-1));
        add(NativeComponentHostKind::hc, ple_host_add(prefix_bytes, sizeof(".norm.spv")-1));
        add(NativeComponentHostKind::hc_mix, ple_host_add(prefix_bytes, sizeof(".mix.spv")-1));
        return sum;
    }
    // Source-owned path payloads (including terminators), summed across phases.
    // The executor keeps roles_.directory and four value/gate directories.
    // Construction: root, row dir, norm and conv strings; key argument,
    // key filesystem join and string conversion; value/gate argument and
    // filesystem join/conversion; module join and conversion. The joins have
    // a source and result simultaneously, so reserve both. HC's prefix and
    // shader temporary are already in hc_host diagnostics.
    // Receipt helpers are not called by composition::stage, but reserve their
    // first-candidate directory and longest file join as well. Streams and
    // component factory payloads are accounted elsewhere/excluded.
    static size_t executor_path_bound(const std::string& directory) {
        if (directory.size() > PortableNativePleExecutor::max_path_bytes)
            throw std::length_error("PLE composition: path too long");
        const size_t root = ple_host_add(directory.size(),
            !directory.empty() && directory.back() != '/' ? 1 : 0);
        auto path = [&](size_t suffix) { return ple_host_add(ple_host_add(root, suffix), 1); };
        size_t sum = ple_host_add(directory.size(), 1); // roles_ copy
        auto add = [&](size_t n, size_t suffix) { sum = ple_host_add(sum, ple_host_mul(n, path(suffix))); };
        add(4, sizeof("t1/value")-1); // retained value/gate directories
        add(1, 0);                    // create_set root
        add(1, sizeof("t1")-1);       // row directory
        add(2, sizeof("t1/conv.spv")-1); // norm and conv strings
        add(1, sizeof("key")-1);      // key constructor argument
        add(2, sizeof("key/quant-effective.spv")-1); // join and string
        add(2, sizeof("t1/value")-1); // argument join and string
        add(2, sizeof("t1/value/effective.spv")-1); // factory join source/result
        add(2, sizeof("t1/conv.spv")-1); // module join and string
        add(1, sizeof("t1/value/first-candidate")-1);
        add(2, sizeof("t1/value/first-candidate/diagnostic.f32le")-1); // receipt join source/result
        return sum;
    }
    // Estimates hold at most two contexts concurrently: value, gate, norm,
    // conv and HC use pairs; gated-value uses one 1 MiB arena. All other
    // estimate arenas are <= 1 MiB. Two largest arenas is a conservative
    // envelope, also reserved when estimates are nested under live owners.
    // ggml_init separately allocates its opaque context bookkeeping; like
    // allocator/backend internals it is outside this repository's payload contract.
    static constexpr size_t estimate_host_bound() { return 2 * 1024 * 1024; }

    template<class EstimateFactory>
    static auto preflight(size_t budget, EstimateFactory&& estimate) {
        if (budget < estimate_host_bound()) throw std::length_error("PLE composition preflight budget exceeded");
        auto bounds = std::invoke(std::forward<EstimateFactory>(estimate));
        require_budget(bounds, budget);
        return bounds;
    }
    static Breakdown calculate(ggml_backend_t backend, const std::array<const ggml_tensor*,4>& hc_roles,
                               const std::string& directory, size_t hc_prefix_bytes,
                               PortableNativeHcComposed::ComponentCaps caps) {
        const size_t executor_path_bytes = directory.size();
        if (executor_path_bytes > PortableNativePleExecutor::max_path_bytes ||
            hc_prefix_bytes > PortableNativePleExecutor::max_path_bytes)
            throw std::length_error("PLE composition: path too long");
        const size_t executor_paths = executor_path_bound(directory);
        const size_t component_total = ple_host_add(ple_host_add(caps.pending, caps.norm), caps.mix);
        const auto e = PortableNativePleExecutor::estimate_breakdown(backend);
        return Breakdown(e.owner_estimates, ple_host_add(e.total-e.owner_estimates, executor_paths),
                         PortableNativePleTransaction::host_bound(2,10240,10240),
                         ple_host_add(PortableNativeHcComposed::tensor_bound(backend, hc_roles), component_total),
                         PortableNativeHcComposed::host_bound(2,hc_prefix_bytes),
                         component_host_bound(directory, hc_prefix_bytes), estimate_host_bound());
    }
    static void require_budget(const Breakdown& bounds, size_t budget) {
        if (bounds.total > budget) throw std::length_error("PLE composition budget exceeded");
    }

    // Budget covers the sum of device and host bounds. No factory is invoked
    // on overflow or budget rejection. Factory exceptions propagate; already
    // constructed owners are destroyed normally.
    template<class ExecutorFactory, class TransactionFactory, class HcFactory>
    static auto create(size_t ed, size_t eh, size_t th, size_t hd, size_t hh,
                       size_t budget, ExecutorFactory&& executor,
                       TransactionFactory&& transaction, HcFactory&& hc) {
        const Breakdown bounds(ed, eh, th, hd, hh);
        require_budget(bounds, budget);
        using Owners = std::tuple<std::decay_t<std::invoke_result_t<ExecutorFactory>>,
                                  std::decay_t<std::invoke_result_t<TransactionFactory>>,
                                  std::decay_t<std::invoke_result_t<HcFactory>>>;
        return Owners{std::invoke(std::forward<ExecutorFactory>(executor)),
                      std::invoke(std::forward<TransactionFactory>(transaction)),
                      std::invoke(std::forward<HcFactory>(hc))};
    }
};
}
