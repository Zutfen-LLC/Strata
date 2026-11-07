#include "../../src/core/portable_native_ple_admission.hpp"
#include <cassert>
#include <limits>
#include <memory>
#include <stdexcept>
#include <tuple>
#include "ggml-cpu.h"
#include "native_component_bound_fake.hpp"

using strata::core::portable_detail::PortableNativePleAdmission;
using Admission = PortableNativePleAdmission;
constexpr Admission::Breakdown bounds(1, 2, 3, 4, 5);
static_assert(bounds.executor_device == 1 && bounds.executor_host == 2 &&
              bounds.transaction_host == 3 && bounds.hc_device == 4 &&
              bounds.hc_host == 5 && bounds.total == 15);

int main() {
    auto backend = ggml_backend_cpu_init();
    assert(backend);
    using namespace strata::core::portable_detail;
    auto ctx = ggml_init({8192, nullptr, true});
    assert(ctx);
    std::array<const ggml_tensor*,4> roles{
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, 4),
        ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 10240, 2),
        ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 10240, 160),
        ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 320, 5120)};
    const auto e = PortableNativePleExecutor::estimate_breakdown(backend);
    const PortableNativeHcComposed::ComponentCaps caps{113, 227, 331};
    using K = NativeComponentHostKind;
    bound_calls.clear();
    auto component = Admission::component_host_bound("abc/", 7);
    const std::vector<std::pair<K, size_t>> expected{
        {K::ple_key_quant, 4+23}, {K::ple_key_mmvq, 4+21},
        {K::ple_value, 4+22}, {K::ple_gate, 4+21}, {K::ple_norm, 4+11}, {K::ple_conv, 4+11},
        {K::ple_value, 4+22}, {K::ple_gate, 4+21}, {K::ple_norm, 4+11}, {K::ple_conv, 4+11},
        {K::hc_pending, 7+12}, {K::hc, 7+9}, {K::hc_mix, 7+8}};
    assert(bound_calls == expected);
    size_t expected_sum = 0;
    for (auto [kind, path] : expected) expected_sum += 100 * (1 + static_cast<size_t>(kind)) + path;
    assert(component == expected_sum);
    bound_calls.clear();
    assert(Admission::component_host_bound("abc", 7) == component);
    assert(bound_calls == expected);
    bound_overflow = true;
    bool component_overflow = false;
    try { (void)Admission::component_host_bound("abc", 7); } catch (const std::overflow_error&) { component_overflow = true; }
    bound_overflow = false;
    assert(component_overflow);
    const auto calculated = Admission::calculate(backend, roles, std::string(23, 'x'), 17, caps);
    assert(calculated.executor_device == e.owner_estimates);
    // Independent expansion: root is 23 bytes plus separator; each payload
    // includes its terminating byte. Covers retained, construction and receipt paths.
    const size_t path_sum = 24 + 4*(24+8+1) + (24+1) + (24+2+1)
        + 2*(24+11+1) + (24+3+1) + 2*(24+23+1)
        + 2*(24+8+1) + 2*(24+22+1) + 2*(24+11+1)
        + (24+24+1) + 2*(24+41+1);
    assert(Admission::executor_path_bound(std::string(23, 'x')) == path_sum);
    assert(calculated.executor_host == e.total - e.owner_estimates + path_sum);
    assert(Admission::executor_path_bound("abc") + 1 == Admission::executor_path_bound("abc/"));
    bool path_rejected = false;
    try { (void)Admission::executor_path_bound(std::string(PortableNativePleExecutor::max_path_bytes + 1, 'x')); }
    catch (const std::length_error&) { path_rejected = true; }
    assert(path_rejected);
    assert(Admission::executor_path_bound(std::string(PortableNativePleExecutor::max_path_bytes, 'x')) > 0);
    bound_calls.clear();
    bool prefix_rejected = false;
    try { (void)Admission::component_host_bound("abc", PortableNativePleExecutor::max_path_bytes + 1); }
    catch (const std::length_error&) { prefix_rejected = true; }
    assert(prefix_rejected && bound_calls.empty());
    // Independent expansion of the transaction's payload accounting for T2.
    assert(calculated.transaction_host == 512 + (29*10240 + 4*10240)*sizeof(float) + 2*(sizeof(int32_t)+sizeof(std::array<int32_t,2>)));
    PortableNativePleTransaction tx(10240, 10240, calculated.transaction_host,
        [](const PortableNativePleTransaction::Request&, const PortableNativePleTransaction::OutputView&) {});
    assert(calculated.hc_device == PortableNativeHcComposed::tensor_bound(backend, roles) + 113 + 227 + 331);
    assert(calculated.hc_host == PortableNativeHcComposed::host_bound(2,17));
    assert(calculated.estimate_host == 2u * 1024 * 1024);
    assert(calculated.total == calculated.executor_device + calculated.executor_host + calculated.transaction_host + calculated.hc_device + calculated.hc_host + calculated.component_host + calculated.estimate_host);
    int estimates = 0;
    auto estimate = [&] { ++estimates; return calculated; };
    bool pre_rejected = false;
    try { (void)Admission::preflight(Admission::estimate_host_bound()-1, estimate); }
    catch (const std::length_error&) { pre_rejected = true; }
    assert(pre_rejected && estimates == 0);
    pre_rejected = false;
    try { (void)Admission::preflight(Admission::estimate_host_bound(), estimate); }
    catch (const std::length_error&) { pre_rejected = true; }
    assert(pre_rejected && estimates == 1);
    assert(Admission::preflight(calculated.total, estimate).total == calculated.total && estimates == 2);
    Admission::require_budget(calculated, calculated.total);
    bool short_budget = false;
    try { Admission::require_budget(calculated, calculated.total-1); } catch (const std::length_error&) { short_budget = true; }
    assert(short_budget);
    bool prefix_overflow = false;
    try { (void)Admission::calculate(backend, roles, std::string(23, 'x'), SIZE_MAX, caps); } catch (const std::length_error&) { prefix_overflow = true; }
    assert(prefix_overflow);
    bool path_overflow = false;
    try { (void)Admission::calculate(backend, roles, std::string(PortableNativePleExecutor::max_path_bytes + 1, 'x'), 17, caps); } catch (const std::length_error&) { path_overflow = true; }
    assert(path_overflow);
    bool caps_overflow = false;
    try { (void)Admission::calculate(backend, roles, std::string(23, 'x'), 17, {SIZE_MAX, 1, 0}); } catch (const std::overflow_error&) { caps_overflow = true; }
    assert(caps_overflow);
    ggml_free(ctx);
    ggml_backend_free(backend);
    int a = 0, b = 0, c = 0;
    auto executor = [&] { ++a; return std::make_unique<int>(1); };
    auto transaction = [&] { ++b; return std::make_unique<int>(2); };
    auto hc = [&] { ++c; return std::make_unique<int>(3); };
    int estimate_calls = 0;
    auto gated_create = [&](size_t budget) {
        auto bnd = Admission::preflight(budget, [&] {
            ++estimate_calls;
            return Admission::Breakdown(1, 2, 3, 4, 5, 0, Admission::estimate_host_bound());
        });
        return Admission::create(bnd.executor_device, bnd.executor_host, bnd.transaction_host,
                                 bnd.hc_device, bnd.hc_host, budget, executor, transaction, hc);
    };
    bool rejected = false;
    try { (void)gated_create(Admission::estimate_host_bound()-1); }
    catch (const std::length_error&) { rejected = true; }
    assert(rejected && estimate_calls == 0 && a == 0 && b == 0 && c == 0);
    rejected = false;
    try { (void)Admission::create(1, 2, 3, 4, 5, 14, executor, transaction, hc); }
    catch (const std::length_error&) { rejected = true; }
    assert(rejected && a == 0 && b == 0 && c == 0);
    rejected = false;
    try { (void)Admission::create(std::numeric_limits<size_t>::max(), 0, 0, 0, 1,
                                  std::numeric_limits<size_t>::max(), executor, transaction, hc); }
    catch (const std::overflow_error&) { rejected = true; }
    assert(rejected && a == 0 && b == 0 && c == 0);
    auto owners = Admission::create(1, 2, 3, 4, 5, 15, executor, transaction, hc);
    assert(a == 1 && b == 1 && c == 1);
    assert(*std::get<0>(owners) == 1 && *std::get<1>(owners) == 2 && *std::get<2>(owners) == 3);
}
