#pragma once
#include "portable_gdn_fixture.hpp"
#include <limits>

namespace gdn_fixture {
using namespace strata::core;
// A borrowed CPU backend exercises resident ownership without GPU execution;
// the same tests also run in the explicit Vulkan fixture when dispatched.
inline void production_policy(const Fixture &f, PortableGdnOptions options, int64_t layer = 0) {
    std::string error;
    options.policy = PortableGdnPolicy::eager_diagnostic;
    auto diagnostic = PortableGdnLayer::create(layer, f.views, options, error);
    check(bool(diagnostic), error);
    options.policy = PortableGdnPolicy::device_production;
    auto production = PortableGdnLayer::create(layer, f.views, options, error);
    check(bool(production), error);
    check(production->policy() == PortableGdnPolicy::device_production && !production->poisoned(), "production policy");
    check(production->state_bytes() < diagnostic->state_bytes() &&
          production->memory_usage().device_state == diagnostic->memory_usage().device_state &&
          production->memory_usage().host_extra_bound < diagnostic->memory_usage().host_extra_bound,
          "production host mirrors removed; device ledger unchanged");
    for (int observer = 0; observer < 3; ++observer) {
        bool refused = false;
        try {
            if (observer == 0) (void)production->recurrent_state();
            if (observer == 1) (void)production->convolution_history();
            if (observer == 2) (void)production->ple_convolution_history();
        } catch (const std::runtime_error &e) {
            refused = std::string(e.what()).find("inspection requires eager diagnostic") != std::string::npos;
        }
        check(refused, "production must never expose stale tensor mirrors");
    }
    const bool ple = bool(options.ple_table);
    const PortableExpertRows provider_ok = [](int64_t, const PortableRoute &route, std::vector<float> &rows, std::string &err) {
        return experts()(0, route, rows, err);
    };
    const int32_t tokens[] = {17, 248044, 23, 248056, 31, 37};
    PortableGdnWork prod_work, diag_work;
    std::vector<float> first;
    for (size_t t = 0; t < std::size(tokens); ++t) {
        PortableRoute a, b;
        check(diagnostic->step(tokens[t], f.input(int(t)), provider_ok, a, error, nullptr, &diag_work), error);
        check(production->step(tokens[t], f.input(int(t)), provider_ok, b, error, nullptr, &prod_work), error);
        check(a.ids == b.ids && a.weights == b.weights && a.activation == b.activation,
              "production routing bytes differ on same backend");
        check(diagnostic->last_residual() == production->last_residual() &&
              diagnostic->token_history() == production->token_history() && production->tokens() == t + 1,
              "production arithmetic/history bytes differ on same backend");
        if (!t) first = production->last_residual();
    }
    check(prod_work.candidate_readback_calls == std::size(tokens) &&
          prod_work.candidate_finite_calls == std::size(tokens) &&
          diag_work.candidate_readback_calls == std::size(tokens) * (ple ? 4 : 3) &&
          prod_work.state_commit_calls == std::size(tokens), "mode-specific residual-only read counts");
    for (int failure = 0; failure < 5; ++failure) {
        check(production->reset(error) && !production->poisoned() && production->tokens() == 0, error);
        PortableRoute route;
        check(production->step(tokens[0], f.input(0), provider_ok, route, error), error);
        check(production->last_residual() == first, "production reset/replay bytes");
        const auto last = production->last_residual();
        PortableGdnTrace trace;
        const auto provider = [failure](int64_t, const PortableRoute &, std::vector<float> &rows, std::string &err) {
            if (failure == 4) throw 42;
            if (failure == 1) throw std::runtime_error("provider threw");
            if (failure == 2) { rows.assign(25600, std::numeric_limits<float>::quiet_NaN()); return true; }
            err = "provider rejected";
            return false;
        };
        check(!production->step(tokens[1], f.input(1), provider, route, error, failure == 3 ? &trace : nullptr),
              "production accepted failed step");
        check(production->poisoned() && production->tokens() == 1 && production->last_residual() == last &&
              route.ids.empty() && trace.attention.empty(), "failed step must poison without publishing result");
        bool called = false;
        check(!production->step(tokens[1], f.input(1),
            [&](int64_t, const PortableRoute &, std::vector<float> &, std::string &) { called = true; return false; },
            route, error) && error.find("poisoned") != std::string::npos && !called, "poisoned retry refused");
    }
    check(production->reset(error) && production->tokens() == 0 && !production->poisoned(), error);
    auto bad = options;
    bad.resident_state = false;
    check(!PortableGdnLayer::create(layer, f.views, bad, error), "production requires resident state");
    bad = options;
    bad.execution.allow_cpu_fallback = true;
    check(!PortableGdnLayer::create(layer, f.views, bad, error), "production refuses implicit host state fallback");
}
} // namespace gdn_fixture
