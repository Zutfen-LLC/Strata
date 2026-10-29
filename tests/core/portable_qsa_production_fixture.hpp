#pragma once
#include "portable_qsa_fixture.hpp"
#include <algorithm>
namespace qsa_fixture {
// Run against a strict selected backend (CPU-backed selection for local tests,
// actual Vulkan for the hardware fixture). Compare complete active state and
// unused capacity; getters deliberately never expose a production mirror.
template<class Input>
void production_tests(PortableQsaOptions opts, const std::vector<PortableTensorView> &views, Input input) {
    opts.capacity = 9;
    opts.reuse_attention = true;
    opts.production = true;
    std::string error;
    auto layer = PortableQsaLayer::create(3, views, opts, error);
    check(bool(layer), error);
    auto diagnostic = opts; diagnostic.production = false;
    auto control = PortableQsaLayer::create(3, views, diagnostic, error);
    check(bool(control), error);
    auto invalid = opts; invalid.reuse_attention = false;
    check(!PortableQsaLayer::create(3, views, invalid, error), "production requires reuse");
    invalid = opts; invalid.execution.allow_cpu_fallback = true;
    check(!PortableQsaLayer::create(3, views, invalid, error), "production forbids fallback");
    invalid = opts; invalid.execution.accelerator = nullptr;
    check(!PortableQsaLayer::create(3, views, invalid, error), "production requires selected backend");
    for (int getter = 0; getter < 4; ++getter) {
        bool refused = false;
        try {
            if (getter == 0) layer->keys();
            if (getter == 1) layer->values();
            if (getter == 2) layer->pooled_keys();
            if (getter == 3) layer->indexer_tail();
        } catch (const std::exception &) { refused = true; }
        check(refused, "production host history must refuse");
    }
    PortableRoute route, expected;
    auto verify = [&] {
        PortableQsaState state;
        check(layer->inspect_state(state, error), error);
        const size_t n = size_t(layer->tokens());
        const auto &keys = control->keys(), &values = control->values();
        check(std::equal(keys.begin(), keys.end(), state.keys.begin()) &&
              std::equal(values.begin(), values.end(), state.values.begin()), "production exact FP16 active KV");
        check(std::all_of(state.keys.begin() + n * 512, state.keys.end(), [](auto x) {return x == 0;}) &&
              std::all_of(state.values.begin() + n * 512, state.values.end(), [](auto x) {return x == 0;}),
              "production writes no future KV cells");
        const auto &pool = control->pooled_keys();
        check(std::equal(pool.begin(), pool.end(), state.pooled.begin()) && state.tail == control->indexer_tail(),
              "production exact pooled/spare/tail state");
        check(std::all_of(state.pooled.begin() + pool.size(), state.pooled.end(), [](auto x) {return x == 0;}),
              "production writes no future pooled rows");
        for (size_t i = 0; i < opts.capacity; ++i) {
            check(state.mapping[i] == int32_t(i / 4), "immutable mapping");
            check(state.bias[i] == (i < n && i >= n / 4 * 4 ? 1e9f : 0.f), "bounded incomplete block bias");
        }
        if (n)
            check(std::equal(state.spare.begin(), state.spare.end(), pool.end() - 128), "immutable dead key");
        check(layer->last_residual() == control->last_residual() && route.ids == expected.ids &&
              route.weights == expected.weights, "production exact residual/routes");
    };
    const auto before = layer->memory_usage();
    check(before.device_state == PortableQsaLayer::device_state_allocation_size(opts.execution.accelerator, opts.capacity),
          "preflight estimate equals actual aligned state on selected backend");
    check(before.device_state >= 2048 * opts.capacity + 512 * (opts.capacity / 4 + 1) + 512 + 2048 + 8 * opts.capacity,
          "actual aligned state covers raw payload");
    for (int sequence : {0, 9, 0}) {
        check(layer->reset(error) && control->reset(error), error);
        for (size_t t = 0; t < opts.capacity; ++t) {
            if (t == 0 || t == 1 || t == 3 || t == 4 || t == 7 || t == 8) {
                const auto last = layer->last_residual();
                check(!layer->step(t, input(int(t) + sequence),
                    [](int64_t, const PortableRoute &, std::vector<float> &, std::string &) {return false;}, route, error),
                    "failure after device append");
                check(layer->poisoned() && layer->tokens() == t && layer->last_residual() == last && route.ids.empty(),
                      "failure poisons without publication");
                check(!layer->step(t, input(int(t) + sequence), experts(), route, error) &&
                      error.find("poisoned") != std::string::npos, "poisoned retry refused");
                check(layer->reset(error) && !layer->poisoned(), error);
                // Replay the already committed prefix after reset, no host restore.
                for (size_t j = 0; j < t; ++j)
                    check(layer->step(j, input(int(j) + sequence), experts(), route, error), error);
            }
            check(layer->step(t, input(int(t) + sequence), experts(), route, error), error);
            check(control->step(t, input(int(t) + sequence), experts(), expected, error), error);
            verify();
        }
        check(!layer->step(opts.capacity, input(0), experts(), route, error) && layer->poisoned() &&
              layer->tokens() == opts.capacity, "capacity failure poisons without advancing");
    }
    check(layer->reset(error), error);
    PortableQsaTrace trace;
    check(!layer->step(0, input(0), experts(), route, error, &trace) && layer->poisoned(), "trace explicitly refused");
    check(layer->reset(error), error);
    const auto after = layer->memory_usage();
    check(after.device_state == before.device_state && after.host_graph == before.host_graph &&
          after.device_graph == before.device_graph, "reset retains admitted state and scheduler allocations");
    std::puts("PASS QSA device production exact state, append boundaries, poison/reset A/B/A positions0-8");
}
}
