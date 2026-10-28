#include "portable_qsa_fixture.hpp"
#include "portable_qsa_selection.hpp"
#include <limits>
using namespace qsa_fixture;
int main() {
    try {
        spare_high_score_test({});
        selection_tests({});
        Fixture f;
        std::string error;
        PortableQsaOptions opts;
        opts.capacity = 9;
        auto layer = PortableQsaLayer::create(3, f.views, opts, error);
        check(bool(layer), error);
        auto clean = PortableQsaLayer::create(3, f.views, opts, error);
        check(bool(clean), error);
        PortableRoute clean_route;
        std::vector<float> first_residual;
        PortableRoute route;
        PortableQsaTrace trace;
        Oracle oracle;
        for (size_t t = 0; t < 9; ++t) {
            const auto keys = layer->keys(), values = layer->values();
            const auto pooled = layer->pooled_keys(), tail = layer->indexer_tail(), last = layer->last_residual();
            const auto bytes = layer->graph_bytes();
            const auto placements = layer->placements().size();
            auto unchanged = [&] {
                check(layer->tokens() == t && layer->keys() == keys && layer->values() == values &&
                          layer->pooled_keys() == pooled && layer->indexer_tail() == tail &&
                          layer->last_residual() == last && layer->graph_bytes() == bytes &&
                          layer->placements().size() == placements,
                      "transaction rollback");
            };
            if (t == 0 || t == 3 || t == 4)
                for (int failure = 0; failure < 4; ++failure) {
                    check(!layer->step(
                              t, f.input(int(t)),
                              [&](int64_t, const PortableRoute &, std::vector<float> &rows, std::string &) {
                                  if (failure == 0)
                                      return false;
                                  if (failure == 1)
                                      throw std::runtime_error("callback throw");
                                  rows.assign(failure == 2 ? 1 : 25600, failure == 3 ? INFINITY : 0);
                                  return true;
                              },
                              route, error),
                          "failed callback");
                    unchanged();
                }
            for (float invalid : {NAN, INFINITY, -INFINITY})
                for (size_t at : {size_t(0), size_t(5120), size_t(10239)}) {
                    auto bad_input = f.input(int(t));
                    bad_input[at] = invalid;
                    bool dispatched = false;
                    trace.query = {1};
                    check(!layer->step(t, bad_input,
                        [&](int64_t, const PortableRoute &, std::vector<float> &, std::string &) {
                            dispatched = true;
                            return true;
                        }, route, error, &trace), "nonfinite input rejection");
                    check(error == "portable QSA: nonfinite input/result" && !dispatched &&
                          trace.query.empty() && route.ids.empty(), "QSA finite diagnostic/trace");
                    unchanged();
                }
            check(!layer->step(t, {}, experts(), route, error), "malformed input rejection");
            unchanged();
            check(!layer->step(t, f.input(int(t)), {}, route, error), "missing callback rejection");
            unchanged();
            check(!layer->step(t + 1, f.input(int(t)), experts(), route, error), "position rejection");
            unchanged();
            check(layer->step(t, f.input(int(t)), experts(), route, error, &trace), error);
            oracle.check_step(f, t, f.input(int(t)), trace, *layer);
            check(clean->step(t, f.input(int(t)), experts(), clean_route, error), error);
            check(layer->keys() == clean->keys() && layer->values() == clean->values() &&
                      layer->pooled_keys() == clean->pooled_keys() &&
                      layer->indexer_tail() == clean->indexer_tail() &&
                      layer->last_residual() == clean->last_residual() && route.ids == clean_route.ids &&
                      route.weights == clean_route.weights,
                  "failure retry equals uninterrupted control");
            if (t == 0)
                first_residual = layer->last_residual();
        }
        const auto last = layer->last_residual();
        check(!layer->step(9, f.input(9), experts(), route, error), "capacity exhaustion");
        check(layer->tokens() == 9 && layer->last_residual() == last, "capacity rollback");
        check(layer->reset(error), error);
        check(layer->tokens() == 0 && layer->keys().empty() && layer->values().empty() &&
                  layer->last_residual().empty(),
              "reset");
        check(layer->step(0, f.input(0), experts(), route, error, &trace), error);
        check(layer->last_residual() == first_residual, "reset equals fresh control");
        auto late_views = f.views;
        for (auto &v : late_views)
            v.name = "blk.47." + v.name.substr(6);
        {
            auto late = PortableQsaLayer::create(47, late_views, opts, error);
            check(bool(late), error);
            check(late->step(0, f.input(0),
                [&](int64_t id, const PortableRoute &r, std::vector<float> &rows, std::string &e) {
                    check(id == 47, "late QSA dispatch ID");
                    return experts()(3, r, rows, e);
                }, route, error), error);
            close(late->last_residual(), first_residual, 0, "late QSA equivalence");
            check(late->keys() == layer->keys(), "late QSA cache equivalence");
        }
        for (int id = 3; id < 48; id += 4)
            check(!PortableQsaLayer::create(id, {}, opts, error) &&
                  error.find("missing/extra") != std::string::npos, "all QSA IDs admitted");
        for (int id : {-1, 0, 2, 46, 48, 51})
            check(!PortableQsaLayer::create(id, {}, opts, error) &&
                  error.find("expected a QSA layer") != std::string::npos, "QSA layer family guard");
        Oracle control;
        control.check_step(f, 0, f.input(0), trace, *layer);
        for (auto field : {0, 1, 2}) {
            auto small = opts;
            if (field == 0)
                small.state_budget = 1;
            if (field == 1)
                small.execution.graph_budget = 1;
            if (field == 2)
                small.execution.weight_budget = 1;
            check(!PortableQsaLayer::create(3, f.views, small, error), "budget rejection");
        }
        auto huge = f.data[6];
        const auto scale = ggml_fp32_to_fp16(65504.0f);
        for (size_t row = 0; row < 512; ++row)
            std::memcpy(huge.data() + row * 2720, &scale, 2);
        auto overflow_views = f.views;
        for (auto &view : overflow_views)
            if (view.name == "blk.3.attn_v.weight")
                view.data = huge.data();
        auto overflow = PortableQsaLayer::create(3, overflow_views, opts, error);
        check(bool(overflow), error);
        trace.query = {1};
        check(!overflow->step(0, f.input(0), experts(), route, error, &trace) && overflow->tokens() == 0 &&
                  overflow->keys().empty() && overflow->values().empty() && overflow->last_residual().empty() &&
                  error == "portable QSA: FP16 cache overflow" && trace.query.empty() && route.ids.empty(),
              "FP16 overflow diagnostic/rollback/trace");
        {
            auto bad_views = f.views;
            std::vector<float> bad_norm(256, 1);
            bad_norm.back() = NAN;
            for (auto &v : bad_views)
                if (v.name == "blk.3.attn_q_norm.weight")
                    v.data = bad_norm.data();
            auto bad = PortableQsaLayer::create(3, bad_views, opts, error);
            check(bool(bad), error);
            bool dispatched = false;
            check(!bad->step(0, f.input(0),
                [&](int64_t, const PortableRoute &, std::vector<float> &, std::string &) {
                    dispatched = true;
                    return true;
                }, route, error, &trace) && error == "portable QSA: nonfinite input/result" && !dispatched,
                  "QSA read-side finite diagnostic: " + error);
            check(bad->tokens() == 0 && bad->keys().empty() && bad->values().empty() &&
                  bad->last_residual().empty() && trace.query.empty(), "QSA read-side rollback/trace");
        }
        auto unavailable = opts;
        unavailable.execution.request_accelerator = true;
        check(!PortableQsaLayer::create(3, f.views, unavailable, error), "strict unavailable backend");
        unavailable.execution.allow_cpu_fallback = true;
        auto fallback = PortableQsaLayer::create(3, f.views, unavailable, error);
        check(bool(fallback), error);
        check(fallback->step(0, f.input(0), experts(), route, error), error);
        bool recorded = false;
        for (const auto &p : fallback->placements())
            if (!p.metadata_only && !p.fallback_reason.empty())
                recorded = true;
        check(recorded, "explicit fallback record");
        check(!PortableQsaLayer::create(7, f.views, opts, error), "other layer rejection");
        auto finite_edges = f.input(0);
        finite_edges[0] = 0.0f;
        finite_edges[1] = -0.0f;
        finite_edges[5120] = std::numeric_limits<float>::min();
        finite_edges.back() = std::numeric_limits<float>::denorm_min();
        check(layer->reset(error) && layer->step(0, finite_edges, experts(), route, error, &trace), error);
        const auto edge_trace = trace;
        const auto edge_keys = layer->keys(), edge_values = layer->values();
        const auto edge_residual = layer->last_residual();
        check(layer->reset(error) && layer->step(0, finite_edges, experts(), route, error, &trace), error);
        check(edge_keys == layer->keys() && edge_values == layer->values() &&
              std::memcmp(edge_residual.data(), layer->last_residual().data(), edge_residual.size() * sizeof(float)) == 0,
              "QSA finite edge cache/residual bytes");
        for (auto member : {&PortableQsaTrace::query, &PortableQsaTrace::key, &PortableQsaTrace::value,
                            &PortableQsaTrace::gate, &PortableQsaTrace::index_query, &PortableQsaTrace::index_raw_key,
                            &PortableQsaTrace::pooled_key, &PortableQsaTrace::cell_scores, &PortableQsaTrace::selected,
                            &PortableQsaTrace::attention, &PortableQsaTrace::gated_attention, &PortableQsaTrace::attention_residual}) {
            const auto &a = trace.*member, &b = edge_trace.*member;
            check(a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0),
                  "QSA finite edge trace replay bytes");
        }
        std::printf("PASS portable QSA H2560 layer3 positions0-8, scalar "
                    "attention/indexer, rollback/reset/budgets/fallback\n");
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
