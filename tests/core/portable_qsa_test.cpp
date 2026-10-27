#include "portable_qsa_fixture.hpp"
#include "portable_qsa_selection.hpp"
using namespace qsa_fixture;
int main() {
  try {
    selection_tests({});
    Fixture f;
    std::string error;
    PortableQsaOptions opts;
    opts.capacity = 9;
    auto layer = PortableQsaLayer::create(3, f.views, opts, error);
    check(bool(layer), error);
    PortableRoute route;
    PortableQsaTrace trace;
    Oracle oracle;
    for (size_t t = 0; t < 9; ++t) {
      const auto keys = layer->keys(), values = layer->values();
      const auto pooled = layer->pooled_keys(), tail = layer->indexer_tail(),
                 last = layer->last_residual();
      const auto bytes = layer->graph_bytes();
      const auto placements = layer->placements().size();
      auto unchanged = [&] {
        check(layer->tokens() == t && layer->keys() == keys &&
                  layer->values() == values && layer->pooled_keys() == pooled &&
                  layer->indexer_tail() == tail &&
                  layer->last_residual() == last &&
                  layer->graph_bytes() == bytes &&
                  layer->placements().size() == placements,
              "transaction rollback");
      };
      if (t == 3 || t == 4)
        for (int failure = 0; failure < 4; ++failure) {
          check(!layer->step(
                    t, f.input(int(t)),
                    [&](int64_t, const PortableRoute &,
                        std::vector<float> &rows, std::string &) {
                      if (failure == 0)
                        return false;
                      if (failure == 1)
                        throw std::runtime_error("callback throw");
                      rows.assign(failure == 2 ? 1 : 25600,
                                  failure == 3 ? INFINITY : 0);
                      return true;
                    },
                    route, error),
                "failed callback");
          unchanged();
        }
      check(!layer->step(t + 1, f.input(int(t)), experts(), route, error),
            "position rejection");
      unchanged();
      check(layer->step(t, f.input(int(t)), experts(), route, error, &trace),
            error);
      oracle.check_step(f, t, f.input(int(t)), trace, *layer);
    }
    const auto last = layer->last_residual();
    check(!layer->step(9, f.input(9), experts(), route, error),
          "capacity exhaustion");
    check(layer->tokens() == 9 && layer->last_residual() == last,
          "capacity rollback");
    check(layer->reset(error), error);
    check(layer->tokens() == 0 && layer->keys().empty() &&
              layer->values().empty() && layer->last_residual().empty(),
          "reset");
    check(layer->step(0, f.input(0), experts(), route, error, &trace), error);
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
      check(!PortableQsaLayer::create(3, f.views, small, error),
            "budget rejection");
    }
    auto unavailable = opts;
    unavailable.execution.request_accelerator = true;
    check(!PortableQsaLayer::create(3, f.views, unavailable, error),
          "strict unavailable backend");
    unavailable.execution.allow_cpu_fallback = true;
    auto fallback = PortableQsaLayer::create(3, f.views, unavailable, error);
    check(bool(fallback), error);
    check(fallback->step(0, f.input(0), experts(), route, error), error);
    bool recorded = false;
    for (const auto &p : fallback->placements())
      if (!p.metadata_only && !p.fallback_reason.empty())
        recorded = true;
    check(recorded, "explicit fallback record");
    check(!PortableQsaLayer::create(7, f.views, opts, error),
          "other layer rejection");
    std::printf("PASS portable QSA H2560 layer3 positions0-8, scalar "
                "attention/indexer, rollback/reset/budgets/fallback\n");
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL %s\n", e.what());
    return 1;
  }
}
