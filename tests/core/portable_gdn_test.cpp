#include "gguf_fixture.hpp"
#include "portable_gdn_fixture.hpp"
#include <chrono>
#include <limits>
using namespace strata::core;
using namespace gdn_fixture;
int main() {
    try {
        Fixture f;
        std::string error;
        PortableGdnOptions options;
        auto layer = PortableGdnLayer::create(0, f.views, options, error);
        check(bool(layer), error);
        PortableRoute route;
        PortableGdnTrace trace;
        for (int t = 0; t < 3; ++t) {
            auto previous = layer->recurrent_state(), history = layer->convolution_history();
            check(layer->step(f.input(t), experts(), route, error, &trace), error);
            reference(f, previous, history, trace, *layer);
            check(layer->tokens() == uint64_t(t + 1), "token count");
        }
        auto state = layer->recurrent_state(), history = layer->convolution_history(),
             residual = layer->last_residual();
        auto reject = [](int64_t, const PortableRoute &, std::vector<float> &, std::string &e) {
            e = "deliberate callback rejection";
            return false;
        };
        check(!layer->step(f.input(3), reject, route, error), "callback rejection");
        close(layer->recurrent_state(), state, 0, "state rollback");
        close(layer->convolution_history(), history, 0, "history rollback");
        close(layer->last_residual(), residual, 0, "residual rollback");
        check(layer->tokens() == 3, "counter rollback");
        check(!layer->step(
                  f.input(3),
                  [](int64_t, const PortableRoute &, std::vector<float> &, std::string &) -> bool {
                      throw std::runtime_error("callback threw");
                  },
                  route, error),
              "throwing callback");
        close(layer->recurrent_state(), state, 0, "throw state rollback");
        close(layer->convolution_history(), history, 0, "throw history rollback");
        close(layer->last_residual(), residual, 0, "throw residual rollback");
        check(!layer->step(
                  f.input(3),
                  [](int64_t, const PortableRoute &, std::vector<float> &rows, std::string &) {
                      rows.assign(25600, std::numeric_limits<float>::quiet_NaN());
                      return true;
                  },
                  route, error),
              "nonfinite callback");
        close(layer->recurrent_state(), state, 0, "nonfinite state rollback");
        close(layer->convolution_history(), history, 0, "nonfinite history rollback");
        close(layer->last_residual(), residual, 0, "nonfinite residual rollback");
        auto bad_input = f.input(3);
        bad_input[0] = std::numeric_limits<float>::infinity();
        check(!layer->step(bad_input, experts(), route, error), "nonfinite input");
        check(layer->tokens() == 3, "failed token count");
        check(layer->step(f.input(3), experts(), route, error, &trace), error);
        reference(f, state, history, trace, *layer);
        check(layer->reset(error), error);
        check(layer->tokens() == 0 && layer->last_residual().empty(), "reset counter/output");
        for (float x : layer->recurrent_state())
            check(x == 0, "reset state");
        for (float x : layer->convolution_history())
            check(x == 0, "reset history");
        state = layer->recurrent_state();
        history = layer->convolution_history();
        check(layer->step(f.input(0), experts(), route, error, &trace), error);
        reference(f, state, history, trace, *layer);
        const auto first_output = layer->last_residual();
        check(!PortableGdnLayer::create(1, f.views, options, error), "PLE layer rejection");
        check(!PortableGdnLayer::create(3, f.views, options, error), "QSA layer rejection");
        options.state_budget = 1;
        check(!PortableGdnLayer::create(0, f.views, options, error), "state budget");
        options = {};
        options.execution.graph_budget = 1;
        check(!PortableGdnLayer::create(0, f.views, options, error), "graph budget");
        options = {};
        options.execution.weight_budget = 1;
        check(!PortableGdnLayer::create(0, f.views, options, error), "weight budget");
        // Exact-shape synthetic GGUF loader; mapping closes before execution.
        const auto root = std::filesystem::temp_directory_path() /
                          ("strata-gdn-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        check(std::filesystem::create_directory(root), "fixture directory");
        struct Cleanup {
            std::filesystem::path p;
            ~Cleanup() {
                std::error_code e;
                std::filesystem::remove_all(p, e);
            }
        } cleanup{root};
        std::vector<fixture::Kv> kv{fixture::str("general.architecture", "qwen4exp"),
                                    fixture::u32s("qwen4exp.ple.layers", {1})};
        for (auto pair : {std::pair<const char *, uint64_t>{"block_count", 48},
                          {"embedding_length", 2560},
                          {"expert_count", 512},
                          {"expert_used_count", 10},
                          {"expert_feed_forward_length", 640},
                          {"expert_shared_feed_forward_length", 640},
                          {"attention.head_count", 24},
                          {"attention.head_count_kv", 2},
                          {"ssm.state_size", 128},
                          {"ssm.group_count", 16},
                          {"ssm.time_step_rank", 48},
                          {"ssm.inner_size", 6144},
                          {"ssm.conv_kernel", 4},
                          {"hyper_connection.count", 4},
                          {"hyper_connection.low_rank", 320}})
            kv.push_back(fixture::u32("qwen4exp." + std::string(pair.first), pair.second));
        std::vector<fixture::Tensor> tensors;
        for (const auto &v : f.views) {
            fixture::Tensor t;
            t.name = v.name;
            t.type = v.type;
            for (auto d : v.shape)
                t.shape.push_back(uint64_t(d));
            tensors.push_back(t);
        }
        const auto path = root / "layer0.gguf";
        const auto written = fixture::write(path, kv, tensors);
        {
            std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
            for (size_t i = 0; i < f.views.size(); ++i) {
                file.seekp(std::streamoff(written.data_start + written.offsets[i]));
                file.write(static_cast<const char *>(f.views[i].data), std::streamsize(f.views[i].bytes));
            }
            check(bool(file), "payload write");
        }
        std::unique_ptr<PortableGdnLayer> bound;
        {
            auto model = strata::GgufModel::open(path.string());
            bound = PortableGdnLayer::from_gguf(model, 0, {}, error);
            check(bool(bound), error);
        }
        check(bound->step(f.input(0), experts(), route, error), error);
        close(bound->last_residual(), first_output, 0, "GGUF binding");
        kv[1] = fixture::u32s("qwen4exp.ple.layers", {0});
        fixture::write(root / "bad-ple.gguf", kv, {});
        {
            auto model = strata::GgufModel::open((root / "bad-ple.gguf").string());
            check(!PortableGdnLayer::from_gguf(model, 0, {}, error), "configured PLE0 rejection");
        }
        std::printf("portable_gdn_test: actual H2560/FF640/S128/Hk16/Hv48; independent multi-token state/convolution, "
                    "reset, rollback, budgets, PLE/QSA rejection passed; weights=%zu graphs=%zu host_state_bound=%zu\n",
                    layer->weight_bytes(), layer->graph_bytes(), layer->state_bytes());
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
