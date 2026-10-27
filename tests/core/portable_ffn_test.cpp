#include "strata/core/portable_ffn.hpp"
#include "ggml.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "gguf_fixture.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>

using namespace strata::core;
namespace {
void check(bool value, const std::string& message) { if (!value) throw std::runtime_error(message); }
void close(const std::vector<float>& a, const std::vector<float>& b, float tolerance = 2e-5f) {
    check(a.size() == b.size(), "vector size");
    for (size_t i = 0; i < a.size(); ++i)
        check(std::fabs(a[i] - b[i]) <= tolerance * (1 + std::fabs(b[i])), "numerical mismatch at " + std::to_string(i));
}
float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
struct Fixture {
    PortableFfnGeometry g{32, 4, 32, 32, 4, 2, 1e-6f};
    std::vector<std::vector<float>> data;
    std::vector<PortableTensorView> views;
    Fixture() {
        const char* names[] = {"hc_ffn_norm.weight", "hc_ffn_down.weight", "hc_ffn_up.weight", "hc_ffn_inject.weight",
                              "ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight", "ffn_gate_shexp.weight",
                              "ffn_up_shexp.weight", "ffn_down_shexp.weight"};
        const std::vector<std::vector<int64_t>> shapes{{128},{128,32},{32,128},{128,4},{32,4},{32},{32,32},{32,32},{32,32}};
        data.resize(9);
        for (size_t i = 0; i < 9; ++i) {
            size_t n = 1; for (auto d : shapes[i]) n *= size_t(d);
            data[i].resize(n);
            for (size_t j = 0; j < n; ++j) data[i][j] = i == 0 ? 0.7f + float(j % 7) * 0.03f : float(int((j * 17 + i * 11) % 29) - 14) * 0.013f;
            views.push_back({"blk.0." + std::string(names[i]), shapes[i], 0, data[i].data(), n * sizeof(float)});
        }
    }
    std::vector<float> mm(size_t tensor, const std::vector<float>& x) const {
        const auto& w = data[tensor];
        std::vector<float> out(w.size() / x.size(), 0);
        for (size_t r = 0; r < out.size(); ++r)
            for (size_t c = 0; c < x.size(); ++c) out[r] += w[r * x.size() + c] * x[c];
        return out;
    }
    // Independent scalar transcription of the existing gr.hpp/layer.cpp order.
    std::vector<float> oracle(const std::vector<float>& residual, PortableRoute& route) const {
        std::vector<float> norm(128);
        for (size_t s = 0; s < 4; ++s) {
            float ss = 0; for (size_t h = 0; h < 32; ++h) ss += residual[s*32+h] * residual[s*32+h];
            float scale = 1 / std::sqrt(ss / 32 + g.epsilon);
            for (size_t h = 0; h < 32; ++h) norm[s*32+h] = residual[s*32+h] * scale * data[0][s*32+h];
        }
        auto low = mm(1, norm);
        for (auto& x : low) { x /= 4; x *= sigmoid(x); }
        auto gate = mm(2, low), inject = mm(3, norm);
        route.activation.assign(32, 0);
        for (size_t s = 0; s < 4; ++s)
            for (size_t h = 0; h < 32; ++h) route.activation[h] += norm[s*32+h] * sigmoid(gate[s*32+h]);
        for (auto& x : route.activation) x /= 4;
        auto logits = mm(4, route.activation);
        std::vector<int32_t> ids{0,1,2,3};
        std::stable_sort(ids.begin(), ids.end(), [&](int a, int b) { return logits[a] > logits[b]; });
        route.ids.assign(ids.begin(), ids.begin()+2);
        double total = 0;
        for (auto id : route.ids) total += std::exp(double(logits[id]));
        route.weights.clear();
        for (auto id : route.ids) route.weights.push_back(float(std::exp(double(logits[id])) / total));
        auto sg = mm(6, route.activation), su = mm(7, route.activation);
        for (size_t h = 0; h < sg.size(); ++h) sg[h] *= sigmoid(sg[h]) * su[h];
        auto block = mm(8, sg);
        const float shared_gate = sigmoid(mm(5, route.activation)[0]);
        for (size_t h = 0; h < block.size(); ++h) {
            block[h] *= shared_gate;
            for (size_t k = 0; k < 2; ++k) block[h] += float(route.ids[k] + 1) * (0.03f + float(h) * 0.001f) * route.weights[k];
        }
        auto out = residual;
        for (size_t s = 0; s < 4; ++s)
            for (size_t h = 0; h < 32; ++h) out[s*32+h] += block[h] * 2 * sigmoid(inject[s]/4);
        return out;
    }
};
}

int main() {
    try {
        Fixture f;
        std::string error;
        PortableFfnOptions options;
        auto runner = PortableFfn::create(f.g, 0, f.views, options, error);
        check(bool(runner), error);
        PortableRoute route, expected_route;
        const PortableExpertRows experts = [&](int64_t layer, const PortableRoute& r, std::vector<float>& rows, std::string&) {
            check(layer == 0 && r.ids.size() == 2 && r.activation.size() == 32, "callback boundary");
            rows.resize(64);
            for (size_t k = 0; k < 2; ++k)
                for (size_t h = 0; h < 32; ++h) rows[k*32+h] = float(r.ids[k]+1) * (0.03f + float(h)*0.001f);
            return true;
        };
        check(!runner->step(experts, route, error), "reset required");
        std::vector<float> initial(128);
        for (size_t i = 0; i < initial.size(); ++i) initial[i] = 0.2f + float(int(i % 19)-9) * 0.06f;
        check(runner->reset(initial, error), error);
        auto expected = f.oracle(initial, expected_route);
        check(runner->step(experts, route, error), error);
        close(runner->residual(), expected);
        close(route.activation, expected_route.activation);
        close(route.weights, expected_route.weights);
        check(route.ids == expected_route.ids, "routing IDs");
        expected = f.oracle(expected, expected_route);
        check(runner->step(experts, route, error), error);
        close(runner->residual(), expected);
        check(runner->reset(initial, error) && runner->step(experts, route, error), error);
        close(runner->residual(), f.oracle(initial, expected_route));
        const auto saved = runner->residual();
        check(!runner->step([](int64_t, const PortableRoute&, std::vector<float>&, std::string& e) { e="dispatch rejected"; return false; }, route, error), "dispatch rejection");
        close(runner->residual(), saved, 0);
        check(!runner->step([](int64_t, const PortableRoute&, std::vector<float>& rows, std::string&) { rows={1}; return true; }, route, error), "row count rejection");
        close(runner->residual(), saved, 0);
        check(!runner->step([](int64_t, const PortableRoute&, std::vector<float>& rows, std::string&) { rows.assign(64, std::numeric_limits<float>::quiet_NaN()); return true; }, route, error), "nonfinite rejection");
        close(runner->residual(), saved, 0);
        check(runner->step(experts, route, error), error); // recovery after rejected callbacks
        check(runner->weight_bytes() > 0 && runner->graph_bytes() > 0, "allocation accounting");

        options.request_accelerator = true;
        check(!PortableFfn::create(f.g, 0, f.views, options, error), "strict unavailable device");
        options.allow_cpu_fallback = true;
        auto fallback = PortableFfn::create(f.g, 0, f.views, options, error);
        check(bool(fallback), error);
        size_t rejected = 0;
        for (const auto& p : fallback->placements()) if (!p.fallback_reason.empty()) ++rejected;
        check(rejected > 0, "fallback reporting");
        check(fallback->reset(initial, error) && fallback->step(experts, route, error), error);
        close(fallback->residual(), f.oracle(initial, expected_route));
        {
            // A real CPU execution backend with a private device descriptor that
            // declines graph ops. Exercises supports_op rejection, not GPU math.
            struct DecliningBackend {
                ggml_backend_t backend = ggml_backend_cpu_init();
                ggml_backend_device device = *backend->device;
                ggml_backend_dev_t original = backend->device;
                DecliningBackend() {
                    device.iface.supports_op = [](ggml_backend_dev_t, const ggml_tensor*) { return false; };
                    backend->device = &device;
                    backend->iface.get_name = [](ggml_backend_t) { return "TEST_DECLINING_CPU"; };
                }
                ~DecliningBackend() { backend->device = original; ggml_backend_free(backend); }
            } declining;
            options.accelerator = declining.backend;
            options.allow_cpu_fallback = false;
            check(!PortableFfn::create(f.g, 0, f.views, options, error) && error.find("declines") != std::string::npos,
                  "strict operation rejection");
            options.allow_cpu_fallback = true;
            auto operation_fallback = PortableFfn::create(f.g, 0, f.views, options, error);
            check(bool(operation_fallback), error);
            bool reported = false;
            for (const auto& p : operation_fallback->placements())
                if (p.fallback_reason.find("declines") != std::string::npos) reported = true;
            check(reported, "declined operation reporting");
            check(operation_fallback->reset(initial, error) && operation_fallback->step(experts, route, error), error);
            close(operation_fallback->residual(), f.oracle(initial, expected_route));
        }
        options = {}; options.graph_budget = 1;
        check(!PortableFfn::create(f.g, 0, f.views, options, error), "graph budget");
        options = {}; options.weight_budget = 1;
        check(!PortableFfn::create(f.g, 0, f.views, options, error), "weight budget");
        auto malformed = f.views; malformed[1].shape[0] = 127;
        check(!PortableFfn::create(f.g, 0, malformed, {}, error), "bad tensor shape");
        // Exercise the actual fleet projection encoding without requiring model files.
        std::vector<std::vector<uint8_t>> q8(9);
        auto quantized = f.views;
        for (size_t i : {size_t(1),size_t(2),size_t(3),size_t(6),size_t(7),size_t(8)}) {
            q8[i].resize(f.data[i].size()/32*34);
            ggml_quantize_chunk(GGML_TYPE_Q8_0, f.data[i].data(), q8[i].data(), 0,
                                int64_t(f.data[i].size()/size_t(f.views[i].shape[0])), f.views[i].shape[0], nullptr);
            quantized[i].type = 8; quantized[i].bytes = q8[i].size(); quantized[i].data = q8[i].data();
        }
        auto qrunner = PortableFfn::create(f.g, 0, quantized, {}, error);
        check(bool(qrunner), error);
        check(qrunner->reset(initial, error) && qrunner->step(experts, route, error), error);
        close(qrunner->residual(), f.oracle(initial, expected_route), 0.003f);
        // Real split-reader binding: metadata-only first shard, tensors in the
        // second. The payload is synthetic but bytes/layout match the inputs.
        const auto root = std::filesystem::temp_directory_path() /
            ("strata-portable-ffn-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        check(std::filesystem::create_directory(root), "fixture directory");
        struct Cleanup { std::filesystem::path p; ~Cleanup() { std::error_code e; std::filesystem::remove_all(p,e); } } cleanup{root};
        auto kv = fixture::split_keys(0, 2, 9);
        kv.push_back(fixture::str("general.architecture", "qwen4exp"));
        for (auto pair : {std::pair<const char*, uint64_t>{"block_count",48}, {"embedding_length",32},
                          {"expert_count",4}, {"expert_used_count",2}, {"attention.head_count",24},
                          {"attention.head_count_kv",2}, {"hyper_connection.count",4},
                          {"hyper_connection.low_rank",32}, {"expert_feed_forward_length",32},
                          {"expert_shared_feed_forward_length",32}})
            kv.push_back(fixture::u32("qwen4exp." + std::string(pair.first), pair.second));
        const auto first = root / "fixture-00001-of-00002.gguf";
        const auto second = root / "fixture-00002-of-00002.gguf";
        fixture::write(first, kv, {});
        std::vector<fixture::Tensor> tensors;
        for (const auto& v : quantized) {
            fixture::Tensor t; t.name = v.name; t.type = v.type;
            for (auto d : v.shape) t.shape.push_back(uint64_t(d));
            tensors.push_back(t);
        }
        const auto written = fixture::write(second, fixture::split_keys(1,2,9), tensors);
        {
            std::fstream file(second, std::ios::in | std::ios::out | std::ios::binary);
            for (size_t i = 0; i < quantized.size(); ++i) {
                file.seekp(std::streamoff(written.data_start + written.offsets[i]));
                file.write(static_cast<const char*>(quantized[i].data), std::streamsize(quantized[i].bytes));
            }
            check(bool(file), "fixture payload write");
        }
        std::unique_ptr<PortableFfn> bound;
        {
            auto model = strata::GgufModel::open(first.string());
            bound = PortableFfn::from_gguf(model, 0, {}, error);
            check(bool(bound), error);
            check(!PortableFfn::from_gguf(model, 48, {}, error), "GGUF layer range");
        } // mapping lifetime ends: copied weights must remain usable
        check(bound->reset(initial, error) && bound->step(experts, route, error), error);
        close(bound->residual(), qrunner->residual(), 0);
        std::puts("portable_ffn_test: scalar parity, state/reset, callback failures, placement/fallback, budgets, Q8_0, split GGUF passed");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
