#pragma once

#include "strata/core/portable_ffn.hpp"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

using namespace strata::core;
namespace {
void check(bool value, const std::string& message) { if (!value) throw std::runtime_error(message); }
void close(const std::vector<float>& a, const std::vector<float>& b, float tolerance = 2e-5f) {
    check(a.size() == b.size(), "vector size");
    for (size_t i = 0; i < a.size(); ++i)
        check(std::fabs(a[i] - b[i]) <= tolerance * (1 + std::fabs(b[i])), "numerical mismatch at " + std::to_string(i) +
              ": actual=" + std::to_string(a[i]) + " expected=" + std::to_string(b[i]) + " tolerance=" + std::to_string(tolerance));
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
