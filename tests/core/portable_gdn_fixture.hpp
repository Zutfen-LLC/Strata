#pragma once
#include "ggml.h"
#include "strata/core/portable_gdn.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>

namespace gdn_fixture {
inline void check(bool b, const std::string &e) {
    if (!b)
        throw std::runtime_error(e);
}
inline void close(const std::vector<float> &a, const std::vector<float> &b, float tolerance, const char *field) {
    check(a.size() == b.size(), std::string(field) + " size");
    for (size_t i = 0; i < a.size(); ++i)
        check(std::isfinite(a[i]) && std::fabs(a[i] - b[i]) <= tolerance * (1 + std::fabs(b[i])),
              std::string(field) + " mismatch at " + std::to_string(i) + " actual=" + std::to_string(a[i]) +
                  " expected=" + std::to_string(b[i]));
}
struct Fixture {
    std::vector<std::vector<uint8_t>> data;
    std::vector<strata::core::PortableTensorView> views;
    Fixture() {
        struct Spec {
            const char *name;
            std::vector<int64_t> shape;
            bool q8;
        };
        const std::vector<Spec> specs{{"hc_attn_norm.weight", {10240}, false},
                                      {"hc_attn_down.weight", {10240, 320}, true},
                                      {"hc_attn_up.weight", {320, 10240}, true},
                                      {"hc_attn_inject.weight", {10240, 4}, true},
                                      {"attn_qkv.weight", {2560, 10240}, true},
                                      {"attn_gate.weight", {2560, 6144}, true},
                                      {"ssm_alpha.weight", {2560, 48}, true},
                                      {"ssm_beta.weight", {2560, 48}, true},
                                      {"ssm_a", {48}, false},
                                      {"ssm_dt.bias", {48}, false},
                                      {"ssm_conv1d.weight", {4, 10240}, false},
                                      {"ssm_norm.weight", {128}, false},
                                      {"ssm_out.weight", {6144, 2560}, true},
                                      {"hc_ffn_norm.weight", {10240}, false},
                                      {"hc_ffn_down.weight", {10240, 320}, true},
                                      {"hc_ffn_up.weight", {320, 10240}, true},
                                      {"hc_ffn_inject.weight", {10240, 4}, true},
                                      {"ffn_gate_inp.weight", {2560, 512}, false},
                                      {"ffn_gate_inp_shexp.weight", {2560}, false},
                                      {"ffn_gate_shexp.weight", {2560, 640}, true},
                                      {"ffn_up_shexp.weight", {2560, 640}, true},
                                      {"ffn_down_shexp.weight", {640, 2560}, true}};
        data.resize(specs.size());
        for (size_t n = 0; n < specs.size(); ++n) {
            const auto &s = specs[n];
            size_t elements = 1;
            for (auto d : s.shape)
                elements *= size_t(d);
            data[n].assign(s.q8 ? elements / 32 * 34 : elements * 4, 0);
            if (s.q8) {
                const auto half = ggml_fp32_to_fp16(0.015625f);
                const size_t row_bytes = size_t(s.shape[0]) / 32 * 34;
                for (size_t r = 0; r < elements / size_t(s.shape[0]); ++r) {
                    std::memcpy(data[n].data() + r * row_bytes, &half, 2);
                    const int8_t code = int8_t(int(r % 5) - 2);
                    std::memcpy(data[n].data() + r * row_bytes + 2 + r % 32, &code, 1);
                }
            } else {
                std::vector<float> values(elements, 0);
                const std::string name = s.name;
                for (size_t j = 0; j < elements; ++j) {
                    if (name.find("norm") != std::string::npos)
                        values[j] = 0.9f + float(j % 7) * 0.02f;
                    else if (name == "ssm_a")
                        values[j] = -0.5f - float(j % 3) * 0.1f;
                    else if (name == "ssm_dt.bias")
                        values[j] = -0.7f;
                    else if (name == "ssm_conv1d.weight")
                        values[j] = float((j % 4) + 1) * 0.15f;
                    else
                        values[j] = float(int((j * 13 + n) % 29) - 14) * 0.0001f;
                }
                std::memcpy(data[n].data(), values.data(), data[n].size());
            }
            views.push_back({"blk.0." + std::string(s.name), s.shape, s.q8 ? 8u : 0u, data[n].data(), data[n].size()});
        }
    }
    std::vector<float> input(int token) const {
        std::vector<float> x(10240);
        for (size_t i = 0; i < x.size(); ++i)
            x[i] = 0.2f + float(int((i * 7 + size_t(token) * 13) % 97) - 48) * 0.01f;
        return x;
    }
    float conv_weight(size_t channel, size_t time) const {
        float v;
        std::memcpy(&v, data[10].data() + (channel * 4 + time) * 4, 4);
        return v;
    }
};
inline strata::core::PortableExpertRows experts() {
    return [](int64_t layer, const strata::core::PortableRoute &r, std::vector<float> &rows, std::string &) {
        check(layer == 0 && r.activation.size() == 2560 && r.ids.size() == 10, "actual model expert boundary");
        rows.resize(25600);
        for (size_t k = 0; k < 10; ++k)
            for (size_t h = 0; h < 2560; ++h)
                rows[k * 2560 + h] = float(r.ids[k] % 11 - 5) * 0.0005f * float(1 + h % 7);
        return true;
    };
}
// Independent scalar recurrence. Layout is M[head,value,key], with q/k heads
// broadcast by head modulo 16, not grouped repetition of each key head.
inline void reference(const Fixture &f, const std::vector<float> &previous, const std::vector<float> &history,
                      const strata::core::PortableGdnTrace &trace, const strata::core::PortableGdnLayer &layer) {
    check(trace.qkv.size() == 10240 && trace.query.size() == 2048 && trace.value.size() == 6144, "trace geometry");
    std::vector<float> conv(10240), next_history(30720);
    for (size_t c = 0; c < 10240; ++c) {
        float sum = trace.qkv[c] * f.conv_weight(c, 3);
        for (size_t t = 0; t < 3; ++t)
            sum += history[c * 3 + t] * f.conv_weight(c, t);
        conv[c] = sum / (1 + std::exp(-sum));
        next_history[c * 3] = history[c * 3 + 1];
        next_history[c * 3 + 1] = history[c * 3 + 2];
        next_history[c * 3 + 2] = trace.qkv[c];
    }
    close(trace.convolution, conv, 1e-5f, "convolution");
    close(layer.convolution_history(), next_history, 1e-6f, "history");
    std::vector<float> q(2048), k(2048);
    for (size_t h = 0; h < 16; ++h) {
        float sq = 0, sk = 0;
        for (size_t i = 0; i < 128; ++i) {
            sq += conv[h * 128 + i] * conv[h * 128 + i];
            sk += conv[2048 + h * 128 + i] * conv[2048 + h * 128 + i];
        }
        for (size_t i = 0; i < 128; ++i) {
            q[h * 128 + i] = conv[h * 128 + i] / std::sqrt(sq + 1e-6f);
            k[h * 128 + i] = conv[2048 + h * 128 + i] / std::sqrt(sk + 1e-6f);
        }
    }
    close(trace.query, q, 2e-5f, "query normalization");
    close(trace.key, k, 2e-5f, "key normalization");
    close(trace.value, std::vector<float>(conv.begin() + 4096, conv.end()), 1e-5f, "value split");
    std::vector<float> next(previous.size()), out(6144);
    for (size_t h = 0; h < 48; ++h) {
        check(trace.decay[h] < 0 && trace.beta[h] > 0 && trace.beta[h] < 1, "decay/beta domain");
        const float decay = std::exp(trace.decay[h]);
        for (size_t j = 0; j < 128; ++j) {
            const size_t at = (h * 128 + j) * 128;
            float prediction = 0;
            for (size_t i = 0; i < 128; ++i)
                prediction += previous[at + i] * decay * trace.key[(h % 16) * 128 + i];
            const float delta = (trace.value[h * 128 + j] - prediction) * trace.beta[h];
            float y = 0;
            for (size_t i = 0; i < 128; ++i) {
                next[at + i] = previous[at + i] * decay + delta * trace.key[(h % 16) * 128 + i];
                y += next[at + i] * trace.query[(h % 16) * 128 + i];
            }
            out[h * 128 + j] = y / std::sqrt(128.0f);
        }
    }
    close(layer.recurrent_state(), next, 3e-6f, "independent recurrent state");
    close(trace.attention, out, 3e-6f, "independent attention output");
}
} // namespace gdn_fixture
