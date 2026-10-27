#pragma once
#include "portable_gdn_fixture.hpp"
#include "strata/core/portable_qsa.hpp"
namespace qsa_fixture {
using namespace strata::core;
using gdn_fixture::check;
using gdn_fixture::close;
struct Fixture {
    gdn_fixture::Fixture base;
    std::vector<std::vector<uint8_t>> data;
    std::vector<PortableTensorView> views;
    Fixture() {
        for (const auto &v : base.views)
            if (v.name.find("hc_ffn_") != std::string::npos || v.name.find(".ffn_") != std::string::npos) {
                auto t = v;
                t.name.replace(0, 6, "blk.3.");
                views.push_back(t);
            }
        const std::vector<std::pair<std::string, std::vector<int64_t>>> specs{
            {"hc_attn_norm.weight", {10240}},       {"hc_attn_down.weight", {10240, 320}},
            {"hc_attn_up.weight", {320, 10240}},    {"hc_attn_inject.weight", {10240, 4}},
            {"attn_q.weight", {2560, 12288}},       {"attn_k.weight", {2560, 512}},
            {"attn_v.weight", {2560, 512}},         {"attn_output.weight", {6144, 2560}},
            {"attn_q_norm.weight", {256}},          {"attn_k_norm.weight", {256}},
            {"indexer.q_proj.weight", {2560, 512}}, {"indexer.k_proj.weight", {2560, 128}},
            {"indexer.q_norm.weight", {128}},       {"indexer.k_norm.weight", {128}}};
        data.resize(specs.size());
        for (size_t n = 0; n < specs.size(); ++n) {
            const auto &spec = specs[n];
            const bool quant = spec.second.size() == 2;
            size_t elements = 1;
            for (auto d : spec.second)
                elements *= size_t(d);
            auto &b = data[n];
            b.resize(quant ? elements / 32 * 34 : elements * 4);
            if (quant && n > 3) {
                const size_t stride = size_t(spec.second[0]) / 32 * 34;
                const auto d = ggml_fp32_to_fp16(n == 7 ? 0.125f : 0.25f);
                for (size_t r = 0; r < elements / size_t(spec.second[0]); ++r) {
                    // Distinguish heads, gate halves and signs. Gates are raw
                    // nonzero logits, and indexer dots include negative heads.
                    std::memcpy(b.data() + r * stride, &d, 2);
                    const int8_t code = int8_t(int((r / 128 + r * 7 + n) % 13) - 6);
                    std::memcpy(b.data() + r * stride + 2 + r % 32, &code, 1);
                }
            } else if (!quant)
                for (size_t i = 0; i < elements; ++i) {
                    float x = 0.8f + float((i + n) % 11) * 0.025f;
                    std::memcpy(b.data() + i * 4, &x, 4);
                }
            views.push_back({"blk.3." + spec.first, spec.second, quant ? 8u : 0u, b.data(), b.size()});
        }
    }
    float norm(size_t weight, size_t i) const {
        float f;
        std::memcpy(&f, data[weight].data() + i * 4, 4);
        return f;
    }
    std::vector<float> mm(size_t weight, const std::vector<float> &x) const {
        // Scalar dot over the fixture's one nonzero Q8 block per row.
        const size_t rows = data[weight].size() / (x.size() / 32 * 34), stride = x.size() / 32 * 34;
        std::vector<float> out(rows);
        for (size_t r = 0; r < rows; ++r) {
            uint16_t d;
            int8_t code;
            std::memcpy(&d, data[weight].data() + r * stride, 2);
            std::memcpy(&code, data[weight].data() + r * stride + 2 + r % 32, 1);
            out[r] = ggml_fp16_to_fp32(d) * float(code) * x[r % 32];
        }
        return out;
    }
    std::vector<float> input(int t) const { return base.input(t); }
};
inline PortableExpertRows experts() {
    return [](int64_t layer, const PortableRoute &r, std::vector<float> &rows, std::string &error) {
        check(layer == 3, "QSA callback layer ID");
        return gdn_fixture::experts()(0, r, rows, error);
    };
}
inline void normalize(std::vector<float> &x, size_t d, const Fixture &f, size_t weight) {
    for (size_t h = 0; h < x.size() / d; ++h) {
        double square = 0;
        for (size_t i = 0; i < d; ++i)
            square += double(x[h * d + i]) * x[h * d + i];
        const float scale = 1 / std::sqrt(float(square / d) + 1e-6f);
        for (size_t i = 0; i < d; ++i)
            x[h * d + i] *= scale * f.norm(weight, i);
    }
}
inline void rotate(std::vector<float> &x, size_t d, size_t position) {
    // All four text position axes are the same. IMRoPE rotates the first
    // 64 channels as two halves; channels >=64 pass through unchanged.
    for (size_t h = 0; h < x.size() / d; ++h)
        for (size_t j = 0; j < 32; ++j) {
            const double angle = double(position) * std::pow(1e7, -double(j) / 32);
            const float a = x[h * d + j], b = x[h * d + j + 32];
            x[h * d + j] = float(a * std::cos(angle) - b * std::sin(angle));
            x[h * d + j + 32] = float(a * std::sin(angle) + b * std::cos(angle));
        }
}
struct Oracle {
    std::vector<float> raw, keys, values, pooled;
    void check_step(const Fixture &f, size_t t, const std::vector<float> &input, const PortableQsaTrace &trace,
                    const PortableQsaLayer &layer) {
        std::vector<float> norm = input, mixed(2560, 0);
        for (size_t stream = 0; stream < 4; ++stream) {
            double square = 0;
            for (size_t i = 0; i < 2560; ++i)
                square += double(input[stream * 2560 + i]) * input[stream * 2560 + i];
            const float scale = 1 / std::sqrt(float(square / 2560) + 1e-6f);
            for (size_t i = 0; i < 2560; ++i)
                mixed[i] += input[stream * 2560 + i] * scale * f.norm(0, stream * 2560 + i) * 0.125f;
        }
        auto full = f.mm(4, mixed), k = f.mm(5, mixed), v = f.mm(6, mixed), iq = f.mm(10, mixed),
             ik = f.mm(11, mixed);
        std::vector<float> q(6144), gate(6144);
        for (size_t h = 0; h < 24; ++h)
            for (size_t j = 0; j < 256; ++j) {
                q[h * 256 + j] = full[h * 512 + j];
                gate[h * 256 + j] = full[h * 512 + j + 256];
            }
        normalize(q, 256, f, 8);
        normalize(k, 256, f, 9);
        normalize(iq, 128, f, 12);
        rotate(q, 256, t);
        rotate(k, 256, t);
        rotate(iq, 128, t);
        close(trace.query, q, 0.025f, "Q per-head layout/norm/RoPE");
        close(trace.key, k, 0.025f, "K norm/RoPE");
        close(trace.gate, gate, 0.015f, "raw per-head gates");
        close(trace.value, v, 0.015f, "unrotated V");
        close(trace.index_query, iq, 0.025f, "index query norm/RoPE");
        close(trace.index_raw_key, ik, 0.015f, "raw index key");
        // Carry the graph's projected raw values so separate projection rounding
        // cannot mask ordering errors in the scalar pooling/cache/attention oracle.
        raw.insert(raw.end(), trace.index_raw_key.begin(), trace.index_raw_key.end());
        if ((t + 1) % 4 == 0) {
            std::vector<float> mean(128, 0);
            for (size_t j = 0; j < 128; ++j)
                for (size_t b = t - 3; b <= t; ++b)
                    mean[j] += raw[b * 128 + j] * 0.25f;
            normalize(mean, 128, f, 13);
            rotate(mean, 128, t - 3);
            close(trace.pooled_key, mean, 3e-5f, "raw pooling then RMS then first-position RoPE");
            pooled.insert(pooled.end(), mean.begin(), mean.end());
        }
        std::vector<float> expected_pool = pooled;
        expected_pool.resize(pooled.size() + 128, 0);
        close(layer.pooled_keys(), expected_pool, 3e-5f, "completed and spare keys");
        std::vector<float> tail(512, 0);
        for (size_t b = (t + 1) / 4 * 4; b <= t; ++b)
            for (size_t j = 0; j < 128; ++j)
                tail[(b % 4) * 128 + j] = raw[b * 128 + j];
        close(layer.indexer_tail(), tail, 0, "raw tail");
        for (size_t j = 0; j < 512; ++j) {
            keys.push_back(ggml_fp16_to_fp32(ggml_fp32_to_fp16(trace.key[j])));
            values.push_back(ggml_fp16_to_fp32(ggml_fp32_to_fp16(trace.value[j])));
        }
        check(layer.keys().size() == keys.size() && layer.values().size() == values.size(), "full KV retention");
        for (size_t j = 0; j < keys.size(); ++j)
            check(ggml_fp16_to_fp32(layer.keys()[j]) == keys[j] &&
                      ggml_fp16_to_fp32(layer.values()[j]) == values[j],
                  "FP16 KV policy");
        std::vector<float> scores(t + 1, 1e9f);
        for (size_t b = 0; b < (t + 1) / 4; ++b) {
            float sum = 0;
            for (size_t h = 0; h < 4; ++h) {
                float dot = 0;
                for (size_t j = 0; j < 128; ++j)
                    dot += trace.index_query[h * 128 + j] * pooled[b * 128 + j];
                sum += std::max(0.0f, dot);
            }
            for (size_t j = 0; j < 4; ++j)
                scores[b * 4 + j] = sum;
        }
        close(trace.cell_scores, scores, 1e-4f, "sum ReLU heads and tail priority");
        close(trace.selected, std::vector<float>(t + 1, 1), 0, "identity selection");
        std::vector<float> attn(6144), gated(6144);
        for (size_t h = 0; h < 24; ++h) {
            std::vector<double> logits(t + 1);
            double max = -INFINITY, total = 0;
            for (size_t b = 0; b <= t; ++b) {
                double dot = 0;
                for (size_t j = 0; j < 256; ++j)
                    dot += double(trace.query[h * 256 + j]) * keys[b * 512 + (h / 12) * 256 + j];
                logits[b] = dot / 16;
                max = std::max(max, logits[b]);
            }
            for (auto &l : logits) {
                l = std::exp(l - max);
                total += l;
            }
            for (size_t j = 0; j < 256; ++j) {
                double value = 0;
                for (size_t b = 0; b <= t; ++b)
                    value += logits[b] / total * values[b * 512 + (h / 12) * 256 + j];
                attn[h * 256 + j] = float(value);
                gated[h * 256 + j] = float(value) / (1 + std::exp(-trace.gate[h * 256 + j]));
            }
        }
        close(trace.attention, attn, 5e-4f, "independent grouped attention");
        close(trace.gated_attention, gated, 5e-4f, "sigmoid raw gate");
        auto projected = f.mm(7, gated);
        auto expected_residual = input;
        for (size_t s = 0; s < 4; ++s)
            for (size_t j = 0; j < 2560; ++j)
                expected_residual[s * 2560 + j] += projected[j];
        close(trace.attention_residual, expected_residual, 0.01f, "output projection and HC combine");
    }
};
} // namespace qsa_fixture
