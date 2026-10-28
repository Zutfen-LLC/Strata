#pragma once
#include "ggml-cpu.h"
#include "portable_gdn_fixture.hpp"
namespace ple_fixture {
using gdn_fixture::check;
using gdn_fixture::close;
using namespace strata::core;
inline PortablePleConfig small_config() {
    PortablePleConfig c;
    for (int h = 0; h < 16; ++h) {
        c.hash.vocab[h] = 17;
        c.hash.offset[h] = uint64_t(h * 17);
    }
    return c;
}
inline std::vector<uint8_t> table_bytes(size_t rows) {
    // Each activation block contains a +127 endpoint, so the CPU Q8_0
    // activation image is exact. This permits a float scalar oracle without
    // conflating PLE arithmetic with backend activation quantization error.
    std::vector<uint8_t> bytes(rows * 170);
    for (size_t r = 0; r < rows; ++r)
        for (size_t b = 0; b < 5; ++b) {
            const auto scale = ggml_fp32_to_fp16(float(b + 1) / 128);
            std::memcpy(bytes.data() + r * 170 + b * 34, &scale, 2);
            for (int i = 0; i < 32; ++i)
                bytes[r * 170 + b * 34 + 2 + size_t(i)] =
                    uint8_t(i == 31 ? 127 : int((r * 7 + b * 13 + size_t(i)) % 127) - 63);
        }
    return bytes;
}
inline std::shared_ptr<PortablePleTable> small_table(std::string &error) {
    auto bytes = table_bytes(272);
    return PortablePleTable::copy({"per_layer_token_embd.weight", {160, 272}, 8, bytes.data(), bytes.size()},
                                  small_config(), bytes.size(), error);
}
struct Fixture : gdn_fixture::Fixture {
    std::vector<std::vector<uint8_t>> ple_data;
    Fixture() {
        for (auto &v : views)
            v.name.replace(0, 6, "blk.1.");
        const std::vector<std::pair<std::string, std::vector<int64_t>>> specs{
            {"ple_key.weight", {2560, 10240}},   {"ple_value.weight", {2560, 2560}},
            {"ple_norm_key.weight", {2560, 4}},  {"ple_norm_query.weight", {2560, 4}},
            {"ple_norm_conv.weight", {2560, 4}}, {"ple_conv1d.weight", {4, 10240}}};
        ple_data.resize(specs.size());
        for (size_t n = 0; n < specs.size(); ++n) {
            size_t count = size_t(specs[n].second[0] * specs[n].second[1]);
            auto &bytes = ple_data[n];
            bytes.resize(n < 2 ? count / 32 * 34 : count * 4);
            if (n < 2) {
                // Sparse exactly representable Q8 projections; scalar reference below
                // independently multiplies the dequantized rows.
                const size_t stride = 2560 / 32 * 34;
                for (size_t r = 0; r < count / 2560; ++r) {
                    const auto d = ggml_fp32_to_fp16(0.125f);
                    std::memcpy(bytes.data() + r * stride, &d, 2);
                    bytes[r * stride + 2 + r % 32] = uint8_t(int(r % 5) - 2);
                }
            } else
                for (size_t i = 0; i < count; ++i) {
                    const float v = n == 5 ? float(1 + i % 4) * 0.1f : 0.8f + float(i % 7) * 0.03f;
                    std::memcpy(bytes.data() + i * 4, &v, 4);
                }
            views.push_back({"blk.1." + specs[n].first, specs[n].second, n < 2 ? 8u : 0u, bytes.data(), bytes.size()});
        }
    }
    float f32(size_t weight, size_t i) const {
        float f;
        std::memcpy(&f, ple_data[weight].data() + i * 4, 4);
        return f;
    }
};
inline PortableExpertRows experts() {
    return [](int64_t layer, const PortableRoute &r, std::vector<float> &rows, std::string &error) {
        check(layer == 1, "PLE expert layer ID");
        return gdn_fixture::experts()(0, r, rows, error);
    };
}
inline void reference(const Fixture &f, const std::vector<float> &input, const std::vector<float> &history,
                      const PortableGdnTrace &trace, const PortableGdnLayer &layer) {
    std::vector<float> key(10240), value(2560), query(input), norm(10240), conv(10240), residual(10240), gates(4),
        next(92160);
    // A general scalar matrix dot using ggml's pinned row decoder as a separate
    // oracle for the production Strata decoder and graph projections.
    for (size_t n = 0; n < 2; ++n) {
        auto &out = n ? value : key;
        std::vector<float> row(2560);
        for (size_t r = 0; r < out.size(); ++r) {
            ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(f.ple_data[n].data() + r * 2720, row.data(), 2560);
            for (size_t i = 0; i < 2560; ++i)
                out[r] += row[i] * trace.ple_embedding[i];
        }
    }
    auto grouped = [&](std::vector<float> &x, size_t w) {
        for (size_t h = 0; h < 4; ++h) {
            double square = 0;
            for (size_t i = 0; i < 2560; ++i)
                square += double(x[h * 2560 + i]) * x[h * 2560 + i];
            const float scale = 1 / std::sqrt(float(square / 2560) + 1e-6f);
            for (size_t i = 0; i < 2560; ++i)
                x[h * 2560 + i] *= scale * f.f32(w, h * 2560 + i);
        }
    };
    grouped(key, 2);
    grouped(query, 3);
    for (size_t h = 0; h < 4; ++h) {
        float score = 0;
        for (size_t i = 0; i < 2560; ++i)
            score += key[h * 2560 + i] * query[h * 2560 + i];
        score /= std::sqrt(2560.0f);
        const float signed_root = (score > 0   ? 1.0f
                                   : score < 0 ? -1.0f
                                               : 0.0f) *
                                  std::sqrt(std::max(std::fabs(score), 1e-6f));
        gates[h] = 1 / (1 + std::exp(-signed_root));
        for (size_t i = 0; i < 2560; ++i)
            norm[h * 2560 + i] = value[i] * gates[h];
    }
    grouped(norm, 4);
    for (size_t c = 0; c < 10240; ++c) {
        float sum = 0;
        for (size_t tap = 0; tap < 4; ++tap)
            sum += f.f32(5, c * 4 + tap) * (tap == 3 ? norm[c] : history[c * 9 + tap * 3]);
        conv[c] = sum / (1 + std::exp(-sum));
        residual[c] = input[c] + value[c % 2560] * gates[c / 2560] + conv[c];
        for (size_t t = 0; t < 8; ++t)
            next[c * 9 + t] = history[c * 9 + t + 1];
        next[c * 9 + 8] = norm[c];
    }
    close(trace.ple_key, key, 0.003f, "PLE key");
    close(trace.ple_value, value, 0.003f, "PLE value");
    close(trace.ple_gate, gates, 0.003f, "PLE signed-root gate");
    close(trace.ple_normalized, norm, 0.003f, "PLE normalized history input");
    close(trace.ple_convolution, conv, 0.003f, "PLE dilated conv");
    close(trace.ple_residual, residual, 0.003f, "PLE residual");
    close(layer.ple_convolution_history(), next, 0.003f, "PLE history advance");
}
inline void exercise(PortableGdnLayer &layer, const Fixture &f, PortableGdnLayer *cpu = nullptr) {
    std::string error;
    PortableRoute route;
    PortableGdnTrace trace;
    const int32_t tokens[] = {0, 19, 248044, 42, 248056, 248046, 93, 15, 26, 37};
    std::vector<float> first;
    std::array<int32_t, 2> previous_tokens{{-1, -1}};
    const auto *state0=layer.recurrent_state().data(), *history0=layer.convolution_history().data(),
               *ple0=layer.ple_convolution_history().data();
    const float *state1=nullptr, *history1=nullptr, *ple1=nullptr;
    for (size_t t = 0; t < 10; ++t) {
        auto history = layer.ple_convolution_history(), state = layer.recurrent_state(),
             gdn_history = layer.convolution_history();
        check(layer.step(tokens[t], f.input(int(t)), experts(), route, error, &trace), error);
        if(t==0) {
            state1=layer.recurrent_state().data();history1=layer.convolution_history().data();
            ple1=layer.ple_convolution_history().data();
            check(state1!=state0 && history1!=history0 && ple1!=ple0,"PLE/GDN candidate banks do not alias committed mirrors");
        }
        check(layer.recurrent_state().data()==(t%2?state0:state1) &&
              layer.convolution_history().data()==(t%2?history0:history1) &&
              layer.ple_convolution_history().data()==(t%2?ple0:ple1),"PLE/GDN reuse exactly two banks");
        const auto c = small_config();
        uint64_t context[3] = {uint64_t(tokens[t]), uint64_t(c.eos), uint64_t(c.eos)};
        if (previous_tokens[1] >= 0 && previous_tokens[1] != c.eos) {
            context[1] = uint64_t(previous_tokens[1]);
            if (previous_tokens[0] >= 0 && previous_tokens[0] != c.eos)
                context[2] = uint64_t(previous_tokens[0]);
        }
        for (size_t h = 0; h < 16; ++h) {
            uint64_t hash = context[0] * c.hash.mult[0] ^ context[1] * c.hash.mult[1];
            if (h >= 8)
                hash ^= context[2] * c.hash.mult[2];
            check(trace.ple_rows[h] == c.hash.offset[h] + hash % c.hash.vocab[h], "integrated token/EOS hash history");
        }
        previous_tokens = {{previous_tokens[1], tokens[t]}};
        check(layer.token_history() == previous_tokens, "committed token IDs");
        reference(f, f.input(int(t)), history, trace, layer);
        gdn_fixture::reference(f, state, gdn_history, trace, layer);
        if (t == 0)
            first = layer.last_residual();
        if (cpu) {
            PortableRoute cr;
            PortableGdnTrace ct;
            check(cpu->step(tokens[t], f.input(int(t)), experts(), cr, error, &ct), error);
            close(layer.last_residual(), cpu->last_residual(), 0.003f, "PLE full-layer CPU/Vulkan");
            close(layer.recurrent_state(), cpu->recurrent_state(), 0.003f, "PLE GDN state CPU/Vulkan");
            close(layer.ple_convolution_history(), cpu->ple_convolution_history(), 0.003f, "PLE history CPU/Vulkan");
            close(route.weights, cr.weights, 0.003f, "route CPU/Vulkan");
            check(route.ids == cr.ids, "route IDs CPU/Vulkan");
        }
        // Downstream rejection is attempted after each committed token. The
        // candidate token must not enter either token or normalized history.
        const auto previous = layer.token_history();
        history = layer.ple_convolution_history();
        state = layer.recurrent_state();
        gdn_history = layer.convolution_history();
        auto last = layer.last_residual();
        const auto *committed_state=layer.recurrent_state().data(), *committed_ple=layer.ple_convolution_history().data();
        for (int failure = 0; failure < 3; ++failure) {
            check(!layer.step(
                      999, f.input(10),
                      [failure](int64_t, const PortableRoute &, std::vector<float> &rows, std::string &e) {
                          if (failure == 1)
                              throw std::runtime_error("injected throw");
                          if (failure == 2) {
                              rows.assign(25600, NAN);
                              return true;
                          }
                          e = "injected failure";
                          return false;
                      },
                      route, error, &trace),
                  "injected callback failure");
            check(layer.token_history() == previous && layer.tokens() == t + 1, "token rollback");
            close(layer.ple_convolution_history(), history, 0, "PLE rollback");
            close(layer.recurrent_state(), state, 0, "GDN rollback");
            close(layer.convolution_history(), gdn_history, 0, "GDN convolution rollback");
            close(layer.last_residual(), last, 0, "residual rollback");
            check(route.ids.empty() && trace.ple_embedding.empty(), "failed output cleared");
            check(layer.recurrent_state().data()==committed_state &&
                  layer.ple_convolution_history().data()==committed_ple,"failed step published candidate storage");
        }
    }
    check(!layer.step(f.input(0), experts(), route, error), "layer1 requires token ID");
    check(layer.reset(error), error);
    check(layer.token_history() == std::array<int32_t, 2>{{-1, -1}} && layer.tokens() == 0, "reset tokens");
    for (float x : layer.ple_convolution_history())
        check(x == 0, "reset PLE history");
    check(layer.step(0, f.input(0), experts(), route, error), error);
    close(layer.last_residual(), first, 0.003f, "reset reproducibility");
}
} // namespace ple_fixture
