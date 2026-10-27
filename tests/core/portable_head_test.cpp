#include "strata/artifact/dequant.hpp"
#include "strata/core/portable_model.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
using namespace strata::core;
namespace {
void check(bool ok, const std::string &e) {
    if (!ok)
        throw std::runtime_error(e);
}
} // namespace
int main() {
    try {
        constexpr size_t H = 32, HC = 4, R = 32, V = 9;
        std::vector<float> norm(H * HC);
        for (size_t i = 0; i < norm.size(); ++i)
            norm[i] = 0.7f + float(i % 7) * 0.1f;
        std::vector<std::vector<uint8_t>> data(3);
        std::vector<std::vector<float>> decoded(3);
        const size_t inputs[]{H * HC, R, H}, outputs[]{R, H * HC, V};
        for (size_t w = 0; w < 3; ++w) {
            data[w].resize(inputs[w] * outputs[w] / 32 * 34);
            decoded[w].resize(inputs[w] * outputs[w]);
            for (size_t b = 0; b < data[w].size() / 34; ++b) {
                const uint16_t scale = 0x2800; // exactly 1/32
                std::memcpy(data[w].data() + b * 34, &scale, 2);
                for (size_t i = 0; i < 32; ++i) {
                    const int8_t code = int8_t(int((i * 3 + b * 7 + w) % 11) - 5);
                    std::memcpy(data[w].data() + b * 34 + 2 + i, &code, 1);
                }
                strata::dequantize_q8_0(data[w].data() + b * 34, decoded[w].data() + b * 32);
            }
        }
        std::vector<PortableTensorView> weights{
            {"output_hc_norm.weight", {H * HC}, 0, norm.data(), norm.size() * 4},
            {"output_hc_down.weight", {H * HC, R}, 8, data[0].data(), data[0].size()},
            {"output_hc_up.weight", {R, H * HC}, 8, data[1].data(), data[1].size()},
            {"output.weight", {H, V}, 8, data[2].data(), data[2].size()}};
        std::string error;
        auto head = PortableHead::create({H, HC, R, V, 1e-6f}, weights, {}, error);
        check(bool(head), error);
        auto mm = [&](size_t w, const std::vector<float> &x) {
            std::vector<float> result(outputs[w]);
            for (size_t r = 0; r < result.size(); ++r) {
                double sum = 0;
                for (size_t i = 0; i < x.size(); ++i)
                    sum += double(decoded[w][r * x.size() + i]) * x[i];
                result[r] = float(sum);
            }
            return result;
        };
        for (size_t token = 0; token < 3; ++token) {
            std::vector<float> input(H * HC), normalized(H * HC), logits;
            for (size_t i = 0; i < input.size(); ++i)
                input[i] = float(int((i * 13 + token * 7) % 29) - 14) * 0.1f;
            for (size_t s = 0; s < HC; ++s) {
                double sq = 0;
                for (size_t i = 0; i < H; ++i)
                    sq += double(input[s * H + i]) * input[s * H + i];
                const float inv = 1 / std::sqrt(float(sq / H) + 1e-6f);
                for (size_t i = 0; i < H; ++i)
                    normalized[s * H + i] = input[s * H + i] * inv * norm[s * H + i];
            }
            auto low = mm(0, normalized);
            for (float &v : low) {
                v *= 0.25f;
                v = v / (1 + std::exp(-v));
            }
            const auto gates = mm(1, low);
            std::vector<float> mixed(H, 0);
            for (size_t s = 0; s < HC; ++s)
                for (size_t i = 0; i < H; ++i)
                    mixed[i] += normalized[s * H + i] / (1 + std::exp(-gates[s * H + i])) * 0.25f;
            const auto expected = mm(2, mixed);
            check(head->run(input, logits, error), error);
            for (size_t i = 0; i < V; ++i)
                check(std::fabs(logits[i] - expected[i]) <= 0.012f * (1 + std::fabs(expected[i])),
                      "independent HC/head scalar reference");
            // ggml Q8 matvec quantizes activation; independent F32 dots above
            // use an a priori screening tolerance, not an exact-bit claim.
        }
        std::vector<float> out;
        check(!head->run({}, out, error) && out.empty(), "head input rejection");
        auto small = PortableFfnOptions{};
        small.weight_budget = 1;
        check(!PortableHead::create({H, HC, R, V, 1e-6f}, weights, small, error), "head weight cap");
        small = {};
        small.graph_budget = 1;
        check(!PortableHead::create({H, HC, R, V, 1e-6f}, weights, small, error), "head graph cap");
        const auto memory = head->memory_usage();
        check(memory.host_weights == head->weight_bytes() && memory.device_graph == 0 && memory.host_graph > 0,
              "head allocation report");
        std::puts("portable_head_test: independent Q8 HC/global head scalar reference, input/cap checks passed");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
