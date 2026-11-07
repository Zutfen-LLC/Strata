// CPU layout and tap control for native_gdn_conv_rows.comp. Does not run Vulkan.
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
constexpr size_t channels = 10240;
using History = std::vector<std::array<float, 3>>;
using Row = std::vector<float>;
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
float convolution(const std::array<float, 3>& h, float x, size_t c) {
    const float w0 = 0.001f * float(1 + c % 7), w1 = -0.002f;
    const float w2 = 0.003f, w3 = -0.004f;
    const float second = h[1] * w1;
    float v = std::fma(h[0], w0, second);
    v = std::fma(h[2], w2, v);
    return std::fma(x, w3, v);
}
// Sequential T1 mutation, independent of the row indexing below.
Row step(History& h, const Row& x) {
    Row pre(channels);
    for (size_t c = 0; c < channels; ++c) {
        pre[c] = convolution(h[c], x[c], c);
        h[c] = {h[c][1], h[c][2], x[c]};
    }
    return pre;
}
// Direct immutable-history indexing used by the row shader.
std::vector<Row> rows(const History& old, const std::vector<Row>& x, History& next) {
    check(x.size() == 1 || x.size() == 2, "row count");
    std::vector<Row> pre(x.size(), Row(channels));
    next.resize(channels);
    for (size_t c = 0; c < channels; ++c) {
        for (size_t r = 0; r < x.size(); ++r) {
            std::array<float, 3> taps = r == 0 ? old[c] :
                std::array<float, 3>{old[c][1], old[c][2], x[0][c]};
            pre[r][c] = convolution(taps, x[r][c], c);
            if (r + 1 == x.size()) next[c] = {taps[1], taps[2], x[r][c]};
        }
    }
    return pre;
}
void exercise(float seed) {
    History initial(channels);
    std::array<Row, 5> input;
    for (auto& x : input) x.resize(channels);
    for (size_t c = 0; c < channels; ++c) {
        initial[c] = {seed + float(c % 11) / 16, seed - 0.5f, seed + 0.25f};
        for (size_t r = 0; r < input.size(); ++r)
            input[r][c] = float(1 + r) / 8 + float(c % 19) / 32;
    }
    auto serial = initial;
    std::array<Row, 5> expected;
    for (size_t r = 0; r < input.size(); ++r) expected[r] = step(serial, input[r]);
    auto current = initial;
    for (size_t start : {size_t(0), size_t(2), size_t(4)}) {
        const size_t count = start == 4 ? 1 : 2;
        std::vector<Row> batch(input.begin() + start, input.begin() + start + count);
        History next;
        auto actual = rows(current, batch, next);
        for (size_t r = 0; r < count; ++r) check(actual[r] == expected[start + r], "T2/T1 pre mismatch");
        current.swap(next);
    }
    check(current == serial, "T2 plus tail final history mismatch");
    History first;
    auto one = rows(initial, {input[0]}, first);
    check(one[0] == expected[0], "T1 pre mismatch");
    check(first[0][0] == initial[0][1] && first[0][1] == initial[0][2] && first[0][2] == input[0][0], "T1 retains old history");
}
}
int main() {
    try {
        exercise(1.25f); exercise(-0.75f); exercise(1.25f); // reset A/B/A
        std::cout << "GDN causal convolution row indexing control passed\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
