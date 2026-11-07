// Independent scalar history/tap contract for the retained native PLE stage.
// This does not exercise the Vulkan implementation.
#include <array>
#include <cassert>
#include <cstddef>
#include <cmath>
#include <iostream>

namespace {
constexpr size_t channels = 7;
using History = std::array<std::array<float, 9>, channels>;
using Row = std::array<float, channels>;
using Taps = std::array<float, 4>;
using Values = std::array<float, 4>; // preactivation, exponent, convolution, result

Values evaluate(const Taps& taps, const Taps& weights, float hidden, float gated) {
    const float t0 = taps[0] * weights[0], t1 = taps[1] * weights[1];
    const float t2 = taps[2] * weights[2], t3 = taps[3] * weights[3];
    const float pre = ((t0 + t1) + t2) + t3;
    const float exp = std::exp(-pre);
    const float conv = pre / (1.0f + exp);
    return {pre, exp, conv, hidden + (gated + conv)};
}

// A sequential T1 control, intentionally expressed as a history mutation.
Taps step(std::array<float, 9>& history, float current) {
    Taps taps{history[0], history[3], history[6], current};
    for (size_t j = 0; j < 8; ++j) history[j] = history[j + 1];
    history[8] = current;
    return taps;
}

// Direct full-T indexing against an immutable old history.
template<size_t T>
void full(const History& old, const std::array<Row, T>& rows,
          std::array<std::array<Taps, channels>, T>& taps, History& next) {
    static_assert(T == 1 || T == 2);
    for (size_t c = 0; c < channels; ++c) {
        for (size_t r = 0; r < T; ++r)
            taps[r][c] = {old[c][r], old[c][r + 3], old[c][r + 6], rows[r][c]};
        for (size_t j = 0; j < 9 - T; ++j) next[c][j] = old[c][j + T];
        for (size_t r = 0; r < T; ++r) next[c][9 - T + r] = rows[r][c];
    }
}
}

int main() {
    History old{};
    std::array<Row, 3> rows{};
    for (size_t c = 0; c < channels; ++c) {
        for (size_t j = 0; j < 9; ++j) old[c][j] = float(100 * c + j + 1);
        for (size_t r = 0; r < 3; ++r) rows[r][c] = float(1000 + 100 * r + c);
    }
    auto control = old;
    std::array<std::array<Taps, channels>, 3> expected{};
    for (size_t r = 0; r < 3; ++r)
        for (size_t c = 0; c < channels; ++c) expected[r][c] = step(control[c], rows[r][c]);

    std::array<Row, 2> pair{rows[0], rows[1]};
    std::array<std::array<Taps, channels>, 2> pair_taps{};
    History after_pair{};
    full(old, pair, pair_taps, after_pair);
    auto two_steps = old;
    for (size_t c = 0; c < channels; ++c) {
        assert(pair_taps[0][c] == expected[0][c]);
        assert(pair_taps[1][c] == expected[1][c]);
        (void) step(two_steps[c], rows[0][c]);
        (void) step(two_steps[c], rows[1][c]);
    }
    assert(after_pair == two_steps);
    std::array<Row, 1> tail{rows[2]};
    std::array<std::array<Taps, channels>, 1> tail_taps{};
    History after_tail{};
    full(after_pair, tail, tail_taps, after_tail);
    for (size_t c = 0; c < channels; ++c) assert(tail_taps[0][c] == expected[2][c]);
    assert(after_tail == control);

    // Distinct finite inputs for every row, channel and tap; compare all
    // intermediates as well as the result against the sequential T1 control.
    for (size_t r = 0; r < 3; ++r) for (size_t c = 0; c < channels; ++c) {
        Taps weights{0.0001f * float(c + 1), -0.0002f * float(c + 2),
                     0.0003f * float(c + 3), 0.0004f * float(c + 4)};
        float hidden = 0.125f * float(1 + 3*r + c);
        float gated = -0.0625f * float(2 + r + 2*c);
        const auto& actual_taps = r < 2 ? pair_taps[r][c] : tail_taps[0][c];
        auto actual = evaluate(actual_taps, weights, hidden, gated);
        auto reference = evaluate(expected[r][c], weights, hidden, gated);
        for (size_t i = 0; i < actual.size(); ++i) {
            assert(std::isfinite(actual[i]));
            assert(actual[i] == reference[i]);
        }
    }
    std::cout << "native PLE row history and scalar intermediates contract passed\n";
}
