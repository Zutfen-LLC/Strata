#include "native_head_vocab_layout.hpp"
#include <cassert>
#include <cstdio>
#include <initializer_list>
#include <stdexcept>
using namespace strata::core::portable_detail;
int main() {
    for (size_t rows : {size_t(1), head_vocab_slice_rows - 1, head_vocab_slice_rows})
        head_vocab_require_layout(rows, rows * head_vocab_row_bytes, head_vocab_packed_bytes, rows * 4);
    for (size_t rows : {size_t(0), head_vocab_slice_rows + 1}) {
        bool refused = false;
        try {
            head_vocab_require_layout(rows, rows * head_vocab_row_bytes, head_vocab_packed_bytes, rows * 4);
        } catch (const std::runtime_error &) {
            refused = true;
        }
        assert(refused);
    }
    bool refused = false;
    try {
        head_vocab_require_layout(1, head_vocab_row_bytes, head_vocab_packed_bytes - 4, 4);
    } catch (const std::runtime_error &) {
        refused = true;
    }
    assert(refused);
    // Qwen's vocabulary spans full slices and a nonempty final slice.
    const size_t vocabulary = 248320, full = vocabulary / head_vocab_slice_rows,
                 last = vocabulary % head_vocab_slice_rows;
    assert(full == 15 && last == 2560 && full * head_vocab_slice_rows + last == vocabulary);
    std::puts("head vocabulary row-slice boundaries and exact byte layouts pass");
}
