#include "native_qsa_qkv_layout.hpp"
#include <cassert>
#include <stdexcept>
using namespace strata::core::portable_detail;
int main() {
    assert(qsa_qkv_blocks == 80 && qsa_qkv_packed_bytes == 2880 && qsa_qkv_row_bytes == 2720);
    qsa_qkv_require_layout(12288, 12288u * 2720u, 2880, 12288u * 4u);
    qsa_qkv_require_layout(512, 512u * 2720u, 2880, 512u * 4u);
    for (size_t h = 0; h < 24; ++h) {
        assert(qsa_qkv_query_row(h, 0) == h * 512);
        assert(qsa_qkv_query_row(h, 255) == h * 512 + 255);
        assert(qsa_qkv_gate_row(h, 0) == h * 512 + 256);
        assert(qsa_qkv_gate_row(h, 255) == h * 512 + 511);
    }
    bool refused = false;
    try { qsa_qkv_require_layout(512, 512u * 2720u, 2879, 512u * 4u); }
    catch (const std::runtime_error &) { refused = true; }
    assert(refused);
    refused = false;
    try { (void) qsa_qkv_query_row(24, 0); } catch (const std::runtime_error &) { refused = true; }
    assert(refused);
}
