#include "native_qsa_qkv_layout.hpp"
#include "native_qsa_wo_domain.hpp"
#include <cassert>
#include <stdexcept>
#include <limits>
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
    std::array<uint8_t,2880> packed{};
    for (unsigned h=1;h<=1023;++h) {
        packed[0]=uint8_t(h);packed[1]=uint8_t(h>>8);
        qsa_qkv_packed_domain(packed.data(),packed.size());
    }
    for (uint16_t h : {uint16_t(0x7c00),uint16_t(0xfc00),uint16_t(0x7e00)}) {
        packed[0]=uint8_t(h);packed[1]=uint8_t(h>>8);refused=false;
        try { qsa_qkv_packed_domain(packed.data(),packed.size()); }
        catch (const std::runtime_error &e) { refused=std::string(e.what()).find("QSA QKV:")==0 && std::string(e.what()).find("blocks=80")!=std::string::npos; }
        assert(refused);
    }
    std::array<uint8_t,192*36> wo{};wo[0]=1;refused=false;
    try { qsa_wo_packed_domain(wo.data(),wo.size()); } catch (const std::runtime_error &) { refused=true; }
    assert(refused); // This slice does not widen the separate WO producer domain.
    refused=false;
    try { qsa_q8_input_domain(std::vector<float>(2560,std::numeric_limits<float>::infinity()),80,"QSA QKV"); }
    catch (const std::runtime_error &e) { refused=std::string(e.what()).find("QSA QKV:")==0; }
    assert(refused);
    refused = false;
    try { (void) qsa_qkv_query_row(24, 0); } catch (const std::runtime_error &) { refused = true; }
    assert(refused);
}
