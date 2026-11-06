#pragma once
#include "native_qsa_wo_domain.hpp"
#include <array>
#include <cstring>
#include <string>
#include <vector>

// These are the same domain functions used by PortableNativeGdnInput. They
// check one block at a time for diagnostic attribution; a passing block is not
// a full 80-block model invocation or a numerical qualification.
inline std::string boundary_gdn_input_error(const unsigned char* bytes) {
    std::vector<float> x(32);std::memcpy(x.data(),bytes,32*sizeof(float));
    try {strata::core::portable_detail::qsa_q8_input_domain(x,1);return {};}
    catch(const std::runtime_error& e) {return e.what();}
}
inline std::string boundary_gdn_scale_error(const unsigned char* packed) {
    using namespace strata::core::portable_detail;
    try {qsa_wo_require_scales(qsa_wo_scale_classes(packed,1,36),"producer Q8_1");return {};}
    catch(const std::runtime_error& e) {return e.what();}
}
