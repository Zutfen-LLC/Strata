#pragma once
#include <sstream>
#include <stdexcept>
namespace strata::core::portable_detail {
// Logical-device creation receipt, separate from GGML's acceleration selector.
struct NativeQ8IntegerDotState {
    bool requested=false;
    bool supported=false;
    bool enabled=false;
    void logical_device_created(bool queried_support) {
        supported=requested && queried_support;
        enabled=supported; // called only AFTER successful vkCreateDevice
    }
    void require_enabled(bool accelerated) const {
        if(requested && supported && enabled)return;
        std::ostringstream out;
        out<<"Q8 component: functional integer dot not enabled; requested="<<requested
           <<" queried_support="<<supported<<" logical_enabled="<<enabled
           <<" ggml_accelerated="<<accelerated;
        throw std::runtime_error(out.str());
    }
};
}
