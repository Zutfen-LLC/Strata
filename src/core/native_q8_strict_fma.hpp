#pragma once
#include <stdexcept>
namespace strata::core::portable_detail {
// Only the explicit strict-FMA build requests this feature. A physical query
// alone never establishes logical-device enablement.
struct NativeQ8StrictFmaState {
    bool requested=false, supported=false, enabled=false;
    void queried(bool extension,bool fp32) {
        requested=true;
        enabled=false;
        supported=extension && fp32;
        if(!extension)throw std::runtime_error("native Q8 strict FMA: VK_KHR_shader_fma not advertised");
        if(!fp32)throw std::runtime_error("native Q8 strict FMA: shaderFmaFloat32 unsupported");
    }
    void logical_device_created() {enabled=requested && supported;}
    void require_enabled() const {
        if(!(requested && supported && enabled))throw std::runtime_error("native Q8 strict FMA: logical shaderFmaFloat32 not enabled");
    }
};
}
