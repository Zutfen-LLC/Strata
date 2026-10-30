#pragma once
#include <cstdint>
#include <sstream>
#include <stdexcept>
namespace strata::core::portable_detail {
// Failure-only diagnostics for the existing SPIR-V admission decision. Raw
// VkBool32 fields are reported separately; no arithmetic policy is selected.
template<class FloatControls, class Properties>
void native_q8_require_float_controls(const FloatControls& fp,const Properties& properties) {
    if(fp.shaderDenormPreserveFloat32 && fp.shaderSignedZeroInfNanPreserveFloat32 &&
       fp.shaderRoundingModeRTEFloat32) return;
    std::ostringstream out;
    out<<"native stages: mandatory FP32 preserve/RTE controls unavailable; no relaxed fallback"
       <<" shaderDenormPreserveFloat32="<<fp.shaderDenormPreserveFloat32
       <<" shaderSignedZeroInfNanPreserveFloat32="<<fp.shaderSignedZeroInfNanPreserveFloat32
       <<" shaderRoundingModeRTEFloat32="<<fp.shaderRoundingModeRTEFloat32
       <<" shaderDenormFlushToZeroFloat32="<<fp.shaderDenormFlushToZeroFloat32
       <<" shaderRoundingModeRTZFloat32="<<fp.shaderRoundingModeRTZFloat32
       <<" denormBehaviorIndependence="<<uint32_t(fp.denormBehaviorIndependence)
       <<" roundingModeIndependence="<<uint32_t(fp.roundingModeIndependence)
       <<" shaderDenormPreserveFloat16="<<fp.shaderDenormPreserveFloat16
       <<" shaderRoundingModeRTEFloat16="<<fp.shaderRoundingModeRTEFloat16
       <<" apiVersion="<<properties.apiVersion<<" driverVersion="<<properties.driverVersion
       <<" vendorID="<<properties.vendorID<<" deviceID="<<properties.deviceID
       <<" deviceName="<<properties.deviceName.data();
    throw std::runtime_error(out.str());
}
}
