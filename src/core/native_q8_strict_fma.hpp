#pragma once
#include <cstdint>
#include <cstdio>
#include <stdexcept>
namespace strata::core::portable_detail {
// Uses the properties already queried for this same physical device. Emit
// before admission so a missing extension has an attributable identity.
inline void log_native_q8_fma_device(std::FILE* out,size_t index,size_t physical_index,
    const char* name,uint32_t vendor,uint32_t device,uint32_t driver,uint32_t api,
    const uint8_t* uuid,bool extension,bool fp32) {
    char identity[33];
    static constexpr char hex[]="0123456789abcdef";
    for(size_t i=0;i<16;++i){identity[2*i]=hex[uuid[i]>>4];identity[2*i+1]=hex[uuid[i]&15];}
    identity[32]='\0';
    std::fprintf(out,"native_q8_fma_device index=%zu physical_index=%zu name=\"%s\" vendor=0x%04x device=0x%04x driver_version=0x%08x api_version=0x%08x uuid=%s extension=%d fp32_queried=%d fp32=%d\n",
                 index,physical_index,name,vendor,device,driver,api,identity,int(extension),int(extension),int(fp32));
}
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
