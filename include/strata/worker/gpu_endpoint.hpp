#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::worker {
inline constexpr size_t kMaxGpuWorkers = 8;
struct GpuEndpoint {
    std::string host, plan;
    uint16_t port = 0;
    bool vulkan = false; // explicit per-endpoint opt-in; CUDA remains the default
};
inline bool gpu_numeric_ipv4(const std::string& host) {
    size_t start=0;
    for(int part=0;part<4;++part) {
        const size_t end=part==3 ? host.size() : host.find('.',start);
        if(end==std::string::npos || end<=start || end-start>3 || (end-start>1 && host[start]=='0'))return false;
        unsigned value=0;
        for(size_t i=start;i<end;++i) {if(host[i]<'0'||host[i]>'9')return false;value=value*10+unsigned(host[i]-'0');}
        if(value>255)return false;
        start=end+1;
    }
    return true;
}
// Match the existing controlled connector's numeric IPv4 contract. No DNS/new route.
inline bool parse_gpu_endpoint(const std::string& text, GpuEndpoint& out) {
    std::string address=text;
    bool vulkan=false;
    if(address.rfind("vulkan@",0)==0) {vulkan=true;address.erase(0,7);}
    else if(address.rfind("cuda@",0)==0) address.erase(0,5);
    const auto colon = address.rfind(':');
    if (colon == std::string::npos || !colon || colon+1 == address.size()) return false;
    unsigned port = 0;
    for (size_t i=colon+1; i<address.size(); ++i) {
        if (address[i]<'0' || address[i]>'9') return false;
        port = port*10 + unsigned(address[i]-'0');
        if (port>65535) return false;
    }
    if (!port) return false;
    if (!gpu_numeric_ipv4(address.substr(0,colon))) return false;
    out.vulkan=vulkan; out.host=address.substr(0,colon); out.port=(uint16_t)port; return true;
}
} // namespace strata::worker
