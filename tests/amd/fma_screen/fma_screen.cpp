// Standalone installed-driver arithmetic discriminator. No ggml or model code.
#include <vulkan/vulkan.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr VkDeviceSize kBytes = 5 * sizeof(uint32_t);
constexpr uint64_t kFenceTimeout = 2'000'000'000;
struct Case { const char* name; std::array<uint32_t,3> input; uint32_t fused, separate; };
constexpr Case cases[] = {
    {"q8-positive", {0x3f804008,0x44802000,0xc4806000},0x3b401000,0x3b400000},
    {"q8-negative", {0x3f804008,0xc4802000,0x44806000},0xbb401000,0xbb400000},
    {"cancellation",{0x3f800001,0x3f7ffffe,0xbf800000},0xa8800000,0x00000000},
    {"control",     {0x3f800000,0x40000000,0x40400000},0x40a00000,0x40a00000},
};
void require(bool ok,const char* why) { if (!ok) throw std::runtime_error(why); }
void check(VkResult r,const char* action) {
    if (r != VK_SUCCESS) throw std::runtime_error(std::string(action)+" VkResult="+std::to_string(r));
}
std::string hex(const uint8_t* p,size_t n) {
    std::ostringstream s; s << std::hex << std::setfill('0');
    for(size_t i=0;i<n;++i) s << std::setw(2) << unsigned(p[i]);
    return s.str();
}
float as_float(uint32_t u) { float f; std::memcpy(&f,&u,4); return f; }
uint32_t as_bits(float f) { uint32_t u; std::memcpy(&u,&f,4); return u; }
void print_case(const Case& c,uint32_t f,uint32_t s,const char* source) {
    std::printf("%s case=%s a=%08x b=%08x c=%08x fma=%08x separate=%08x expected_fused_rne=%08x expected_separate_rne=%08x\n",
        source,c.name,c.input[0],c.input[1],c.input[2],f,s,c.fused,c.separate);
}
int host_check() {
    for(const auto& c:cases) {
        const float a=as_float(c.input[0]),b=as_float(c.input[1]),v=as_float(c.input[2]);
        volatile float product=a*b; // Preserve the separate rounding step.
        const uint32_t f=as_bits(std::fma(a,b,v)),s=as_bits(product+v);
        print_case(c,f,s,"host");
        require(f==c.fused && s==c.separate,"host oracle differs from constants");
    }
    return 0;
}
std::vector<uint32_t> shader(const std::string& path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    require(bool(f),"cannot open shader");
    const std::streamoff n=f.tellg();
    require(n>=20 && n<=65536 && n%4==0,"shader must be 20..65536 bytes and word aligned");
    std::vector<uint32_t> words(static_cast<size_t>(n)/4);
    f.seekg(0); f.read(reinterpret_cast<char*>(words.data()),n);
    require(bool(f) && words[0]==0x07230203 && words[1]<=0x00010500,"invalid/unsupported SPIR-V header");
    return words;
}
struct Instance {
    VkInstance h=VK_NULL_HANDLE;
    ~Instance() { if(h) vkDestroyInstance(h,nullptr); }
};
struct DeviceInfo {
    VkPhysicalDevice h;
    VkPhysicalDeviceProperties props{};
    VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceDriverProperties driver{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceFloatControlsProperties fp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES};
    std::vector<VkExtensionProperties> extensions;
    bool has(const char* name) const {
        for(const auto& x:extensions) if(std::strcmp(x.extensionName,name)==0) return true;
        return false;
    }
    explicit DeviceInfo(VkPhysicalDevice physical):h(physical) {
        uint32_t n=0; check(vkEnumerateDeviceExtensionProperties(h,nullptr,&n,nullptr),"count device extensions");
        require(n<=1024,"extension inventory bound"); extensions.resize(n);
        check(vkEnumerateDeviceExtensionProperties(h,nullptr,&n,extensions.data()),"read device extensions"); extensions.resize(n);
        VkPhysicalDeviceProperties2 p{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        vkGetPhysicalDeviceProperties(h,&props);
        p.pNext=&id;
        void** tail=&id.pNext;
        if(props.apiVersion>=VK_API_VERSION_1_2 || has(VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME)) {
            *tail=&driver; tail=&driver.pNext;
        }
        if(props.apiVersion>=VK_API_VERSION_1_2 || has(VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME)) *tail=&fp;
        vkGetPhysicalDeviceProperties2(h,&p); props=p.properties;
        // Do not retain pointers into a copied/moved DeviceInfo.
        id.pNext=driver.pNext=fp.pNext=nullptr;
    }
    void print(size_t ordinal) const {
        std::printf("device ordinal=%zu uuid=%s name=%s vendor=%04x device=%04x type=%u api=%u.%u.%u driverVersionRaw=%u\n",
            ordinal,hex(id.deviceUUID,VK_UUID_SIZE).c_str(),props.deviceName,props.vendorID,props.deviceID,unsigned(props.deviceType),
            VK_VERSION_MAJOR(props.apiVersion),VK_VERSION_MINOR(props.apiVersion),VK_VERSION_PATCH(props.apiVersion),props.driverVersion);
        std::printf("driver id=%u name=%s info=%s\n",unsigned(driver.driverID),driver.driverName,driver.driverInfo);
        std::printf("float32 rte=%u rtz=%u denormPreserve=%u denormFlushToZero=%u signedZeroInfNanPreserve=%u\n",
            fp.shaderRoundingModeRTEFloat32,fp.shaderRoundingModeRTZFloat32,fp.shaderDenormPreserveFloat32,
            fp.shaderDenormFlushToZeroFloat32,fp.shaderSignedZeroInfNanPreserveFloat32);
        const bool fc2=has("VK_KHR_shader_float_controls2"),fma=has("VK_KHR_shader_fma");
        std::printf("extensions VK_KHR_shader_float_controls2=%u VK_KHR_shader_fma=%u\n",unsigned(fc2),unsigned(fma));
        if(has(VK_EXT_PCI_BUS_INFO_EXTENSION_NAME)) {
            VkPhysicalDevicePCIBusInfoPropertiesEXT pci{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT};
            VkPhysicalDeviceProperties2 p{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2}; p.pNext=&pci;
            vkGetPhysicalDeviceProperties2(h,&p);
            std::printf("pci=%04x:%02x:%02x.%x\n",pci.pciDomain,pci.pciBus,pci.pciDevice,pci.pciFunction);
        } else std::puts("pci=unavailable");
#ifdef VK_KHR_shader_float_controls2
        if(fc2 || props.apiVersion>=VK_MAKE_VERSION(1,4,0)) {
            VkPhysicalDeviceShaderFloatControls2FeaturesKHR feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT_CONTROLS_2_FEATURES_KHR};
            VkPhysicalDeviceFeatures2 f{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; f.pNext=&feature;
            vkGetPhysicalDeviceFeatures2(h,&f);
            std::printf("feature shaderFloatControls2=%u\n",feature.shaderFloatControls2);
        } else std::puts("feature shaderFloatControls2=unsupported");
#else
        std::puts("feature shaderFloatControls2=query_unavailable_in_build_headers");
#endif
#ifdef VK_KHR_shader_fma
        if(fma) {
            VkPhysicalDeviceShaderFmaFeaturesKHR feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FMA_FEATURES_KHR};
            VkPhysicalDeviceFeatures2 f{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; f.pNext=&feature;
            vkGetPhysicalDeviceFeatures2(h,&f);
            std::printf("feature shaderFmaFloat32=%u\n",feature.shaderFmaFloat32);
        } else std::puts("feature shaderFmaFloat32=unsupported");
#else
        std::puts("feature shaderFmaFloat32=query_unavailable_in_build_headers");
#endif
        std::puts("screen requested_float_controls2=0 requested_rte=0 requested_shader_fma=0 (unchanged legacy module)");
    }
};
struct Screen {
    VkDevice d=VK_NULL_HANDLE;
    VkBuffer buffer=VK_NULL_HANDLE;
    VkDeviceMemory memory=VK_NULL_HANDLE;
    void* mapped=nullptr;
    VkDescriptorSetLayout set_layout=VK_NULL_HANDLE;
    VkPipelineLayout layout=VK_NULL_HANDLE;
    VkShaderModule module=VK_NULL_HANDLE;
    VkPipeline pipeline=VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool=VK_NULL_HANDLE;
    VkCommandPool command_pool=VK_NULL_HANDLE;
    VkFence fence=VK_NULL_HANDLE;
    ~Screen() {
        // Every successful submission is fenced before normal cleanup. A fence
        // failure terminates the process instead of freeing in-flight objects.
        if(!d) return;
        if(fence) vkDestroyFence(d,fence,nullptr);
        if(command_pool) vkDestroyCommandPool(d,command_pool,nullptr);
        if(descriptor_pool) vkDestroyDescriptorPool(d,descriptor_pool,nullptr);
        if(pipeline) vkDestroyPipeline(d,pipeline,nullptr);
        if(module) vkDestroyShaderModule(d,module,nullptr);
        if(layout) vkDestroyPipelineLayout(d,layout,nullptr);
        if(set_layout) vkDestroyDescriptorSetLayout(d,set_layout,nullptr);
        if(mapped) vkUnmapMemory(d,memory);
        if(buffer) vkDestroyBuffer(d,buffer,nullptr);
        if(memory) vkFreeMemory(d,memory,nullptr);
        vkDestroyDevice(d,nullptr);
    }
    int run(const DeviceInfo& info,const std::vector<uint32_t>& words) {
        uint32_t n=0; vkGetPhysicalDeviceQueueFamilyProperties(info.h,&n,nullptr);
        require(n>0 && n<=64,"queue family inventory bound");
        std::vector<VkQueueFamilyProperties> families(n); vkGetPhysicalDeviceQueueFamilyProperties(info.h,&n,families.data());
        uint32_t family=0;
        while(family<n && !(families[family].queueFlags&VK_QUEUE_COMPUTE_BIT)) ++family;
        require(family<n,"no compute queue");
        const float priority=1;
        VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qi.queueFamilyIndex=family; qi.queueCount=1; qi.pQueuePriorities=&priority;
        VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; di.queueCreateInfoCount=1; di.pQueueCreateInfos=&qi;
        check(vkCreateDevice(info.h,&di,nullptr,&d),"create device");
        VkQueue queue; vkGetDeviceQueue(d,family,0,&queue);
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size=kBytes; bi.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        check(vkCreateBuffer(d,&bi,nullptr,&buffer),"create buffer");
        VkMemoryRequirements req; vkGetBufferMemoryRequirements(d,buffer,&req);
        require(req.size>0 && req.size<=1024*1024,"buffer allocation exceeds 1 MiB");
        VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(info.h,&mp);
        uint32_t type=0; const auto flags=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        while(type<mp.memoryTypeCount && (!(req.memoryTypeBits&(1u<<type)) || (mp.memoryTypes[type].propertyFlags&flags)!=flags)) ++type;
        require(type<mp.memoryTypeCount,"no host-visible coherent buffer memory; stop");
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize=req.size; ai.memoryTypeIndex=type;
        check(vkAllocateMemory(d,&ai,nullptr,&memory),"allocate buffer memory");
        check(vkBindBufferMemory(d,buffer,memory,0),"bind buffer memory");
        check(vkMapMemory(d,memory,0,VK_WHOLE_SIZE,0,&mapped),"map buffer");
        std::printf("resources buffer_bytes=%llu allocation_bytes=%llu dispatches=4 invocations_per_dispatch=1 fence_timeout_ms=2000\n",
            static_cast<unsigned long long>(kBytes),static_cast<unsigned long long>(req.size));
        VkDescriptorSetLayoutBinding binding{}; binding.binding=0; binding.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; binding.descriptorCount=1; binding.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo si{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; si.bindingCount=1; si.pBindings=&binding;
        check(vkCreateDescriptorSetLayout(d,&si,nullptr,&set_layout),"create descriptor layout");
        VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; li.setLayoutCount=1; li.pSetLayouts=&set_layout;
        check(vkCreatePipelineLayout(d,&li,nullptr,&layout),"create pipeline layout");
        VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; mi.codeSize=words.size()*4; mi.pCode=words.data();
        check(vkCreateShaderModule(d,&mi,nullptr,&module),"create shader module");
        VkComputePipelineCreateInfo pi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; pi.layout=layout;
        pi.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; pi.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT; pi.stage.module=module; pi.stage.pName="main";
        check(vkCreateComputePipelines(d,VK_NULL_HANDLE,1,&pi,nullptr,&pipeline),"create compute pipeline");
        VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1};
        VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pool.maxSets=1; pool.poolSizeCount=1; pool.pPoolSizes=&size;
        check(vkCreateDescriptorPool(d,&pool,nullptr,&descriptor_pool),"create descriptor pool");
        VkDescriptorSetAllocateInfo sai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; sai.descriptorPool=descriptor_pool; sai.descriptorSetCount=1; sai.pSetLayouts=&set_layout;
        VkDescriptorSet set; check(vkAllocateDescriptorSets(d,&sai,&set),"allocate descriptor set");
        VkDescriptorBufferInfo db{buffer,0,kBytes};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; write.dstSet=set; write.dstBinding=0; write.descriptorCount=1; write.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; write.pBufferInfo=&db;
        vkUpdateDescriptorSets(d,1,&write,0,nullptr);
        VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; ci.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; ci.queueFamilyIndex=family;
        check(vkCreateCommandPool(d,&ci,nullptr,&command_pool),"create command pool");
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cai.commandPool=command_pool; cai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount=1;
        VkCommandBuffer cb; check(vkAllocateCommandBuffers(d,&cai,&cb),"allocate command buffer");
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; check(vkCreateFence(d,&fi,nullptr,&fence),"create fence");
        bool fused_match=true,separate_match=true,split_match=true;
        for(const auto& c:cases) {
            const uint32_t input[5]={c.input[0],c.input[1],c.input[2],0x7fc12345,0x7fc54321};
            std::memcpy(mapped,input,sizeof(input));
            check(vkResetCommandBuffer(cb,0),"reset command buffer");
            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; begin.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            check(vkBeginCommandBuffer(cb,&begin),"begin command buffer");
            vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);
            vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,layout,0,1,&set,0,nullptr);
            vkCmdDispatch(cb,1,1,1);
            VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; barrier.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&barrier,0,nullptr,0,nullptr);
            check(vkEndCommandBuffer(cb),"end command buffer");
            check(vkResetFences(d,1,&fence),"reset fence");
            VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount=1; submit.pCommandBuffers=&cb;
            check(vkQueueSubmit(queue,1,&submit,fence),"submit one invocation");
            const VkResult wait=vkWaitForFences(d,1,&fence,VK_TRUE,kFenceTimeout);
            if(wait!=VK_SUCCESS) {
                std::fprintf(stderr,"fence failed VkResult=%d; exiting without destroying possibly in-flight objects\n",int(wait));
                std::fflush(nullptr); std::_Exit(3);
            }
            uint32_t output[5]; std::memcpy(output,mapped,sizeof(output));
            require(std::memcmp(input,output,3*sizeof(uint32_t))==0,"shader modified input words");
            print_case(c,output[3],output[4],"gpu");
            fused_match &= output[3]==c.fused;
            split_match &= output[3]==c.separate;
            separate_match &= output[4]==c.separate;
        }
        std::printf("screen fused_rne_all=%u separate_rne_all=%u legacy_matches_split_all=%u\n",unsigned(fused_match),unsigned(separate_match),unsigned(split_match));
        std::puts("scope=four-case-screen-only; no real-expert or universal exactness claim");
        return fused_match && separate_match ? 0 : 2;
    }
};
void usage() {
    std::puts("Usage: fma-screen --host-check\n       fma-screen --list\n       fma-screen --device-uuid <32 lowercase hex digits> --shader <fma_probe.spv>\nExit: 0 screen matches fused/separate RNE (or successful inventory/host-check), 2 mismatch, 3 fence failure, 1 error.");
}
} // namespace
int main(int argc,char** argv) {
    try {
        if(argc==2 && std::string(argv[1])=="--host-check") return host_check();
        if(argc==2 && std::string(argv[1])=="--help") { usage(); return 0; }
        const bool list=argc==2 && std::string(argv[1])=="--list";
        std::string uuid,path;
        if(!list) {
            require(argc==5 && std::string(argv[1])=="--device-uuid" && std::string(argv[3])=="--shader","explicit UUID and shader required; use --help");
            uuid=argv[2]; path=argv[4];
            require(uuid.size()==32 && uuid.find_first_not_of("0123456789abcdef")==std::string::npos,"UUID must be exactly 32 lowercase hexadecimal digits");
        }
        const auto words=list?std::vector<uint32_t>{}:shader(path); // Validate before any Vulkan call.
        uint32_t loader=0; check(vkEnumerateInstanceVersion(&loader),"query Vulkan loader");
        require(loader>=VK_API_VERSION_1_2,"Vulkan 1.2 loader required");
        std::printf("build Vulkan_header_version=%u loader=%u.%u.%u\n",VK_HEADER_VERSION,VK_VERSION_MAJOR(loader),VK_VERSION_MINOR(loader),VK_VERSION_PATCH(loader));
        Instance instance;
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.pApplicationName="Strata FMA four-case screen"; app.apiVersion=VK_API_VERSION_1_2;
        VkInstanceCreateInfo ii{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ii.pApplicationInfo=&app;
        check(vkCreateInstance(&ii,nullptr,&instance.h),"create instance");
        uint32_t n=0; check(vkEnumeratePhysicalDevices(instance.h,&n,nullptr),"count physical devices");
        require(n>0 && n<=64,"physical device inventory bound");
        std::vector<VkPhysicalDevice> devices(n); check(vkEnumeratePhysicalDevices(instance.h,&n,devices.data()),"enumerate physical devices");
        size_t matches=0,ordinal=0; VkPhysicalDevice selected=VK_NULL_HANDLE;
        for(uint32_t i=0;i<n;++i) {
            DeviceInfo info(devices[i]);
            if(list) info.print(i);
            else if(hex(info.id.deviceUUID,VK_UUID_SIZE)==uuid) { selected=devices[i]; ordinal=i; ++matches; }
        }
        if(list) { std::puts("inventory_only=1 dispatches=0"); return 0; }
        require(matches==1,"device UUID must match exactly one device");
        DeviceInfo info(selected); info.print(ordinal);
        require(info.props.apiVersion>=VK_API_VERSION_1_2,"Vulkan 1.2 device required");
        require(info.props.deviceType==VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && info.props.vendorID==0x1002 && info.props.deviceID==0x6864,
                "screen requires the explicitly selected physical AMD V340 (1002:6864)");
        std::printf("shader path=%s bytes=%zu\n",path.c_str(),words.size()*4); std::fflush(stdout);
        Screen screen; return screen.run(info,words);
    } catch(const std::exception& e) { std::fprintf(stderr,"error: %s\n",e.what()); return 1; }
}
