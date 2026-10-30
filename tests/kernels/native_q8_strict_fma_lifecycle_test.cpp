#include <vulkan/vulkan.h>
#include <algorithm>
#include <cstring>
#include <memory>
#include <cstdio>
#include <stdexcept>
#include <vector>
// These counters observe actual shared_ptr cache ownership and execution of the
// pinned destructor body. There are no Vulkan objects or driver calls.
struct Counters {int constructed=0,destroyed=0,queries=0,creates=0,destroys=0,invalid=0;} counts;
bool extension_available=true,feature_available=true,query_success=true,create_success=true,fail_after_create=false;
struct Extension {const char* extensionName;};
struct FakeCreateInfo {VkPhysicalDeviceFeatures2* features;std::vector<const char*>* extensions;};
struct LogicalDevice {
    bool created=false;
    explicit operator bool() const{return created;}
    void destroyFence(int){++counts.destroys;if(!created)++counts.invalid;}
    void destroyDescriptorSetLayout(int){++counts.destroys;if(!created)++counts.invalid;}
    void destroy(){++counts.destroys;if(!created)++counts.invalid;created=false;}
};
struct PhysicalDevice {
    LogicalDevice createDevice(const FakeCreateInfo& info) {
        ++counts.creates;
        auto* fma=static_cast<VkPhysicalDeviceShaderFmaFeaturesKHR*>(info.features->pNext);
        if(!fma || fma->sType!=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FMA_FEATURES_KHR ||
            !fma->shaderFmaFloat32 || fma->shaderFmaFloat16 || fma->shaderFmaFloat64 ||
            info.extensions->size()!=1 || strcmp(info.extensions->front(),VK_KHR_SHADER_FMA_EXTENSION_NAME))
            throw std::runtime_error("invalid actual FP32-only creation chain");
        if(!create_success)throw std::runtime_error("mock vkCreateDevice failure");
        return {true};
    }
};
void vkGetPhysicalDeviceFeatures2(PhysicalDevice,VkPhysicalDeviceFeatures2* info) {
    ++counts.queries;
    if(!query_success)throw std::runtime_error("mock query failure");
    if(auto* fma=static_cast<VkPhysicalDeviceShaderFmaFeaturesKHR*>(info->pNext)) {
        fma->shaderFmaFloat32=feature_available;
        fma->shaderFmaFloat16=fma->shaderFmaFloat64=VK_TRUE;
    }
}
struct Queue {struct Pool {void destroy(LogicalDevice&) {}} cmd_pool;};
struct Pipeline {};
using vk_pipeline=std::shared_ptr<Pipeline>;
void ggml_vk_destroy_pipeline(LogicalDevice&,vk_pipeline) {}
void ggml_vk_destroy_buffer(int) {}
struct Lifetime {~Lifetime(){++counts.destroyed;}};
struct vk_device_struct {
    vk_device_struct(){++counts.constructed;}
    ~vk_device_struct();
    Lifetime lifetime;
    LogicalDevice device;
    PhysicalDevice physical_device;
    int fence=0,sync_staging=0,dsl=0;
    std::shared_ptr<Queue> compute_queue,transfer_queue;
    std::vector<std::weak_ptr<Pipeline>> all_pipelines;
};
#define STRATA_NATIVE_Q8_STRICT_FMA 1
#include "native_q8_vk_device.hpp"
using vk_device=std::shared_ptr<vk_device_struct>;
struct Instance {std::vector<vk_device> devices{1};} vk_instance;
#define VK_LOG_DEBUG(...) ((void)0)
#include "strata_native_q8_fma_lifecycle_test.hpp"
void need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main(){try {
    for(int failure=0;failure<5;++failure) {
        counts={};extension_available=failure!=0;feature_available=failure!=1;
        query_success=failure!=2;create_success=failure!=3;fail_after_create=failure==4;
        // Two lookups must each refuse, rather than return a cached invalid
        // object. Unwinding must never call Vulkan destruction without a device.
        for(int attempt=1;attempt<=2;++attempt) {
            bool refused=false;try{ggml_vk_get_device(0);}catch(const std::runtime_error&){refused=true;}
            need(refused && !vk_instance.devices[0],"failed factory poisoned device cache");
            need(counts.constructed==attempt && counts.destroyed==attempt,"failed cache object leaked");
            need(counts.invalid==0,"destructor invoked Vulkan without logical device");
            need(counts.destroys==(failure==4?attempt*3:0),"failure cleanup bypassed logical-device lifetime");
        }
        // A later supported attempt must build once, cache successfully, return
        // that same object, and perform normal pinned destruction on release.
        extension_available=feature_available=query_success=create_success=true;fail_after_create=false;
        auto first=ggml_vk_get_device(0);auto second=ggml_vk_get_device(0);
        need(first==second && first==vk_instance.devices[0] && counts.constructed==3,"valid cache not reused after refusal");
        static_cast<StrataVkQ8Device*>(first.get())->strata_q8_fma.require_enabled();
        vk_instance.devices[0].reset();first.reset();second.reset();
        need(counts.destroyed==3 && counts.invalid==0,"successful cached object cleanup invalid");
        need(counts.destroys==(failure==4?9:3),"logical device not destroyed exactly once");
    }
    puts("strict FMA cached lifecycle: repeated extension/feature/query/create/post-create failures, rollback, retry, reuse and real pinned destructor checked without GPU");return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
