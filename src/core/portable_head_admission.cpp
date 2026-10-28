#include "portable_head_admission.hpp"
#include <exception>
#ifdef STRATA_PORTABLE_VULKAN_WORKSPACE
#include "ggml-vulkan.h"
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstring>
#endif

namespace strata::core::portable_detail {
PhysicalStorageLimit head_storage_limit(ggml_backend_t backend) {
#ifdef STRATA_PORTABLE_VULKAN_WORKSPACE
    if (!backend || !ggml_backend_is_vk(backend)) return {};
    try {
        ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(ggml_backend_get_device(backend), &props);
        const std::string pci = props.device_id ? props.device_id : "";
        if (pci.empty()) return {};
        const PhysicalStorageLimit unknown{pci, 0};
        VkApplicationInfo app{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "strata-head-admission";
        app.apiVersion = VK_API_VERSION_1_2; // pinned GGML also requires >=1.2
        VkInstanceCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        info.pApplicationInfo = &app;
        VkInstance instance = VK_NULL_HANDLE;
        if (vkCreateInstance(&info, nullptr, &instance) != VK_SUCCESS) return unknown;
        struct Cleanup {
            VkInstance instance;
            ~Cleanup() { vkDestroyInstance(instance, nullptr); }
        } cleanup{instance};
        uint32_t count = 0;
        if (vkEnumeratePhysicalDevices(instance, &count, nullptr) != VK_SUCCESS) return unknown;
        std::vector<VkPhysicalDevice> devices(count);
        if (vkEnumeratePhysicalDevices(instance, &count, devices.data()) != VK_SUCCESS) return unknown;
        devices.resize(count);
        std::vector<PhysicalStorageLimit> observations;
        for (auto device : devices) {
            uint32_t extension_count = 0;
            if (vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, nullptr) != VK_SUCCESS)
                return unknown;
            std::vector<VkExtensionProperties> extensions(extension_count);
            if (vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, extensions.data()) != VK_SUCCESS)
                return unknown;
            extensions.resize(extension_count);
            bool pci_extension = false;
            for (const auto& extension : extensions)
                pci_extension |= std::strcmp(extension.extensionName, VK_EXT_PCI_BUS_INFO_EXTENSION_NAME) == 0;
            if (!pci_extension) continue;
            VkPhysicalDevicePCIBusInfoPropertiesEXT bus{};
            bus.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT;
            VkPhysicalDeviceProperties2 physical{};
            physical.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            physical.pNext = &bus;
            vkGetPhysicalDeviceProperties2(device, &physical);
            char id[32]{};
            std::snprintf(id, sizeof(id), "%04x:%02x:%02x.%x", bus.pciDomain, bus.pciBus, bus.pciDevice, bus.pciFunction);
            const auto type = physical.properties.deviceType;
            // Software/virtual devices do not prove a physical GPU limit.
            const bool gpu = type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU || type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
            observations.push_back({id, gpu ? physical.properties.limits.maxStorageBufferRange : 0});
        }
        return match_storage_limit(pci, observations);
    } catch (const std::exception&) {
        return {}; // failed observation keeps the conservative allowance
    }
#else
    (void)backend;
    return {};
#endif
}
} // namespace strata::core::portable_detail
