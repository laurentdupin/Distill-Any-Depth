#include "executor.h"

#include <stdexcept>

namespace dad {

std::unique_ptr<Executor> create_executor(
    const std::string& model_path, dad_encoder encoder,
    int vulkan_device_index, std::uint32_t flags,
    const std::string& cache_path) {
    const bool force_metal = (flags & DAD_CREATE_FORCE_METAL) != 0u;
    const bool force_vulkan = (flags & DAD_CREATE_FORCE_VULKAN) != 0u;
    if (force_metal && force_vulkan)
        throw std::invalid_argument("Metal and Vulkan cannot both be forced");
#if defined(DAD_WITH_METAL)
    if (!force_vulkan)
        return create_metal_executor(model_path, encoder, flags, cache_path);
#else
    if (force_metal)
        throw std::runtime_error("this runtime was built without Metal");
#endif
#if defined(DAD_WITH_VULKAN)
    return create_vulkan_executor(model_path, encoder, vulkan_device_index);
#else
    throw std::runtime_error("this runtime has no available executor");
#endif
}

GpuCapabilities probe_gpu_capabilities(int vulkan_device_index) {
#if defined(DAD_WITH_VULKAN)
    return probe_vulkan_gpu_capabilities(vulkan_device_index);
#else
    (void)vulkan_device_index;
    return {};
#endif
}

}  // namespace dad
