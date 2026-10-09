#include "vulkan.h"
#include "inferbridge/native_harness_vulkan_queue_priority.h"
#if defined(__linux__) && !defined(__ANDROID__)
#include <inferbridge/linux_capture_vulkan.h>
#include <linux_capture_preprocess_spv.h>
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace dad {
namespace {

template <typename Handle>
void exchange_handle(Handle& left, Handle& right) {
    std::swap(left, right);
}

#if defined(_WIN32)
bool has_extension(
    const std::vector<VkExtensionProperties>& extensions,
    const char* name) {
    return std::any_of(
        extensions.begin(),
        extensions.end(),
        [name](const VkExtensionProperties& extension) {
            return std::strcmp(extension.extensionName, name) == 0;
        });
}
#endif

}  // namespace

struct VulkanSubmission::Resources {
    std::vector<VulkanBatchedDescriptor> descriptor_sets;
    std::vector<VulkanDeferredBuffer> deferred_buffers;
    VulkanSemaphore wait;
    VulkanSemaphore signal;
};

#include <inferbridge/vulkan/context/vulkan_context_check.h>

VulkanSemaphore::VulkanSemaphore(VulkanSemaphore&& other) noexcept {
    *this = std::move(other);
}

#include <inferbridge/vulkan/context/vulkan_semaphore_operator_assign.h>

#include <inferbridge/vulkan/context/vulkan_semaphore_destroy_VulkanSemaphore.h>

VulkanSubmission::VulkanSubmission(VulkanSubmission&& other) noexcept {
    *this = std::move(other);
}

#include <inferbridge/vulkan/context/vulkan_submission_operator_assign.h>

#include <inferbridge/vulkan/context/vulkan_submission_destroy_VulkanSubmission.h>

#include <inferbridge/vulkan/context/vulkan_submission_ready.h>

#include <inferbridge/vulkan/context/vulkan_submission_wait.h>

VulkanBuffer::VulkanBuffer(VulkanBuffer&& other) noexcept {
    *this = std::move(other);
}

#include <inferbridge/vulkan/context/vulkan_buffer_operator_assign.h>

#include <inferbridge/vulkan/context/vulkan_buffer_destroy_VulkanBuffer.h>

VulkanImage::VulkanImage(VulkanImage&& other) noexcept {
    *this = std::move(other);
}

#include <inferbridge/vulkan/context/vulkan_image_operator_assign.h>

#include <inferbridge/vulkan/context/vulkan_image_destroy_VulkanImage.h>

VulkanPipeline::VulkanPipeline(VulkanPipeline&& other) noexcept {
    *this = std::move(other);
}

#include <inferbridge/vulkan/context/vulkan_pipeline_operator_assign.h>

#include <inferbridge/vulkan/context/vulkan_pipeline_destroy_VulkanPipeline.h>

#include <inferbridge/vulkan/context/vulkan_pipeline_set_debug_name.h>

VulkanContext::VulkanContext(
    std::uint32_t device_index,
    bool track_resource_hazards)
    : track_resource_hazards_(track_resource_hazards) {
    try {
    const VkApplicationInfo application{
        VK_STRUCTURE_TYPE_APPLICATION_INFO,
        nullptr,
        "Distill AnyDepth C",
        1,
        "DAD",
        1,
        VK_API_VERSION_1_3,
    };
    const VkInstanceCreateInfo instance_info{
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        nullptr,
        0,
        &application,
        0,
        nullptr,
        0,
        nullptr,
    };
    check(vkCreateInstance(&instance_info, nullptr, &instance_), "vkCreateInstance");

    std::uint32_t device_count = 0;
    check(
        vkEnumeratePhysicalDevices(instance_, &device_count, nullptr),
        "vkEnumeratePhysicalDevices");
    if (device_count == 0 || device_index >= device_count) {
        throw std::runtime_error("requested Vulkan device does not exist");
    }
    std::vector<VkPhysicalDevice> devices(device_count);
    check(
        vkEnumeratePhysicalDevices(instance_, &device_count, devices.data()),
        "vkEnumeratePhysicalDevices");
    physical_device_ = devices[device_index];

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical_device_, &properties);
    device_name_ = properties.deviceName;
    VkPhysicalDeviceSubgroupSizeControlProperties subgroup_control{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES,
    };
    VkPhysicalDeviceProperties2 subgroup_properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        &subgroup_control,
    };
    vkGetPhysicalDeviceProperties2(
        physical_device_, &subgroup_properties);
    VkPhysicalDeviceSubgroupSizeControlFeatures subgroup_features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES,
    };
    VkPhysicalDeviceShaderIntegerDotProductFeatures integer_dot_features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES,
    };
    subgroup_features.pNext = &integer_dot_features;
    VkPhysicalDeviceFeatures2 device_features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        &subgroup_features,
    };
    vkGetPhysicalDeviceFeatures2(physical_device_, &device_features);
    subgroup_size_forced_ =
        subgroup_features.subgroupSizeControl == VK_TRUE &&
        (subgroup_control.requiredSubgroupSizeStages &
         VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
        subgroup_control.minSubgroupSize <= 32 &&
        subgroup_control.maxSubgroupSize >= 32;
    VkPhysicalDeviceShaderIntegerDotProductProperties integer_dot_properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES,
    };
    VkPhysicalDeviceProperties2 integer_dot_properties2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        &integer_dot_properties,
    };
    vkGetPhysicalDeviceProperties2(physical_device_, &integer_dot_properties2);
    packed_int8_dot_supported_ =
        integer_dot_features.shaderIntegerDotProduct == VK_TRUE &&
        integer_dot_properties
            .integerDotProduct4x8BitPackedSignedAccelerated == VK_TRUE;
#if defined(_WIN32)
    VkPhysicalDeviceIDProperties identity{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES,
    };
    VkPhysicalDeviceProperties2 properties2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        &identity,
    };
    vkGetPhysicalDeviceProperties2(physical_device_, &properties2);
    if (identity.deviceLUIDValid) {
        static_assert(VK_LUID_SIZE == sizeof(adapter_luid_));
        std::memcpy(
            &adapter_luid_, identity.deviceLUID, VK_LUID_SIZE);
    }
#endif
    vkGetPhysicalDeviceMemoryProperties(
        physical_device_, &memory_properties_);

    std::uint32_t extension_count = 0;
    check(
        vkEnumerateDeviceExtensionProperties(
            physical_device_, nullptr, &extension_count, nullptr),
        "vkEnumerateDeviceExtensionProperties");
    std::vector<VkExtensionProperties> extensions(extension_count);
    check(
        vkEnumerateDeviceExtensionProperties(
            physical_device_, nullptr, &extension_count, extensions.data()),
        "vkEnumerateDeviceExtensionProperties");
    std::vector<const char*> enabled_extensions;
#if defined(__linux__) && !defined(__ANDROID__)
    linux_dma_buf_enabled_ = inferbridge::linux_capture::enable_extensions(
        extensions, enabled_extensions);
#endif
#if defined(_WIN32)
    const bool has_external_memory_win32 = has_extension(
        extensions, VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
    const bool has_external_semaphore_win32 = has_extension(
        extensions, VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
    if (has_external_memory_win32) {
        const VkPhysicalDeviceExternalBufferInfo external_buffer{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO,
            nullptr,
            0,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT,
        };
        VkExternalBufferProperties external_properties{
            VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES,
        };
        vkGetPhysicalDeviceExternalBufferProperties(
            physical_device_,
            &external_buffer,
            &external_properties);
        external_capabilities_.d3d12_resource_import =
            (external_properties.externalMemoryProperties
                 .externalMemoryFeatures &
             VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) != 0;
        const auto image_importable =
            [&](VkFormat format, VkImageUsageFlags usage) {
                const VkPhysicalDeviceExternalImageFormatInfo external{
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO,
                    nullptr,
                    VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT,
                };
                const VkPhysicalDeviceImageFormatInfo2 image_info{
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
                    &external,
                    format,
                    VK_IMAGE_TYPE_2D,
                    VK_IMAGE_TILING_OPTIMAL,
                    usage,
                    0,
                };
                VkExternalImageFormatProperties external_properties{
                    VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES,
                };
                VkImageFormatProperties2 image_properties{
                    VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
                    &external_properties,
                };
                const VkResult queried =
                    vkGetPhysicalDeviceImageFormatProperties2(
                        physical_device_,
                        &image_info,
                        &image_properties);
                return queried == VK_SUCCESS &&
                    (external_properties.externalMemoryProperties
                         .externalMemoryFeatures &
                     VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) != 0;
            };
        external_capabilities_
            .d3d12_bgra8_sampled_image_import =
            image_importable(
                VK_FORMAT_B8G8R8A8_UNORM,
                VK_IMAGE_USAGE_SAMPLED_BIT |
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
        external_capabilities_
            .d3d12_rgba8_sampled_image_import =
            image_importable(
                VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_USAGE_SAMPLED_BIT |
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
        external_capabilities_
            .d3d12_r32_storage_image_import =
            image_importable(
                VK_FORMAT_R32_SFLOAT,
                VK_IMAGE_USAGE_STORAGE_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        enabled_extensions.push_back(
            VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
    }
    if (has_external_semaphore_win32) {
        const VkPhysicalDeviceExternalSemaphoreInfo external_semaphore{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO,
            nullptr,
            VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT,
        };
        VkExternalSemaphoreProperties semaphore_properties{
            VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES,
        };
        vkGetPhysicalDeviceExternalSemaphoreProperties(
            physical_device_,
            &external_semaphore,
            &semaphore_properties);
        external_capabilities_.d3d12_fence_import =
            (semaphore_properties.externalSemaphoreFeatures &
             VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT) != 0;
        enabled_extensions.push_back(
            VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
    }
#else
    (void)extensions;
#endif

    std::uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(
        physical_device_, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(
        physical_device_, &family_count, families.data());
#if defined(_WIN32)
    constexpr bool prefer_dedicated = true;
#else
    constexpr bool prefer_dedicated = false;
#endif
    auto family = inferbridge::native_harness::select_inference_queue_family(families, prefer_dedicated);
    if (family == families.end()) {
        throw std::runtime_error("Vulkan device has no compute queue");
    }
    queue_family_ = static_cast<std::uint32_t>(
        std::distance(families.begin(), family));
    const char* profile_environment =
        std::getenv("DAD_VULKAN_PROFILE");
    profile_dispatches_ =
        profile_environment != nullptr &&
        profile_environment[0] != '\0' &&
        profile_environment[0] != '0' &&
        family->timestampValidBits != 0;
    timestamp_period_ns_ = properties.limits.timestampPeriod;

    constexpr float priority = 1.0f;
    const VkDeviceQueueCreateInfo queue_info{
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        nullptr,
        0,
        queue_family_,
        1,
        &priority,
    };
    VkPhysicalDeviceShaderIntegerDotProductFeatures enabled_integer_dot{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES,
        nullptr,
        packed_int8_dot_supported_ ? VK_TRUE : VK_FALSE,
    };
    subgroup_features.pNext = packed_int8_dot_supported_
        ? &enabled_integer_dot : nullptr;
    const VkDeviceCreateInfo device_info{
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        subgroup_size_forced_ ? static_cast<void*>(&subgroup_features)
        : packed_int8_dot_supported_
        ? static_cast<void*>(&enabled_integer_dot) : nullptr,
        0,
        1,
        &queue_info,
        0,
        nullptr,
        static_cast<std::uint32_t>(enabled_extensions.size()),
        enabled_extensions.data(),
        nullptr,
    };
    check(
        inferbridge::native_harness::create_inference_vulkan_device(physical_device_, &device_info, nullptr, &device_),
        "vkCreateDevice");
#if defined(_WIN32)
    get_memory_win32_handle_properties_ =
        reinterpret_cast<
            PFN_vkGetMemoryWin32HandlePropertiesKHR>(
            vkGetDeviceProcAddr(
                device_,
                "vkGetMemoryWin32HandlePropertiesKHR"));
    import_semaphore_win32_handle_ =
        reinterpret_cast<PFN_vkImportSemaphoreWin32HandleKHR>(
            vkGetDeviceProcAddr(
                device_,
                "vkImportSemaphoreWin32HandleKHR"));
    external_capabilities_.d3d12_resource_import =
        external_capabilities_.d3d12_resource_import &&
        get_memory_win32_handle_properties_ != nullptr;
    external_capabilities_.d3d12_fence_import =
        external_capabilities_.d3d12_fence_import &&
        import_semaphore_win32_handle_ != nullptr;
#endif
    vkGetDeviceQueue(device_, queue_family_, 0, &queue_);

    const VkCommandPoolCreateInfo command_pool_info{
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        nullptr,
        VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
            VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        queue_family_,
    };
    check(
        vkCreateCommandPool(
            device_, &command_pool_info, nullptr, &command_pool_),
        "vkCreateCommandPool");

    const VkDescriptorPoolSize pool_sizes[] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16384},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 16384},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 64},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 64},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 64},
    };
    const VkDescriptorPoolCreateInfo descriptor_pool_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        nullptr,
        VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        4096,
        static_cast<std::uint32_t>(std::size(pool_sizes)),
        pool_sizes,
    };
    check(
        vkCreateDescriptorPool(
            device_, &descriptor_pool_info, nullptr, &descriptor_pool_),
        "vkCreateDescriptorPool");
    if (profile_dispatches_) {
        const VkQueryPoolCreateInfo query_pool_info{
            VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            nullptr,
            0,
            VK_QUERY_TYPE_TIMESTAMP,
            2,
            0,
        };
        check(
            vkCreateQueryPool(
                device_,
                &query_pool_info,
                nullptr,
                &profile_query_pool_),
            "vkCreateQueryPool");
    }
    } catch (...) {
        release();
        throw;
    }
}

#include <inferbridge/vulkan/context/vulkan_context_destroy_VulkanContext.h>

#include <inferbridge/vulkan/context/vulkan_context_release_variant_2.h>

#include <inferbridge/vulkan/context/vulkan_context_record_profile.h>

#include <inferbridge/vulkan/context/vulkan_context_print_profile_variant_1.h>

#include <inferbridge/vulkan/context/vulkan_context_find_memory_type.h>

#include <inferbridge/vulkan/context/vulkan_context_create_buffer.h>

#include <inferbridge/vulkan/context/vulkan_context_create_device_buffer.h>

#if defined(_WIN32)
#include <inferbridge/vulkan/context/vulkan_context_import_d3d12_buffer.h>

#include <inferbridge/vulkan/context/vulkan_context_import_d3d12_image.h>

#include <inferbridge/vulkan/context/vulkan_context_import_d3d12_fence.h>

#endif

#include <inferbridge/vulkan/context/vulkan_context_create_host_buffer.h>

#include <inferbridge/vulkan/context/vulkan_context_write_host.h>

#include <inferbridge/vulkan/context/vulkan_context_begin_commands.h>

#include <inferbridge/vulkan/context/vulkan_context_end_commands.h>

#include <inferbridge/vulkan/context/vulkan_context_submit_commands.h>

#include <inferbridge/vulkan/context/vulkan_context_begin_batch.h>

#include <inferbridge/vulkan/context/vulkan_context_release_batch_resources.h>

#include <inferbridge/vulkan/context/vulkan_context_end_batch_async_variant_1.h>

#include <inferbridge/vulkan/context/vulkan_context_end_batch.h>

#include <inferbridge/vulkan/context/vulkan_context_cancel_batch_variant_1.h>

#include <inferbridge/vulkan/context/vulkan_context_copy_buffer.h>

#include <inferbridge/vulkan/context/vulkan_context_upload_variant_1.h>

#include <inferbridge/vulkan/context/vulkan_context_download_variant_3.h>

#include <inferbridge/vulkan/context/vulkan_context_transfer_counters.h>

#include <inferbridge/vulkan/context/vulkan_context_acquire_external_buffer_variant_1.h>

#include <inferbridge/vulkan/context/vulkan_context_release_external_buffer_variant_2.h>

#include <inferbridge/linux_capture_vulkan_context.inl>


#include <inferbridge/vulkan/context/vulkan_context_acquire_external_image_variant_1.h>

#include <inferbridge/vulkan/context/vulkan_context_release_external_image_variant_2.h>

#include <inferbridge/vulkan/context/vulkan_context_create_pipeline_variant_4.h>

#include <inferbridge/vulkan/context/vulkan_context_create_pipeline_variant_1.h>

#include <inferbridge/vulkan/context/vulkan_context_create_pipeline_variant_2.h>

#include <inferbridge/vulkan/context/vulkan_context_dispatch.h>

#include <inferbridge/vulkan/context/vulkan_context_dispatch_image_to_buffer.h>

#include <inferbridge/vulkan/context/vulkan_context_dispatch_buffer_to_image.h>

#include <inferbridge/vulkan/context/vulkan_context_dispatch_resources_variant_2.h>

#include <inferbridge/vulkan/context/vulkan_context_destroy_variant_4.h>

#include <inferbridge/vulkan/context/vulkan_context_destroy_variant_5.h>

#include <inferbridge/vulkan/context/vulkan_context_destroy_variant_2.h>

#include <inferbridge/vulkan/context/vulkan_context_release_submission_resources_variant_2.h>

#include <inferbridge/vulkan/context/vulkan_context_destroy_variant_1.h>

#include <inferbridge/vulkan/context/vulkan_context_recycle_or_destroy.h>

#include <inferbridge/vulkan/context/vulkan_context_discard.h>

#include <inferbridge/vulkan/context/vulkan_context_destroy_variant_3.h>

}  // namespace dad
