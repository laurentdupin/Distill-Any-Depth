#pragma once

#if defined(_WIN32)
#  if !defined(WIN32_LEAN_AND_MEAN)
#    define WIN32_LEAN_AND_MEAN
#  endif
#  if !defined(NOMINMAX)
#    define NOMINMAX
#  endif
#  if !defined(VK_USE_PLATFORM_WIN32_KHR)
#    define VK_USE_PLATFORM_WIN32_KHR
#  endif
#endif
#include <vulkan/vulkan.h>
#if defined(__linux__) && !defined(__ANDROID__)
#include <inferbridge/linux_capture_vulkan.h>
#include <memory>
#endif

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dad {

class VulkanContext;
class VulkanPipeline;
class VulkanBuffer;
class VulkanImage;

#include <inferbridge/vulkan/declarations/vulkan_deferred_buffer.h>

#include <inferbridge/vulkan/declarations/vulkan_batched_descriptor.h>

#include <inferbridge/vulkan/declarations/vulkan_dispatch_resource.h>

#include <inferbridge/vulkan/declarations/vulkan_external_capabilities_variant_1.h>

#include <inferbridge/vulkan/declarations/vulkan_semaphore.h>

#include <inferbridge/vulkan/declarations/vulkan_submission.h>

#include <inferbridge/vulkan/declarations/vulkan_buffer.h>

#include <inferbridge/vulkan/declarations/vulkan_image.h>

#include <inferbridge/vulkan/declarations/vulkan_pipeline.h>

class VulkanContext {
public:
#if defined(__linux__) && !defined(__ANDROID__)
    ibr_linux_capture_capabilities linux_capture_capabilities() const;
    VulkanPipeline& capture_pipeline();
    VulkanImage& capture_image(const inferbridge::linux_capture::LinuxDmaBufImage& source);
    VulkanImage import_dma_buf(int file_descriptor, std::uint64_t allocation_size,
        std::uint64_t byte_offset, std::uint64_t modifier,
        std::uint32_t row_stride, std::uint32_t width, std::uint32_t height,
        VkFormat format, VkImageUsageFlags usage);
#endif

    explicit VulkanContext(
        std::uint32_t device_index,
        bool track_resource_hazards = true);
    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;
    ~VulkanContext();

    const std::string& device_name() const { return device_name_; }
    bool supports_packed_int8_dot() const {
        return packed_int8_dot_supported_;
    }
    const VulkanExternalCapabilities& external_capabilities() const {
        return external_capabilities_;
    }
#if defined(_WIN32)
    std::uint64_t adapter_luid() const { return adapter_luid_; }
    // Input handles are borrowed. DAD duplicates them before importing, so
    // the caller may close its handle after this function returns.
    VulkanBuffer import_d3d12_buffer(
        void* shared_handle,
        VkDeviceSize bytes);
    VulkanSemaphore import_d3d12_fence(
        void* shared_handle,
        std::uint64_t value);
    VulkanImage import_d3d12_image(
        void* shared_handle,
        std::uint32_t width,
        std::uint32_t height,
        VkFormat format,
        VkImageUsageFlags usage);
#endif

    VulkanBuffer create_device_buffer(VkDeviceSize bytes);
    VulkanBuffer create_host_buffer(VkDeviceSize bytes);
    void discard(VulkanBuffer& buffer) noexcept;
    void write_host(
        VulkanBuffer& destination,
        const void* data,
        std::size_t bytes);
    void upload(VulkanBuffer& destination, const void* data, std::size_t bytes);
    void download(
        const VulkanBuffer& source,
        void* data,
        std::size_t bytes);
    void transfer_counters(
        std::uint64_t& upload_bytes,
        std::uint64_t& download_bytes) const;
    void acquire_external_buffer(
        const VulkanBuffer& buffer,
        VkAccessFlags destination_access);
    void release_external_buffer(
        const VulkanBuffer& buffer,
        VkAccessFlags source_access);
    void acquire_external_image(
        const VulkanImage& image,
        VkImageLayout layout,
        VkAccessFlags destination_access);
    void release_external_image(
        const VulkanImage& image,
        VkImageLayout layout,
        VkAccessFlags source_access);

    VulkanPipeline create_pipeline(
        const std::uint32_t* spirv,
        std::size_t spirv_bytes,
        std::uint32_t binding_count,
        std::uint32_t push_constant_bytes);
    VulkanPipeline create_pipeline(
        const std::uint32_t* spirv,
        std::size_t spirv_bytes,
        const std::vector<VkDescriptorType>& descriptor_types,
        std::uint32_t push_constant_bytes);
    VulkanPipeline create_pipeline(
        const std::uint32_t* spirv,
        std::size_t spirv_bytes,
        const std::vector<VkDescriptorType>& descriptor_types,
        const std::vector<VkAccessFlags>& descriptor_access,
        std::uint32_t push_constant_bytes);

    void dispatch(
        const VulkanPipeline& pipeline,
        const std::vector<const VulkanBuffer*>& buffers,
        const void* push_constants,
        std::uint32_t push_constant_bytes,
        std::uint32_t group_x,
        std::uint32_t group_y = 1,
        std::uint32_t group_z = 1,
        const VulkanSemaphore* wait = nullptr);
    void dispatch_image_to_buffer(
        const VulkanPipeline& pipeline,
        const VulkanImage& image,
        VulkanBuffer& buffer,
        const void* push_constants,
        std::uint32_t push_constant_bytes,
        std::uint32_t group_x,
        std::uint32_t group_y = 1,
        std::uint32_t group_z = 1);
    void dispatch_buffer_to_image(
        const VulkanPipeline& pipeline,
        const VulkanBuffer& buffer,
        VulkanImage& image,
        const void* push_constants,
        std::uint32_t push_constant_bytes,
        std::uint32_t group_x,
        std::uint32_t group_y = 1,
        std::uint32_t group_z = 1);

    template <typename Function>
    void batch(Function&& function) {
        if (profile_dispatches_) {
            std::forward<Function>(function)();
            return;
        }
        if (batch_command_ != VK_NULL_HANDLE) {
            std::forward<Function>(function)();
            return;
        }
        begin_batch();
        try {
            std::forward<Function>(function)();
            end_batch();
        } catch (...) {
            cancel_batch();
            throw;
        }
    }

    template <typename Function>
    VulkanSubmission batch_async(
        VulkanSemaphore wait,
        VulkanSemaphore signal,
        Function&& function) {
        if (batch_command_ != VK_NULL_HANDLE) {
            throw std::logic_error(
                "asynchronous Vulkan batch cannot be nested");
        }
        begin_batch();
        try {
            std::forward<Function>(function)();
            return end_batch_async(
                std::move(wait), std::move(signal));
        } catch (...) {
            cancel_batch();
            throw;
        }
    }

private:
#if defined(__linux__) && !defined(__ANDROID__)
    bool linux_dma_buf_enabled_ = false;
    inferbridge::linux_capture::ImageCache<VulkanImage> linux_capture_images_;
    std::unique_ptr<VulkanPipeline> linux_capture_pipeline_;
#endif
    friend class VulkanSemaphore;
    friend class VulkanSubmission;
    friend class VulkanBuffer;
    friend class VulkanImage;
    friend class VulkanPipeline;

    std::uint32_t find_memory_type(
        std::uint32_t type_bits,
        VkMemoryPropertyFlags properties) const;
    VulkanBuffer create_buffer(
        VkDeviceSize bytes,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags properties);
    void copy_buffer(
        VkBuffer source,
        VkBuffer destination,
        VkDeviceSize bytes);
    VkCommandBuffer begin_commands();
    VulkanSubmission submit_commands(
        VkCommandBuffer command_buffer,
        const VulkanSemaphore* wait = nullptr,
        const VulkanSemaphore* signal = nullptr);
    void dispatch_resources(
        const VulkanPipeline& pipeline,
        const std::vector<VulkanDispatchResource>& resources,
        const void* push_constants,
        std::uint32_t push_constant_bytes,
        std::uint32_t group_x,
        std::uint32_t group_y,
        std::uint32_t group_z,
        const VulkanSemaphore* wait);
    void end_commands(
        VkCommandBuffer command_buffer,
        const VulkanSemaphore* wait = nullptr);
    void begin_batch();
    void end_batch();
    VulkanSubmission end_batch_async(
        VulkanSemaphore wait,
        VulkanSemaphore signal);
    void cancel_batch() noexcept;
    void release_batch_resources() noexcept;
    void recycle_or_destroy(VulkanDeferredBuffer buffer) noexcept;
    void destroy(VulkanSemaphore& semaphore) noexcept;
    void destroy(VulkanSubmission& submission) noexcept;
    void destroy(VulkanBuffer& buffer) noexcept;
    void destroy(VulkanImage& image) noexcept;
    void destroy(VulkanPipeline& pipeline) noexcept;
    void release_submission_resources(
        VulkanSubmission& submission) noexcept;
    void release() noexcept;
    void record_profile(
        const VulkanPipeline& pipeline,
        std::uint64_t ticks);
    void print_profile() const noexcept;

    struct ProfileStat {
        std::uint64_t total_ticks = 0;
        std::uint64_t maximum_ticks = 0;
        std::uint64_t dispatches = 0;
    };
    static void check(VkResult result, const char* operation);

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memory_properties_{};
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    std::uint32_t queue_family_ = 0;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkQueryPool profile_query_pool_ = VK_NULL_HANDLE;
    bool profile_dispatches_ = false;
    bool subgroup_size_forced_ = false;
    bool packed_int8_dot_supported_ = false;
    float timestamp_period_ns_ = 0.0f;
    std::unordered_map<std::string, ProfileStat> profile_stats_;
    VkCommandBuffer batch_command_ = VK_NULL_HANDLE;
    bool batch_has_dispatch_ = false;
    bool track_resource_hazards_ = true;
    std::vector<VulkanBatchedDescriptor> batch_descriptor_sets_;
    std::vector<VulkanDeferredBuffer> batch_deferred_buffers_;
    std::unordered_map<VkBuffer, VkAccessFlags>
        batch_buffer_access_;
    std::unordered_map<VkImage, VkAccessFlags>
        batch_image_access_;
    std::unordered_map<VkImage, VkImageLayout>
        batch_image_layout_;
    std::vector<VulkanDeferredBuffer> device_buffer_pool_;
    std::vector<VulkanDeferredBuffer> host_buffer_pool_;
    VkDeviceSize pooled_device_bytes_ = 0;
    VkDeviceSize pooled_host_bytes_ = 0;
    VulkanExternalCapabilities external_capabilities_{};
#if defined(_WIN32)
    std::uint64_t adapter_luid_ = 0;
    PFN_vkGetMemoryWin32HandlePropertiesKHR
        get_memory_win32_handle_properties_ = nullptr;
    PFN_vkImportSemaphoreWin32HandleKHR
        import_semaphore_win32_handle_ = nullptr;
#endif
    std::atomic<std::uint64_t> tensor_upload_bytes_{0};
    std::atomic<std::uint64_t> tensor_download_bytes_{0};
    std::string device_name_;
};

}  // namespace dad
