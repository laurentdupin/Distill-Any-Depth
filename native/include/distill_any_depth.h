#ifndef DISTILL_ANY_DEPTH_H
#define DISTILL_ANY_DEPTH_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(DAD_BUILD_DLL)
#    define DAD_API __declspec(dllexport)
#  else
#    define DAD_API __declspec(dllimport)
#  endif
#  define DAD_CALL __cdecl
#else
#  define DAD_API __attribute__((visibility("default")))
#  define DAD_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define DAD_ABI_VERSION 1u

typedef struct dad_context dad_context;
typedef struct dad_gpu_job dad_gpu_job;
typedef struct dad_gpu_output_lease dad_gpu_output_lease;

typedef enum dad_status {
    DAD_STATUS_OK = 0,
    DAD_STATUS_INVALID_ARGUMENT = 1,
    DAD_STATUS_MODEL_IO = 2,
    DAD_STATUS_MODEL_FORMAT = 3,
    DAD_STATUS_VULKAN_UNAVAILABLE = 4,
    DAD_STATUS_OUT_OF_MEMORY = 5,
    DAD_STATUS_INFERENCE_FAILED = 6,
    DAD_STATUS_BUFFER_TOO_SMALL = 7,
    DAD_STATUS_UNSUPPORTED = 8,
    DAD_STATUS_INTERNAL_ERROR = 9,
    DAD_STATUS_INVALID_STATE = 10,
    DAD_STATUS_CANCELLED = 11
} dad_status;

typedef enum dad_encoder {
    DAD_ENCODER_VITS = 0,
    DAD_ENCODER_VITB = 1,
    DAD_ENCODER_VITL = 2
} dad_encoder;

typedef struct dad_create_options {
    uint32_t struct_size;
    uint32_t abi_version;
    dad_encoder encoder;
    int32_t vulkan_device_index;
    uint32_t flags;
} dad_create_options;

typedef struct dad_image_shape {
    int32_t width;
    int32_t height;
} dad_image_shape;

enum {
    DAD_GPU_CAP_D3D12_SHARED_INPUT = 1ull << 0u,
    DAD_GPU_CAP_D3D12_FENCE_WAIT = 1ull << 1u,
    DAD_GPU_CAP_D3D12_SHARED_OUTPUT = 1ull << 2u,
    DAD_GPU_CAP_D3D12_FENCE_SIGNAL = 1ull << 3u,
    DAD_GPU_CAP_ASYNC_SUBMIT = 1ull << 4u,
    DAD_GPU_CAP_CANCELLATION = 1ull << 5u,
    DAD_GPU_CAP_NO_HOST_PIXEL_STAGING = 1ull << 6u,
    DAD_GPU_CAP_NO_HOST_DEPTH_STAGING = 1ull << 7u,
    DAD_GPU_CAP_D3D12_SHARED_TEXTURE_INPUT = 1ull << 8u,
    DAD_GPU_CAP_D3D12_SHARED_TEXTURE_OUTPUT = 1ull << 9u
};

typedef enum dad_gpu_pixel_format {
    DAD_GPU_PIXEL_BGRA8 = 1,
    DAD_GPU_PIXEL_RGBA8 = 2,
    DAD_GPU_PIXEL_DEPTH_FLOAT32 = 3
} dad_gpu_pixel_format;

typedef enum dad_gpu_job_state {
    DAD_GPU_JOB_QUEUED = 0,
    DAD_GPU_JOB_RUNNING = 1,
    DAD_GPU_JOB_COMPLETE = 2,
    DAD_GPU_JOB_FAILED = 3,
    DAD_GPU_JOB_CANCELLED = 4
} dad_gpu_job_state;

typedef struct dad_gpu_capabilities {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t flags;
    uint64_t adapter_luid;
    uint32_t maximum_in_flight_jobs;
    uint32_t reserved;
} dad_gpu_capabilities;

/*
 * Windows-only GPU submission contract.
 *
 * shared_resource_handle is a borrowed NT handle for a shared D3D12 buffer
 * containing tightly addressable BGRA8 or RGBA8 pixels. wait_fence_handle is
 * a borrowed NT handle for a shared D3D12 fence. DAD duplicates both handles
 * before returning from submit, so the caller may close them immediately.
 */
typedef struct dad_d3d12_submit_request {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t shared_resource_handle;
    uint64_t resource_byte_size;
    uint32_t width;
    uint32_t height;
    uint32_t row_stride_bytes;
    uint32_t pixel_format;
    int32_t input_size;
    uint32_t reserved;
    uint64_t wait_fence_handle;
    uint64_t wait_fence_value;
    uint64_t source_frame_id;
    uint64_t timestamp_ns;
} dad_d3d12_submit_request;

/*
 * shared_texture_handle is a borrowed NT handle for a shared D3D12
 * TEXTURE2D. BGRA8 maps to DXGI_FORMAT_B8G8R8A8_UNORM and RGBA8 maps to
 * DXGI_FORMAT_R8G8B8A8_UNORM. DAD duplicates/imports the texture and fence
 * handles before returning.
 */
typedef struct dad_d3d12_texture_submit_request {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t shared_texture_handle;
    uint32_t width;
    uint32_t height;
    uint32_t pixel_format;
    int32_t input_size;
    uint64_t wait_fence_handle;
    uint64_t wait_fence_value;
    uint64_t source_frame_id;
    uint64_t timestamp_ns;
} dad_d3d12_texture_submit_request;

typedef struct dad_d3d12_texture_binding_request {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t input_texture_handle;
    uint32_t input_width;
    uint32_t input_height;
    uint32_t input_pixel_format;
    int32_t input_size;
    uint64_t wait_fence_handle;
    uint64_t wait_fence_value;
    uint64_t output_texture_handle;
    uint32_t output_width;
    uint32_t output_height;
    uint64_t signal_fence_handle;
    uint64_t signal_fence_value;
    uint64_t source_frame_id;
    uint64_t timestamp_ns;
} dad_d3d12_texture_binding_request;

typedef struct dad_gpu_job_status {
    uint32_t struct_size;
    uint32_t state;
    uint32_t output_count;
    uint32_t reserved;
    uint64_t source_frame_id;
} dad_gpu_job_status;

/*
 * Handles in this descriptor are borrowed and remain valid until the matching
 * output lease is released. The resource is a D3D12 shared buffer containing
 * row-major float32 depth. Consumers must wait for ready_fence_value on the
 * shared D3D12 fence before accessing it. If a consumer transitions the
 * resource, it must return it to D3D12_RESOURCE_STATE_COMMON before releasing
 * the lease so DAD may safely reuse the slot.
 */
typedef struct dad_d3d12_output_descriptor {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t pixel_format;
    uint32_t reserved;
    uint32_t width;
    uint32_t height;
    uint32_t row_stride_bytes;
    uint32_t reserved2;
    uint64_t byte_size;
    uint64_t shared_resource_handle;
    uint64_t ready_fence_handle;
    uint64_t ready_fence_value;
    uint64_t source_frame_id;
    uint64_t timestamp_ns;
} dad_d3d12_output_descriptor;

/*
 * The leased resource is a shared D3D12 TEXTURE2D with
 * DXGI_FORMAT_R32_FLOAT. Both handles are borrowed for the lease lifetime.
 * A consumer must return the texture to D3D12_RESOURCE_STATE_COMMON before
 * releasing the lease if it performs an explicit state transition.
 */
typedef struct dad_d3d12_texture_output_descriptor {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t pixel_format;
    uint32_t reserved;
    uint32_t width;
    uint32_t height;
    uint32_t reserved2;
    uint32_t reserved3;
    uint64_t shared_texture_handle;
    uint64_t ready_fence_handle;
    uint64_t ready_fence_value;
    uint64_t source_frame_id;
    uint64_t timestamp_ns;
} dad_d3d12_texture_output_descriptor;

typedef struct dad_transfer_counters {
    uint32_t struct_size;
    uint32_t reserved;
    uint64_t tensor_upload_bytes;
    uint64_t tensor_download_bytes;
} dad_transfer_counters;

DAD_API uint32_t DAD_CALL dad_abi_version(void);
DAD_API const char* DAD_CALL dad_version_string(void);
DAD_API const char* DAD_CALL dad_status_string(dad_status status);
DAD_API const char* DAD_CALL dad_last_error(void);

DAD_API dad_status DAD_CALL dad_get_network_shape(
    int32_t image_width,
    int32_t image_height,
    int32_t input_size,
    dad_image_shape* network_shape);

DAD_API dad_status DAD_CALL dad_get_inferbridge_shape(
    int32_t image_width,
    int32_t image_height,
    int32_t input_size,
    dad_image_shape* output_shape);

DAD_API dad_status DAD_CALL dad_create(
    const char* model_path_utf8,
    const dad_create_options* options,
    dad_context** context);

DAD_API void DAD_CALL dad_destroy(dad_context* context);

DAD_API dad_status DAD_CALL dad_probe_gpu_capabilities(
    int32_t vulkan_device_index,
    dad_gpu_capabilities* capabilities);

DAD_API dad_status DAD_CALL dad_get_gpu_capabilities(
    const dad_context* context,
    dad_gpu_capabilities* capabilities);

DAD_API dad_status DAD_CALL dad_get_transfer_counters(
    const dad_context* context,
    dad_transfer_counters* counters);

DAD_API dad_status DAD_CALL dad_submit_d3d12(
    dad_context* context,
    const dad_d3d12_submit_request* request,
    dad_gpu_job** job);

DAD_API dad_status DAD_CALL dad_submit_d3d12_texture(
    dad_context* context,
    const dad_d3d12_texture_submit_request* request,
    dad_gpu_job** job);

DAD_API dad_status DAD_CALL dad_submit_d3d12_texture_binding(
    dad_context* context,
    const dad_d3d12_texture_binding_request* request,
    dad_gpu_job** job);

DAD_API dad_status DAD_CALL dad_gpu_job_poll(
    const dad_gpu_job* job,
    dad_gpu_job_status* status);

DAD_API dad_status DAD_CALL dad_gpu_job_cancel(dad_gpu_job* job);
DAD_API void DAD_CALL dad_gpu_job_release(dad_gpu_job* job);

DAD_API dad_status DAD_CALL dad_gpu_output_acquire(
    dad_gpu_job* job,
    uint32_t output_index,
    dad_d3d12_output_descriptor* descriptor,
    dad_gpu_output_lease** lease);

DAD_API dad_status DAD_CALL dad_gpu_texture_output_acquire(
    dad_gpu_job* job,
    uint32_t output_index,
    dad_d3d12_texture_output_descriptor* descriptor,
    dad_gpu_output_lease** lease);

DAD_API void DAD_CALL dad_gpu_output_release(
    dad_gpu_output_lease* lease);

/*
 * Runs the complete Python-compatible image path.
 *
 * Input is interleaved BGR uint8 data, matching cv2.imread. The output is one
 * row-major float32 depth value per source pixel. output_count must be at least
 * image_width * image_height.
 */
DAD_API dad_status DAD_CALL dad_infer_bgr8(
    dad_context* context,
    const uint8_t* bgr,
    int32_t image_width,
    int32_t image_height,
    ptrdiff_t row_stride_bytes,
    int32_t input_size,
    float* output_depth,
    size_t output_count);

/*
 * Exact deployed InferBridge image contract. BGRA bytes B, G, R are model
 * channels 0, 1, 2; resize is PyTorch legacy nearest; output is independently
 * min/max normalized float32 in [0,1].
 */
DAD_API dad_status DAD_CALL dad_inferbridge_bgra8_f32(
    dad_context* context,
    const uint8_t* bgra,
    int32_t image_width,
    int32_t image_height,
    ptrdiff_t row_stride_bytes,
    int32_t input_size,
    float* output_depth,
    size_t output_count);

/*
 * Runs only the neural network. Input is normalized RGB planar float32 NCHW
 * data for a batch of one. Width and height must be positive multiples of 14.
 * Output has network_width * network_height float32 values.
 */
DAD_API dad_status DAD_CALL dad_infer_tensor_f32(
    dad_context* context,
    const float* normalized_rgb_chw,
    int32_t network_width,
    int32_t network_height,
    float* output_depth,
    size_t output_count);

#ifdef __cplusplus
}
#endif

#endif
