#include "distill_any_depth.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Microsoft::WRL::ComPtr;

void check(HRESULT result, const char* operation) {
    if (FAILED(result)) {
        throw std::runtime_error(
            std::string(operation) + " failed with HRESULT " +
            std::to_string(static_cast<long>(result)));
    }
}

void check_device(
    HRESULT result,
    ID3D12Device* device,
    const char* operation) {
    if (FAILED(result)) {
        const HRESULT removed =
            device != nullptr
            ? device->GetDeviceRemovedReason()
            : S_OK;
        throw std::runtime_error(
            std::string(operation) + " failed with HRESULT " +
            std::to_string(static_cast<long>(result)) +
            "; device removed reason " +
            std::to_string(static_cast<long>(removed)));
    }
}

void check(dad_status status, const char* operation) {
    if (status != DAD_STATUS_OK) {
        throw std::runtime_error(
            std::string(operation) + " failed: " +
            dad_status_string(status) + ": " +
            dad_last_error());
    }
}

ComPtr<ID3D12Device> matching_device(std::uint64_t luid) {
    ComPtr<IDXGIFactory6> factory;
    check(
        CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)),
        "CreateDXGIFactory2");
    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> adapter;
        const HRESULT enumerated =
            factory->EnumAdapters1(index, &adapter);
        if (enumerated == DXGI_ERROR_NOT_FOUND) break;
        check(enumerated, "EnumAdapters1");
        DXGI_ADAPTER_DESC1 description{};
        check(adapter->GetDesc1(&description), "GetDesc1");
        std::uint64_t candidate = 0;
        static_assert(sizeof(candidate) == sizeof(description.AdapterLuid));
        std::memcpy(
            &candidate, &description.AdapterLuid, sizeof(candidate));
        if (candidate != luid) continue;
        ComPtr<ID3D12Device> device;
        check(
            D3D12CreateDevice(
                adapter.Get(),
                D3D_FEATURE_LEVEL_11_0,
                IID_PPV_ARGS(&device)),
            "D3D12CreateDevice");
        return device;
    }
    throw std::runtime_error(
        "no D3D12 adapter matches the Vulkan device LUID");
}

ComPtr<ID3D12CommandQueue> make_queue(ID3D12Device* device) {
    const D3D12_COMMAND_QUEUE_DESC description{
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        0,
        D3D12_COMMAND_QUEUE_FLAG_NONE,
        0,
    };
    ComPtr<ID3D12CommandQueue> result;
    check(
        device->CreateCommandQueue(
            &description, IID_PPV_ARGS(&result)),
        "CreateCommandQueue");
    return result;
}

void cpu_wait(ID3D12Fence* fence, std::uint64_t value) {
    if (fence->GetCompletedValue() >= value) return;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (event == nullptr) {
        throw std::runtime_error("CreateEventW failed");
    }
    const HRESULT scheduled =
        fence->SetEventOnCompletion(value, event);
    if (FAILED(scheduled)) {
        CloseHandle(event);
        check(scheduled, "SetEventOnCompletion");
    }
    if (WaitForSingleObject(event, INFINITE) != WAIT_OBJECT_0) {
        CloseHandle(event);
        throw std::runtime_error("fence wait failed");
    }
    CloseHandle(event);
}

struct SharedInput {
    ComPtr<ID3D12Resource> resource;
    ComPtr<ID3D12Resource> upload;
    ComPtr<ID3D12Fence> ready;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> commands;
    HANDLE resource_handle = nullptr;
    HANDLE fence_handle = nullptr;
    std::uint64_t fence_value = 1;

    SharedInput() = default;
    SharedInput(const SharedInput&) = delete;
    SharedInput& operator=(const SharedInput&) = delete;
    SharedInput(SharedInput&& other) noexcept
        : resource(std::move(other.resource)),
          upload(std::move(other.upload)),
          ready(std::move(other.ready)),
          allocator(std::move(other.allocator)),
          commands(std::move(other.commands)),
          resource_handle(
              std::exchange(other.resource_handle, nullptr)),
          fence_handle(std::exchange(other.fence_handle, nullptr)),
          fence_value(other.fence_value) {}

    ~SharedInput() {
        if (resource_handle != nullptr) CloseHandle(resource_handle);
        if (fence_handle != nullptr) CloseHandle(fence_handle);
    }
};

SharedInput upload_capture(
    ID3D12Device* device,
    ID3D12CommandQueue* queue,
    const std::vector<std::uint8_t>& pixels,
    bool signal_ready) {
    const D3D12_HEAP_PROPERTIES default_heap{
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL_UNKNOWN,
        1,
        1,
    };
    const D3D12_HEAP_PROPERTIES upload_heap{
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL_UNKNOWN,
        1,
        1,
    };
    const D3D12_RESOURCE_DESC description{
        D3D12_RESOURCE_DIMENSION_BUFFER,
        0,
        pixels.size(),
        1,
        1,
        1,
        DXGI_FORMAT_UNKNOWN,
        {1, 0},
        D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        D3D12_RESOURCE_FLAG_NONE,
    };
    SharedInput result;
    check(
        device->CreateCommittedResource(
            &default_heap,
            D3D12_HEAP_FLAG_SHARED,
            &description,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS(&result.resource)),
        "CreateCommittedResource(capture)");
    check(
        device->CreateCommittedResource(
            &upload_heap,
            D3D12_HEAP_FLAG_NONE,
            &description,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&result.upload)),
        "CreateCommittedResource(upload)");
    void* mapped = nullptr;
    check(
        result.upload->Map(0, nullptr, &mapped),
        "Map(upload)");
    std::memcpy(mapped, pixels.data(), pixels.size());
    result.upload->Unmap(0, nullptr);

    check(
        device->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(&result.allocator)),
        "CreateCommandAllocator(upload)");
    check(
        device->CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            result.allocator.Get(),
            nullptr,
            IID_PPV_ARGS(&result.commands)),
        "CreateCommandList(upload)");
    result.commands->CopyBufferRegion(
        result.resource.Get(),
        0,
        result.upload.Get(),
        0,
        pixels.size());
    const D3D12_RESOURCE_BARRIER barrier{
        D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        D3D12_RESOURCE_BARRIER_FLAG_NONE,
        {
            {
                result.resource.Get(),
                D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COMMON,
            },
        },
    };
    result.commands->ResourceBarrier(1, &barrier);
    check(result.commands->Close(), "Close(upload)");
    ID3D12CommandList* submitted[] = {result.commands.Get()};
    queue->ExecuteCommandLists(1, submitted);

    check(
        device->CreateFence(
            0,
            D3D12_FENCE_FLAG_SHARED,
            IID_PPV_ARGS(&result.ready)),
        "CreateFence(capture)");
    check(
        device->CreateSharedHandle(
            result.resource.Get(),
            nullptr,
            GENERIC_ALL,
            nullptr,
            &result.resource_handle),
        "CreateSharedHandle(capture)");
    check(
        device->CreateSharedHandle(
            result.ready.Get(),
            nullptr,
            GENERIC_ALL,
            nullptr,
            &result.fence_handle),
        "CreateSharedHandle(capture fence)");
    if (signal_ready) {
        check(
            queue->Signal(result.ready.Get(), result.fence_value),
            "Signal(capture)");
    }
    return result;
}

struct SharedTextureInput {
    ComPtr<ID3D12Resource> resource;
    ComPtr<ID3D12Resource> upload;
    ComPtr<ID3D12Fence> ready;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> commands;
    HANDLE resource_handle = nullptr;
    HANDLE fence_handle = nullptr;
    std::uint64_t fence_value = 1;

    SharedTextureInput() = default;
    SharedTextureInput(const SharedTextureInput&) = delete;
    SharedTextureInput& operator=(const SharedTextureInput&) = delete;
    SharedTextureInput(SharedTextureInput&& other) noexcept
        : resource(std::move(other.resource)),
          upload(std::move(other.upload)),
          ready(std::move(other.ready)),
          allocator(std::move(other.allocator)),
          commands(std::move(other.commands)),
          resource_handle(
              std::exchange(other.resource_handle, nullptr)),
          fence_handle(
              std::exchange(other.fence_handle, nullptr)),
          fence_value(other.fence_value) {}

    ~SharedTextureInput() {
        if (resource_handle != nullptr) CloseHandle(resource_handle);
        if (fence_handle != nullptr) CloseHandle(fence_handle);
    }
};

SharedTextureInput upload_capture_texture(
    ID3D12Device* device,
    ID3D12CommandQueue* queue,
    const std::vector<std::uint8_t>& pixels,
    std::uint32_t width,
    std::uint32_t height,
    DXGI_FORMAT format) {
    const D3D12_HEAP_PROPERTIES default_heap{
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL_UNKNOWN,
        1,
        1,
    };
    const D3D12_RESOURCE_DESC texture_description{
        D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        0,
        width,
        height,
        1,
        1,
        format,
        {1, 0},
        D3D12_TEXTURE_LAYOUT_UNKNOWN,
        D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
    };
    SharedTextureInput result;
    check(
        device->CreateCommittedResource(
            &default_heap,
            D3D12_HEAP_FLAG_SHARED,
            &texture_description,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS(&result.resource)),
        "CreateCommittedResource(capture texture)");

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows = 0;
    UINT64 row_bytes = 0;
    UINT64 upload_bytes = 0;
    device->GetCopyableFootprints(
        &texture_description,
        0,
        1,
        0,
        &footprint,
        &rows,
        &row_bytes,
        &upload_bytes);
    const D3D12_HEAP_PROPERTIES upload_heap{
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL_UNKNOWN,
        1,
        1,
    };
    const D3D12_RESOURCE_DESC upload_description{
        D3D12_RESOURCE_DIMENSION_BUFFER,
        0,
        upload_bytes,
        1,
        1,
        1,
        DXGI_FORMAT_UNKNOWN,
        {1, 0},
        D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        D3D12_RESOURCE_FLAG_NONE,
    };
    check(
        device->CreateCommittedResource(
            &upload_heap,
            D3D12_HEAP_FLAG_NONE,
            &upload_description,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&result.upload)),
        "CreateCommittedResource(texture upload)");
    void* mapped = nullptr;
    check(
        result.upload->Map(0, nullptr, &mapped),
        "Map(texture upload)");
    for (std::uint32_t y = 0; y < height; ++y) {
        std::memcpy(
            static_cast<std::uint8_t*>(mapped) +
                footprint.Offset +
                static_cast<std::size_t>(y) *
                    footprint.Footprint.RowPitch,
            pixels.data() +
                static_cast<std::size_t>(y) * width * 4,
            static_cast<std::size_t>(width) * 4);
    }
    result.upload->Unmap(0, nullptr);
    check(
        device->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(&result.allocator)),
        "CreateCommandAllocator(texture upload)");
    check(
        device->CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            result.allocator.Get(),
            nullptr,
            IID_PPV_ARGS(&result.commands)),
        "CreateCommandList(texture upload)");
    D3D12_TEXTURE_COPY_LOCATION destination{};
    destination.pResource = result.resource.Get();
    destination.Type =
        D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    destination.SubresourceIndex = 0;
    D3D12_TEXTURE_COPY_LOCATION source{};
    source.pResource = result.upload.Get();
    source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    source.PlacedFootprint = footprint;
    result.commands->CopyTextureRegion(
        &destination, 0, 0, 0, &source, nullptr);
    const D3D12_RESOURCE_BARRIER barrier{
        D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        D3D12_RESOURCE_BARRIER_FLAG_NONE,
        {
            {
                result.resource.Get(),
                D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COMMON,
            },
        },
    };
    result.commands->ResourceBarrier(1, &barrier);
    check(result.commands->Close(), "Close(texture upload)");
    ID3D12CommandList* submitted[] = {result.commands.Get()};
    queue->ExecuteCommandLists(1, submitted);
    check(
        device->CreateFence(
            0,
            D3D12_FENCE_FLAG_SHARED,
            IID_PPV_ARGS(&result.ready)),
        "CreateFence(capture texture)");
    check(
        device->CreateSharedHandle(
            result.resource.Get(),
            nullptr,
            GENERIC_ALL,
            nullptr,
            &result.resource_handle),
        "CreateSharedHandle(capture texture)");
    check(
        device->CreateSharedHandle(
            result.ready.Get(),
            nullptr,
            GENERIC_ALL,
            nullptr,
            &result.fence_handle),
        "CreateSharedHandle(capture texture fence)");
    check(
        queue->Signal(result.ready.Get(), result.fence_value),
        "Signal(capture texture)");
    return result;
}

std::vector<std::uint8_t> make_pixels(
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t frame) {
    std::vector<std::uint8_t> result(
        static_cast<std::size_t>(width) * height * 4);
    for (std::size_t index = 0; index < result.size() / 4; ++index) {
        result[index * 4] =
            static_cast<std::uint8_t>((index * 13 + frame * 31) % 251);
        result[index * 4 + 1] =
            static_cast<std::uint8_t>((index * 29 + frame * 7) % 253);
        result[index * 4 + 2] =
            static_cast<std::uint8_t>((index * 43 + frame * 17) % 255);
        result[index * 4 + 3] = 255;
    }
    return result;
}

std::vector<float> read_depth(
    ID3D12Device* device,
    ID3D12CommandQueue* queue,
    const dad_d3d12_output_descriptor& descriptor) {
    ComPtr<ID3D12Resource> output;
    ComPtr<ID3D12Fence> ready;
    check(
        device->OpenSharedHandle(
            reinterpret_cast<HANDLE>(
                static_cast<std::uintptr_t>(
                    descriptor.shared_resource_handle)),
            IID_PPV_ARGS(&output)),
        "OpenSharedHandle(depth)");
    check(
        device->OpenSharedHandle(
            reinterpret_cast<HANDLE>(
                static_cast<std::uintptr_t>(
                    descriptor.ready_fence_handle)),
            IID_PPV_ARGS(&ready)),
        "OpenSharedHandle(depth fence)");
    check(
        queue->Wait(ready.Get(), descriptor.ready_fence_value),
        "Wait(depth fence)");

    const D3D12_HEAP_PROPERTIES readback_heap{
        D3D12_HEAP_TYPE_READBACK,
        D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL_UNKNOWN,
        1,
        1,
    };
    const D3D12_RESOURCE_DESC description{
        D3D12_RESOURCE_DIMENSION_BUFFER,
        0,
        descriptor.byte_size,
        1,
        1,
        1,
        DXGI_FORMAT_UNKNOWN,
        {1, 0},
        D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        D3D12_RESOURCE_FLAG_NONE,
    };
    ComPtr<ID3D12Resource> readback;
    check(
        device->CreateCommittedResource(
            &readback_heap,
            D3D12_HEAP_FLAG_NONE,
            &description,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS(&readback)),
        "CreateCommittedResource(readback)");
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> commands;
    check(
        device->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(&allocator)),
        "CreateCommandAllocator(readback)");
    check(
        device->CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            allocator.Get(),
            nullptr,
            IID_PPV_ARGS(&commands)),
        "CreateCommandList(readback)");
    const D3D12_RESOURCE_BARRIER barrier{
        D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        D3D12_RESOURCE_BARRIER_FLAG_NONE,
        {
            {
                output.Get(),
                D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            },
        },
    };
    commands->ResourceBarrier(1, &barrier);
    commands->CopyBufferRegion(
        readback.Get(), 0, output.Get(), 0, descriptor.byte_size);
    const D3D12_RESOURCE_BARRIER restore{
        D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        D3D12_RESOURCE_BARRIER_FLAG_NONE,
        {
            {
                output.Get(),
                D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            },
        },
    };
    commands->ResourceBarrier(1, &restore);
    check(commands->Close(), "Close(readback)");
    ID3D12CommandList* submitted[] = {commands.Get()};
    queue->ExecuteCommandLists(1, submitted);
    ComPtr<ID3D12Fence> complete;
    check_device(
        device->CreateFence(
            0,
            D3D12_FENCE_FLAG_NONE,
            IID_PPV_ARGS(&complete)),
        device,
        "CreateFence(readback)");
    check(queue->Signal(complete.Get(), 1), "Signal(readback)");
    cpu_wait(complete.Get(), 1);

    std::vector<float> result(
        static_cast<std::size_t>(descriptor.width) *
        descriptor.height);
    void* mapped = nullptr;
    check(readback->Map(0, nullptr, &mapped), "Map(readback)");
    std::memcpy(
        result.data(), mapped, result.size() * sizeof(float));
    readback->Unmap(0, nullptr);
    return result;
}

std::vector<float> read_depth_texture(
    ID3D12Device* device,
    ID3D12CommandQueue* queue,
    const dad_d3d12_texture_output_descriptor& descriptor) {
    ComPtr<ID3D12Resource> output;
    ComPtr<ID3D12Fence> ready;
    check(
        device->OpenSharedHandle(
            reinterpret_cast<HANDLE>(
                static_cast<std::uintptr_t>(
                    descriptor.shared_texture_handle)),
            IID_PPV_ARGS(&output)),
        "OpenSharedHandle(depth texture)");
    check(
        device->OpenSharedHandle(
            reinterpret_cast<HANDLE>(
                static_cast<std::uintptr_t>(
                    descriptor.ready_fence_handle)),
            IID_PPV_ARGS(&ready)),
        "OpenSharedHandle(depth texture fence)");
    const D3D12_RESOURCE_DESC output_description =
        output->GetDesc();
    if (output_description.Dimension !=
            D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
        output_description.Format != DXGI_FORMAT_R32_FLOAT ||
        output_description.Width != descriptor.width ||
        output_description.Height != descriptor.height) {
        throw std::runtime_error(
            "leased depth texture has the wrong D3D12 descriptor");
    }
    check(
        queue->Wait(ready.Get(), descriptor.ready_fence_value),
        "Wait(depth texture fence)");
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows = 0;
    UINT64 row_bytes = 0;
    UINT64 readback_bytes = 0;
    device->GetCopyableFootprints(
        &output_description,
        0,
        1,
        0,
        &footprint,
        &rows,
        &row_bytes,
        &readback_bytes);
    const D3D12_HEAP_PROPERTIES readback_heap{
        D3D12_HEAP_TYPE_READBACK,
        D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL_UNKNOWN,
        1,
        1,
    };
    const D3D12_RESOURCE_DESC readback_description{
        D3D12_RESOURCE_DIMENSION_BUFFER,
        0,
        readback_bytes,
        1,
        1,
        1,
        DXGI_FORMAT_UNKNOWN,
        {1, 0},
        D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        D3D12_RESOURCE_FLAG_NONE,
    };
    ComPtr<ID3D12Resource> readback;
    check(
        device->CreateCommittedResource(
            &readback_heap,
            D3D12_HEAP_FLAG_NONE,
            &readback_description,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS(&readback)),
        "CreateCommittedResource(texture readback)");
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> commands;
    check(
        device->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(&allocator)),
        "CreateCommandAllocator(texture readback)");
    check(
        device->CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            allocator.Get(),
            nullptr,
            IID_PPV_ARGS(&commands)),
        "CreateCommandList(texture readback)");
    const D3D12_RESOURCE_BARRIER barrier{
        D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        D3D12_RESOURCE_BARRIER_FLAG_NONE,
        {
            {
                output.Get(),
                D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            },
        },
    };
    commands->ResourceBarrier(1, &barrier);
    D3D12_TEXTURE_COPY_LOCATION destination{};
    destination.pResource = readback.Get();
    destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    destination.PlacedFootprint = footprint;
    D3D12_TEXTURE_COPY_LOCATION source{};
    source.pResource = output.Get();
    source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    source.SubresourceIndex = 0;
    commands->CopyTextureRegion(
        &destination, 0, 0, 0, &source, nullptr);
    const D3D12_RESOURCE_BARRIER restore{
        D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        D3D12_RESOURCE_BARRIER_FLAG_NONE,
        {
            {
                output.Get(),
                D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            },
        },
    };
    commands->ResourceBarrier(1, &restore);
    check(commands->Close(), "Close(texture readback)");
    ID3D12CommandList* submitted[] = {commands.Get()};
    queue->ExecuteCommandLists(1, submitted);
    ComPtr<ID3D12Fence> complete;
    check_device(
        device->CreateFence(
            0,
            D3D12_FENCE_FLAG_NONE,
            IID_PPV_ARGS(&complete)),
        device,
        "CreateFence(texture readback)");
    check(
        queue->Signal(complete.Get(), 1),
        "Signal(texture readback)");
    cpu_wait(complete.Get(), 1);
    std::vector<float> result(
        static_cast<std::size_t>(descriptor.width) *
        descriptor.height);
    void* mapped = nullptr;
    check(
        readback->Map(0, nullptr, &mapped),
        "Map(texture readback)");
    for (std::uint32_t y = 0; y < descriptor.height; ++y) {
        std::memcpy(
            result.data() +
                static_cast<std::size_t>(y) * descriptor.width,
            static_cast<const std::uint8_t*>(mapped) +
                footprint.Offset +
                static_cast<std::size_t>(y) *
                    footprint.Footprint.RowPitch,
            static_cast<std::size_t>(descriptor.width) *
                sizeof(float));
    }
    readback->Unmap(0, nullptr);
    return result;
}

void wait_depth_ready(
    ID3D12Device* device,
    const dad_d3d12_output_descriptor& descriptor) {
    ComPtr<ID3D12Fence> ready;
    check(
        device->OpenSharedHandle(
            reinterpret_cast<HANDLE>(
                static_cast<std::uintptr_t>(
                    descriptor.ready_fence_handle)),
            IID_PPV_ARGS(&ready)),
        "OpenSharedHandle(depth fence)");
    cpu_wait(ready.Get(), descriptor.ready_fence_value);
}

void wait_depth_texture_ready(
    ID3D12Device* device,
    const dad_d3d12_texture_output_descriptor& descriptor) {
    ComPtr<ID3D12Fence> ready;
    check(
        device->OpenSharedHandle(
            reinterpret_cast<HANDLE>(
                static_cast<std::uintptr_t>(
                    descriptor.ready_fence_handle)),
            IID_PPV_ARGS(&ready)),
        "OpenSharedHandle(depth texture fence)");
    cpu_wait(ready.Get(), descriptor.ready_fence_value);
}

void validate_depth(const std::vector<float>& depth) {
    const auto bounds =
        std::minmax_element(depth.begin(), depth.end());
    if (bounds.first == depth.end() ||
        !std::isfinite(*bounds.first) ||
        !std::isfinite(*bounds.second) ||
        *bounds.second <= *bounds.first) {
        throw std::runtime_error(
            "GPU depth output is not a finite non-constant map");
    }
}

void compare_reference(
    dad_context* context,
    const std::vector<std::uint8_t>& bgra,
    std::uint32_t width,
    std::uint32_t height,
    const std::vector<float>& gpu) {
    std::vector<std::uint8_t> bgr(
        static_cast<std::size_t>(width) * height * 3);
    for (std::size_t index = 0; index < bgr.size() / 3; ++index) {
        bgr[index * 3] = bgra[index * 4];
        bgr[index * 3 + 1] = bgra[index * 4 + 1];
        bgr[index * 3 + 2] = bgra[index * 4 + 2];
    }
    std::vector<float> reference(
        static_cast<std::size_t>(width) * height);
    check(
        dad_infer_bgr8(
            context,
            bgr.data(),
            static_cast<std::int32_t>(width),
            static_cast<std::int32_t>(height),
            static_cast<std::ptrdiff_t>(width * 3),
            140,
            reference.data(),
            reference.size()),
        "dad_infer_bgr8(reference)");
    float maximum_difference = 0.0f;
    std::size_t maximum_index = 0;
    float reference_range =
        *std::max_element(reference.begin(), reference.end()) -
        *std::min_element(reference.begin(), reference.end());
    for (std::size_t index = 0; index < gpu.size(); ++index) {
        const float difference =
            std::abs(gpu[index] - reference[index]);
        if (difference > maximum_difference) {
            maximum_difference = difference;
            maximum_index = index;
        }
    }
    const float relative =
        maximum_difference /
        std::max(reference_range, std::numeric_limits<float>::epsilon());
    std::cout << "CPU correlation max/range=" << relative << '\n';
    if (relative >= 0.01f) {
        throw std::runtime_error(
            "GPU-resident path exceeds the 1% CPU-output deviation gate "
            "at index " + std::to_string(maximum_index) +
            ": gpu=" + std::to_string(gpu[maximum_index]) +
            " reference=" +
            std::to_string(reference[maximum_index]) +
            " maximum_difference=" +
            std::to_string(maximum_difference) +
            " reference_range=" +
            std::to_string(reference_range));
    }
}

std::filesystem::path model_path() {
    if (const char* configured = std::getenv("DAD_VITS_MODEL")) {
        return configured;
    }
#if defined(DAD_DEFAULT_VITS_MODEL)
    return DAD_DEFAULT_VITS_MODEL;
#else
    return {};
#endif
}

}  // namespace

int main() try {
    const std::filesystem::path model = model_path();
    if (model.empty() || !std::filesystem::exists(model)) {
        std::cout << "DAD vits model unavailable\n";
        return 77;
    }
    dad_gpu_capabilities probed{};
    probed.struct_size = sizeof(probed);
    check(
        dad_probe_gpu_capabilities(0, &probed),
        "dad_probe_gpu_capabilities");
    dad_create_options options{
        sizeof(options), DAD_ABI_VERSION, DAD_ENCODER_VITS, 0, 0};
    dad_context* context = nullptr;
    check(
        dad_create(model.string().c_str(), &options, &context),
        "dad_create");

    dad_gpu_capabilities capabilities{};
    capabilities.struct_size = sizeof(capabilities);
    check(
        dad_get_gpu_capabilities(context, &capabilities),
        "dad_get_gpu_capabilities");
    if (probed.flags != capabilities.flags ||
        probed.adapter_luid != capabilities.adapter_luid ||
        probed.maximum_in_flight_jobs !=
            capabilities.maximum_in_flight_jobs) {
        throw std::runtime_error(
            "pre-model and loaded-model GPU capabilities disagree");
    }
    constexpr std::uint64_t required_capabilities =
        DAD_GPU_CAP_D3D12_SHARED_INPUT |
        DAD_GPU_CAP_D3D12_FENCE_WAIT |
        DAD_GPU_CAP_D3D12_SHARED_OUTPUT |
        DAD_GPU_CAP_D3D12_FENCE_SIGNAL |
        DAD_GPU_CAP_ASYNC_SUBMIT |
        DAD_GPU_CAP_CANCELLATION |
        DAD_GPU_CAP_NO_HOST_PIXEL_STAGING |
        DAD_GPU_CAP_NO_HOST_DEPTH_STAGING |
        DAD_GPU_CAP_D3D12_SHARED_TEXTURE_INPUT |
        DAD_GPU_CAP_D3D12_SHARED_TEXTURE_OUTPUT;
    if ((capabilities.flags & required_capabilities) !=
            required_capabilities ||
        capabilities.adapter_luid == 0) {
        std::cout << "GPU capability flags=0x" << std::hex
                  << capabilities.flags << std::dec
                  << " adapter_luid=" << capabilities.adapter_luid
                  << '\n';
        dad_destroy(context);
        std::cout << "complete D3D12/Vulkan interop unavailable\n";
        return 77;
    }
    if (capabilities.maximum_in_flight_jobs != 3) {
        throw std::runtime_error(
            "DAD did not report the three-slot GPU job pool");
    }
    ComPtr<ID3D12Device> device =
        matching_device(capabilities.adapter_luid);
    ComPtr<ID3D12CommandQueue> queue =
        make_queue(device.Get());

    constexpr std::uint32_t width = 73;
    constexpr std::uint32_t height = 51;
    for (std::uint32_t frame = 0; frame < 3; ++frame) {
        const std::vector<std::uint8_t> pixels =
            make_pixels(width, height, frame);
        SharedInput input =
            upload_capture(device.Get(), queue.Get(), pixels, true);
        dad_transfer_counters before{};
        before.struct_size = sizeof(before);
        check(
            dad_get_transfer_counters(context, &before),
            "dad_get_transfer_counters(before)");
        dad_d3d12_submit_request request{};
        request.struct_size = sizeof(request);
        request.abi_version = DAD_ABI_VERSION;
        request.shared_resource_handle =
            reinterpret_cast<std::uintptr_t>(input.resource_handle);
        request.resource_byte_size = pixels.size();
        request.width = width;
        request.height = height;
        request.row_stride_bytes = width * 4;
        request.pixel_format = DAD_GPU_PIXEL_BGRA8;
        request.input_size = 140;
        request.wait_fence_handle =
            reinterpret_cast<std::uintptr_t>(input.fence_handle);
        request.wait_fence_value = input.fence_value;
        request.source_frame_id = 9000 + frame;
        request.timestamp_ns = 123456789 + frame;
        dad_gpu_job* job = nullptr;
        check(
            dad_submit_d3d12(context, &request, &job),
            "dad_submit_d3d12");

        // Submit imported duplicate handles, not caller-owned handles.
        CloseHandle(input.resource_handle);
        input.resource_handle = nullptr;
        CloseHandle(input.fence_handle);
        input.fence_handle = nullptr;
        input.resource.Reset();
        input.ready.Reset();

        dad_d3d12_output_descriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        dad_gpu_output_lease* lease = nullptr;
        check(
            dad_gpu_output_acquire(
                job, 0, &descriptor, &lease),
            "dad_gpu_output_acquire");
        if (descriptor.source_frame_id != request.source_frame_id ||
            descriptor.timestamp_ns != request.timestamp_ns ||
            descriptor.width != width ||
            descriptor.height != height ||
            descriptor.pixel_format !=
                DAD_GPU_PIXEL_DEPTH_FLOAT32) {
            throw std::runtime_error(
                "GPU output metadata correlation failed");
        }

        wait_depth_ready(device.Get(), descriptor);
        dad_gpu_job_status status{};
        for (std::uint32_t attempt = 0; attempt < 1000; ++attempt) {
            status = {};
            status.struct_size = sizeof(status);
            check(
                dad_gpu_job_poll(job, &status),
                "dad_gpu_job_poll");
            if (status.state != DAD_GPU_JOB_RUNNING) break;
            Sleep(1);
        }
        if (status.state != DAD_GPU_JOB_COMPLETE ||
            status.source_frame_id != request.source_frame_id) {
            throw std::runtime_error(
                "GPU completion state correlation failed");
        }
        // The lease, not the job handle, owns descriptor handle lifetime.
        dad_gpu_job_release(job);
        const std::vector<float> depth =
            read_depth(device.Get(), queue.Get(), descriptor);
        validate_depth(depth);
        dad_gpu_output_release(lease);

        dad_transfer_counters after{};
        after.struct_size = sizeof(after);
        check(
            dad_get_transfer_counters(context, &after),
            "dad_get_transfer_counters(after)");
        if (after.tensor_upload_bytes !=
                before.tensor_upload_bytes ||
            after.tensor_download_bytes !=
                before.tensor_download_bytes) {
            throw std::runtime_error(
                "DAD performed CPU tensor staging on the GPU path");
        }
        if (frame == 0) {
            compare_reference(
                context, pixels, width, height, depth);
        }
        std::cout << "frame " << request.source_frame_id
                  << " GPU-resident graph passed\n";
    }

    for (std::uint32_t frame = 0; frame < 3; ++frame) {
        const std::vector<std::uint8_t> pixels_bgra =
            make_pixels(width, height, frame + 20);
        std::vector<std::uint8_t> texture_pixels = pixels_bgra;
        const bool rgba = frame == 1;
        if (rgba) {
            for (std::size_t index = 0;
                 index < texture_pixels.size() / 4;
                 ++index) {
                std::swap(
                    texture_pixels[index * 4],
                    texture_pixels[index * 4 + 2]);
            }
        }
        SharedTextureInput input = upload_capture_texture(
            device.Get(),
            queue.Get(),
            texture_pixels,
            width,
            height,
            rgba
                ? DXGI_FORMAT_R8G8B8A8_UNORM
                : DXGI_FORMAT_B8G8R8A8_UNORM);
        dad_transfer_counters before{};
        before.struct_size = sizeof(before);
        check(
            dad_get_transfer_counters(context, &before),
            "dad_get_transfer_counters(texture before)");
        dad_d3d12_texture_submit_request request{};
        request.struct_size = sizeof(request);
        request.abi_version = DAD_ABI_VERSION;
        request.shared_texture_handle =
            reinterpret_cast<std::uintptr_t>(input.resource_handle);
        request.width = width;
        request.height = height;
        request.pixel_format =
            rgba ? DAD_GPU_PIXEL_RGBA8 : DAD_GPU_PIXEL_BGRA8;
        request.input_size = 140;
        request.wait_fence_handle =
            reinterpret_cast<std::uintptr_t>(input.fence_handle);
        request.wait_fence_value = input.fence_value;
        request.source_frame_id = 10000 + frame;
        request.timestamp_ns = 223456789 + frame;
        dad_gpu_job* job = nullptr;
        check(
            dad_submit_d3d12_texture(
                context, &request, &job),
            "dad_submit_d3d12_texture");
        CloseHandle(input.resource_handle);
        input.resource_handle = nullptr;
        CloseHandle(input.fence_handle);
        input.fence_handle = nullptr;
        input.resource.Reset();
        input.ready.Reset();

        dad_d3d12_texture_output_descriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        dad_gpu_output_lease* lease = nullptr;
        check(
            dad_gpu_texture_output_acquire(
                job, 0, &descriptor, &lease),
            "dad_gpu_texture_output_acquire");
        if (descriptor.source_frame_id != request.source_frame_id ||
            descriptor.timestamp_ns != request.timestamp_ns ||
            descriptor.width != width ||
            descriptor.height != height ||
            descriptor.pixel_format !=
                DAD_GPU_PIXEL_DEPTH_FLOAT32) {
            throw std::runtime_error(
                "GPU texture output metadata correlation failed");
        }
        wait_depth_texture_ready(device.Get(), descriptor);
        dad_gpu_job_status status{};
        for (std::uint32_t attempt = 0; attempt < 1000; ++attempt) {
            status = {};
            status.struct_size = sizeof(status);
            check(
                dad_gpu_job_poll(job, &status),
                "dad_gpu_job_poll(texture)");
            if (status.state != DAD_GPU_JOB_RUNNING) break;
            Sleep(1);
        }
        if (status.state != DAD_GPU_JOB_COMPLETE ||
            status.source_frame_id != request.source_frame_id) {
            throw std::runtime_error(
                "GPU texture completion correlation failed");
        }
        dad_gpu_job_release(job);
        const std::vector<float> depth = read_depth_texture(
            device.Get(), queue.Get(), descriptor);
        validate_depth(depth);
        dad_gpu_output_release(lease);

        dad_transfer_counters after{};
        after.struct_size = sizeof(after);
        check(
            dad_get_transfer_counters(context, &after),
            "dad_get_transfer_counters(texture after)");
        if (after.tensor_upload_bytes !=
                before.tensor_upload_bytes ||
            after.tensor_download_bytes !=
                before.tensor_download_bytes) {
            throw std::runtime_error(
                "DAD staged texture pixels or depth through the CPU");
        }
        if (frame == 0 || rgba) {
            compare_reference(
                context,
                pixels_bgra,
                width,
                height,
                depth);
        }
        std::cout << "texture frame "
                  << request.source_frame_id
                  << (rgba ? " RGBA" : " BGRA")
                  << " GPU-resident graph passed\n";
    }

    dad_transfer_counters pool_before{};
    pool_before.struct_size = sizeof(pool_before);
    check(
        dad_get_transfer_counters(context, &pool_before),
        "dad_get_transfer_counters(pool before)");
    std::array<dad_gpu_output_lease*, 3> retained_leases{};
    std::array<dad_d3d12_texture_output_descriptor, 3>
        retained_outputs{};
    for (std::uint32_t frame = 0; frame < 3; ++frame) {
        const std::vector<std::uint8_t> pool_pixels =
            make_pixels(width, height, frame + 40);
        SharedTextureInput input = upload_capture_texture(
            device.Get(),
            queue.Get(),
            pool_pixels,
            width,
            height,
            DXGI_FORMAT_B8G8R8A8_UNORM);
        dad_d3d12_texture_submit_request pool_request{};
        pool_request.struct_size = sizeof(pool_request);
        pool_request.abi_version = DAD_ABI_VERSION;
        pool_request.shared_texture_handle =
            reinterpret_cast<std::uintptr_t>(
                input.resource_handle);
        pool_request.width = width;
        pool_request.height = height;
        pool_request.pixel_format = DAD_GPU_PIXEL_BGRA8;
        pool_request.input_size = 140;
        pool_request.wait_fence_handle =
            reinterpret_cast<std::uintptr_t>(
                input.fence_handle);
        pool_request.wait_fence_value = input.fence_value;
        pool_request.source_frame_id = 11000 + frame;
        pool_request.timestamp_ns = 323456789 + frame;
        dad_gpu_job* pool_job = nullptr;
        check(
            dad_submit_d3d12_texture(
                context, &pool_request, &pool_job),
            "dad_submit_d3d12_texture(pool)");
        CloseHandle(input.resource_handle);
        input.resource_handle = nullptr;
        CloseHandle(input.fence_handle);
        input.fence_handle = nullptr;
        input.resource.Reset();
        input.ready.Reset();

        retained_outputs[frame].struct_size =
            sizeof(retained_outputs[frame]);
        check(
            dad_gpu_texture_output_acquire(
                pool_job,
                0,
                &retained_outputs[frame],
                &retained_leases[frame]),
            "dad_gpu_texture_output_acquire(pool)");
        wait_depth_texture_ready(
            device.Get(), retained_outputs[frame]);
        dad_gpu_job_status pool_status{};
        pool_status.struct_size = sizeof(pool_status);
        check(
            dad_gpu_job_poll(pool_job, &pool_status),
            "dad_gpu_job_poll(pool)");
        if (pool_status.state != DAD_GPU_JOB_COMPLETE ||
            pool_status.source_frame_id !=
                pool_request.source_frame_id ||
            retained_outputs[frame].source_frame_id !=
                pool_request.source_frame_id ||
            retained_outputs[frame].timestamp_ns !=
                pool_request.timestamp_ns) {
            throw std::runtime_error(
                "retained GPU slot correlation failed");
        }
        const std::vector<float> pool_depth =
            read_depth_texture(
                device.Get(),
                queue.Get(),
                retained_outputs[frame]);
        validate_depth(pool_depth);
        dad_gpu_job_release(pool_job);
    }
    for (std::size_t left = 0;
         left < retained_outputs.size();
         ++left) {
        for (std::size_t right = left + 1;
             right < retained_outputs.size();
             ++right) {
            if (retained_outputs[left].shared_texture_handle ==
                    retained_outputs[right].shared_texture_handle ||
                retained_outputs[left].ready_fence_handle ==
                    retained_outputs[right].ready_fence_handle) {
                throw std::runtime_error(
                    "live GPU slots did not expose distinct handles");
            }
        }
    }

    const std::vector<std::uint8_t> dropped_pixels =
        make_pixels(width, height, 43);
    SharedTextureInput dropped_input = upload_capture_texture(
        device.Get(),
        queue.Get(),
        dropped_pixels,
        width,
        height,
        DXGI_FORMAT_B8G8R8A8_UNORM);
    dad_d3d12_texture_submit_request dropped_request{};
    dropped_request.struct_size = sizeof(dropped_request);
    dropped_request.abi_version = DAD_ABI_VERSION;
    dropped_request.shared_texture_handle =
        reinterpret_cast<std::uintptr_t>(
            dropped_input.resource_handle);
    dropped_request.width = width;
    dropped_request.height = height;
    dropped_request.pixel_format = DAD_GPU_PIXEL_BGRA8;
    dropped_request.input_size = 140;
    dropped_request.wait_fence_handle =
        reinterpret_cast<std::uintptr_t>(
            dropped_input.fence_handle);
    dropped_request.wait_fence_value =
        dropped_input.fence_value;
    dropped_request.source_frame_id = 11003;
    dropped_request.timestamp_ns = 323456792;
    dad_gpu_job* dropped_job = nullptr;
    if (dad_submit_d3d12_texture(
            context,
            &dropped_request,
            &dropped_job) != DAD_STATUS_INVALID_STATE ||
        dropped_job != nullptr) {
        throw std::runtime_error(
            "fourth live GPU job was not rejected");
    }

    const std::uint64_t reusable_texture =
        retained_outputs[0].shared_texture_handle;
    const std::uint64_t reusable_fence =
        retained_outputs[0].ready_fence_handle;
    const std::uint64_t previous_fence_value =
        retained_outputs[0].ready_fence_value;
    dad_gpu_output_release(retained_leases[0]);
    retained_leases[0] = nullptr;

    dad_gpu_job* reused_job = nullptr;
    check(
        dad_submit_d3d12_texture(
            context, &dropped_request, &reused_job),
        "dad_submit_d3d12_texture(reuse)");
    CloseHandle(dropped_input.resource_handle);
    dropped_input.resource_handle = nullptr;
    CloseHandle(dropped_input.fence_handle);
    dropped_input.fence_handle = nullptr;
    dropped_input.resource.Reset();
    dropped_input.ready.Reset();
    dad_d3d12_texture_output_descriptor reused_output{};
    reused_output.struct_size = sizeof(reused_output);
    dad_gpu_output_lease* reused_lease = nullptr;
    check(
        dad_gpu_texture_output_acquire(
            reused_job, 0, &reused_output, &reused_lease),
        "dad_gpu_texture_output_acquire(reuse)");
    if (reused_output.shared_texture_handle !=
            reusable_texture ||
        reused_output.ready_fence_handle != reusable_fence ||
        reused_output.ready_fence_value <=
            previous_fence_value ||
        reused_output.source_frame_id !=
            dropped_request.source_frame_id ||
        reused_output.timestamp_ns !=
            dropped_request.timestamp_ns) {
        throw std::runtime_error(
            "released GPU slot was not stably reused");
    }
    wait_depth_texture_ready(device.Get(), reused_output);
    const std::vector<float> reused_depth =
        read_depth_texture(
            device.Get(), queue.Get(), reused_output);
    validate_depth(reused_depth);
    dad_gpu_job_release(reused_job);
    dad_gpu_output_release(reused_lease);
    for (std::size_t index = 1;
         index < retained_leases.size();
         ++index) {
        dad_gpu_output_release(retained_leases[index]);
        retained_leases[index] = nullptr;
    }
    dad_transfer_counters pool_after{};
    pool_after.struct_size = sizeof(pool_after);
    check(
        dad_get_transfer_counters(context, &pool_after),
        "dad_get_transfer_counters(pool after)");
    if (pool_after.tensor_upload_bytes !=
            pool_before.tensor_upload_bytes ||
        pool_after.tensor_download_bytes !=
            pool_before.tensor_download_bytes) {
        throw std::runtime_error(
            "GPU slot pool staged pixels or depth through the CPU");
    }
    compare_reference(
        context,
        dropped_pixels,
        width,
        height,
        reused_depth);
    std::cout
        << "three retained texture leases, exhaustion, and stable "
           "slot reuse passed\n";

    // Hold the imported wait unsignaled so cancellation is deterministic.
    const std::vector<std::uint8_t> pixels =
        make_pixels(width, height, 99);
    SharedInput blocked =
        upload_capture(device.Get(), queue.Get(), pixels, false);
    dad_d3d12_submit_request request{};
    request.struct_size = sizeof(request);
    request.abi_version = DAD_ABI_VERSION;
    request.shared_resource_handle =
        reinterpret_cast<std::uintptr_t>(blocked.resource_handle);
    request.resource_byte_size = pixels.size();
    request.width = width;
    request.height = height;
    request.row_stride_bytes = width * 4;
    request.pixel_format = DAD_GPU_PIXEL_BGRA8;
    request.input_size = 140;
    request.wait_fence_handle =
        reinterpret_cast<std::uintptr_t>(blocked.fence_handle);
    request.wait_fence_value = blocked.fence_value;
    request.source_frame_id = 9999;
    dad_gpu_job* cancelled = nullptr;
    check(
        dad_submit_d3d12(context, &request, &cancelled),
        "dad_submit_d3d12(cancel)");
    check(
        dad_gpu_job_cancel(cancelled),
        "dad_gpu_job_cancel");
    dad_gpu_job_status cancelled_status{};
    cancelled_status.struct_size = sizeof(cancelled_status);
    check(
        dad_gpu_job_poll(cancelled, &cancelled_status),
        "dad_gpu_job_poll(cancel)");
    if (cancelled_status.state != DAD_GPU_JOB_CANCELLED ||
        cancelled_status.source_frame_id != request.source_frame_id) {
        throw std::runtime_error(
            "cancelled GPU job state correlation failed");
    }
    dad_d3d12_output_descriptor cancelled_output{};
    cancelled_output.struct_size = sizeof(cancelled_output);
    dad_gpu_output_lease* cancelled_lease = nullptr;
    if (dad_gpu_output_acquire(
            cancelled,
            0,
            &cancelled_output,
            &cancelled_lease) == DAD_STATUS_OK) {
        throw std::runtime_error(
            "cancelled GPU job exposed an output");
    }

    // Destroying the public context is safe while the job retains the model.
    dad_destroy(context);
    context = nullptr;
    check(
        queue->Signal(blocked.ready.Get(), blocked.fence_value),
        "Signal(cancelled capture)");
    dad_gpu_job_release(cancelled);
    std::cout
        << "cancellation, retained fence lifetime, and shutdown passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
