#include "executor.h"
#include "gpu_pipeline.h"
#include <dxgi1_6.h>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <atomic>
namespace dad {
using inferbridge::rtx::check_cuda;
GpuCapabilities probe_rtx_gpu_capabilities(int index) {
    int count=0;check_cuda(cudaGetDeviceCount(&count),"enumerate CUDA adapters");
    if(index<0||index>=count)throw std::invalid_argument("CUDA adapter index is out of range");
    cudaDeviceProp p{};check_cuda(cudaGetDeviceProperties(&p,index),"query CUDA adapter");
    const int sm=p.major*10+p.minor;
    if(sm!=75&&sm!=80&&sm!=86&&sm!=89&&sm!=120&&sm!=121)return {};
    GpuCapabilities result{};memcpy(&result.adapter_luid,p.luid,8);
    result.flags=DAD_GPU_CAP_D3D12_SHARED_TEXTURE_INPUT|DAD_GPU_CAP_D3D12_SHARED_TEXTURE_OUTPUT|
        DAD_GPU_CAP_D3D12_FENCE_WAIT|DAD_GPU_CAP_D3D12_FENCE_SIGNAL|DAD_GPU_CAP_ASYNC_SUBMIT|
        DAD_GPU_CAP_CANCELLATION|DAD_GPU_CAP_NO_HOST_PIXEL_STAGING|DAD_GPU_CAP_NO_HOST_DEPTH_STAGING;
    result.maximum_in_flight_jobs=3;return result;
}
struct RtxState {
    int device_index;
    GpuCapabilities capabilities;
    Microsoft::WRL::ComPtr<ID3D12Device> device;
    std::unique_ptr<inferbridge::rtx::GpuPipeline> pipeline;
    Microsoft::WRL::ComPtr<ID3D12Resource> host_texture;
    Microsoft::WRL::ComPtr<ID3D12Fence> host_fence;
    HANDLE host_texture_handle=nullptr,host_fence_handle=nullptr;
    uint32_t host_width=0,host_height=0;DXGI_FORMAT host_format=DXGI_FORMAT_UNKNOWN;
    uint64_t host_identity=0,host_fence_value=0;
    std::atomic<uint64_t> upload_bytes{0};
    std::unique_ptr<inferbridge::rtx::SharedTexture> host_import;
    explicit RtxState(const std::string& path,int index):device_index(index),capabilities(probe_rtx_gpu_capabilities(index)) {
        if(!capabilities.flags)throw std::runtime_error("selected adapter does not support TensorRT for RTX");
        check_cuda(cudaSetDevice(index),"select RTX adapter");
        LUID luid{};memcpy(&luid,&capabilities.adapter_luid,8);
        Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
        Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
        if(FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))||
           FAILED(factory->EnumAdapterByLuid(luid,IID_PPV_ARGS(&adapter)))||
           FAILED(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_11_0,IID_PPV_ARGS(&device))))
            throw std::runtime_error("cannot create the matching RTX D3D12 device");
        std::ostringstream id;auto bytes=reinterpret_cast<const unsigned char*>(&luid);
        for(int i=0;i<8;++i)id<<std::hex<<std::setfill('0')<<std::setw(2)<<static_cast<int>(bytes[i]);
        auto engine=std::filesystem::u8path(path);
        auto cache=engine.parent_path()/std::filesystem::u8path("gpu-"+id.str())/engine.filename();
        cache.replace_extension(".cache");
        pipeline=std::make_unique<inferbridge::rtx::GpuPipeline>(device.Get(),engine,cache,engine.filename().u8string().rfind("metric_",0)==0);
    }
    ~RtxState(){cudaSetDevice(device_index);pipeline.reset();host_import.reset();if(host_texture_handle)CloseHandle(host_texture_handle);if(host_fence_handle)CloseHandle(host_fence_handle);}
};
class RtxJob final:public GpuJob {
    std::shared_ptr<RtxState> state_;
    GpuOutput output_;
    cudaEvent_t completion_=nullptr;
    std::atomic<bool> cancelled_{false};
public:
    RtxJob(std::shared_ptr<RtxState> s,GpuOutput output):state_(std::move(s)),output_(output){check_cuda(cudaEventCreateWithFlags(&completion_,cudaEventDisableTiming),"create job event");}
    ~RtxJob(){cudaSetDevice(state_->device_index);if(completion_)cudaEventDestroy(completion_);}
    cudaEvent_t completion()const{return completion_;}
    dad_gpu_job_state state() const override {
        check_cuda(cudaSetDevice(state_->device_index),"select RTX polling adapter");
        auto result=cudaEventQuery(completion_);
        if(result==cudaErrorNotReady)return DAD_GPU_JOB_RUNNING;
        check_cuda(result,"query job completion");
        return cancelled_?DAD_GPU_JOB_CANCELLED:DAD_GPU_JOB_COMPLETE;
    }
    void cancel() override {cancelled_=true;}
    GpuOutput output() const override {
        if(state()!=DAD_GPU_JOB_COMPLETE)throw std::logic_error("RTX depth is not ready");
        return output_;
    }
};
class RtxExecutor final:public Executor {
    std::shared_ptr<RtxState> state_;
public:
    RtxExecutor(const std::string& path,int index):state_(std::make_shared<RtxState>(path,index)){}
    void infer(const float*,int,int,float*) override {throw std::runtime_error("RTX playback requires GPU texture bindings");}
    void infer_resized(const float*,int,int,float*,int,int) override {throw std::runtime_error("RTX playback requires GPU texture bindings");}
    GpuCapabilities gpu_capabilities() const override {return state_->capabilities;}
    std::unique_ptr<GpuJob> submit_gpu(const GpuSubmitRequest&) override {throw std::runtime_error("RTX requires shared textures");}
    std::unique_ptr<GpuJob> submit_gpu_texture(const GpuTextureSubmitRequest& r) override {
        check_cuda(cudaSetDevice(state_->device_index),"select RTX inference adapter");
        if(r.input_size!=state_->pipeline->width())throw std::invalid_argument("RTX size changed without model reload");
        state_->pipeline->wait_ready();
        inferbridge::rtx::TextureFrame f{};
        f.input=reinterpret_cast<HANDLE>(r.shared_texture_handle);f.input_identity=r.shared_texture_identity;
        f.input_fence=reinterpret_cast<HANDLE>(r.wait_fence_handle);f.input_fence_value=r.wait_fence_value;
        f.width=r.width;f.height=r.height;f.bgra=r.pixel_format==DAD_GPU_PIXEL_BGRA8;
        f.output=reinterpret_cast<HANDLE>(r.output_texture_handle);f.output_identity=r.output_texture_identity;
        f.output_width=r.output_width;f.output_height=r.output_height;
        f.output_fence=reinterpret_cast<HANDLE>(r.signal_fence_handle);f.output_fence_value=r.signal_fence_value;
        // Allocate the job before submission so allocation failure cannot strand GPU work.
        GpuOutput output{};output.kind=GpuOutputKind::texture;output.width=r.output_width;output.height=r.output_height;
        output.row_stride_bytes=r.output_width*4;output.byte_size=static_cast<uint64_t>(r.output_width)*r.output_height*4;
        output.shared_resource_handle=r.output_texture_handle;output.ready_fence_handle=r.signal_fence_handle;
        output.ready_fence_value=r.signal_fence_value;output.source_frame_id=r.source_frame_id;output.timestamp_ns=r.timestamp_ns;
        auto job=std::make_unique<RtxJob>(state_,output);state_->pipeline->enqueue(f);state_->pipeline->record_completion(job->completion());return job;
    }
    std::unique_ptr<GpuJob> submit_host_texture(const GpuTextureSubmitRequest& request,const uint8_t* pixels,ptrdiff_t stride) override {
        check_cuda(cudaSetDevice(state_->device_index),"select RTX host upload adapter");
        state_->pipeline->wait_ready();
        const auto format=request.pixel_format==DAD_GPU_PIXEL_BGRA8?DXGI_FORMAT_B8G8R8A8_UNORM:DXGI_FORMAT_R8G8B8A8_UNORM;
        if(!state_->host_fence){
            if(FAILED(state_->device->CreateFence(0,D3D12_FENCE_FLAG_SHARED,IID_PPV_ARGS(&state_->host_fence)))||
               FAILED(state_->device->CreateSharedHandle(state_->host_fence.Get(),nullptr,GENERIC_ALL,nullptr,&state_->host_fence_handle)))
                throw std::runtime_error("cannot create RTX host upload fence");
        }
        if(state_->host_width!=request.width||state_->host_height!=request.height||state_->host_format!=format){
            D3D12_RESOURCE_DESC d{};d.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;d.Width=request.width;d.Height=request.height;
            d.DepthOrArraySize=1;d.MipLevels=1;d.Format=format;d.SampleDesc.Count=1;d.Flags=D3D12_RESOURCE_FLAG_NONE;
            D3D12_HEAP_PROPERTIES h{};h.Type=D3D12_HEAP_TYPE_DEFAULT;
            Microsoft::WRL::ComPtr<ID3D12Resource> texture;HANDLE handle=nullptr;
            if(FAILED(state_->device->CreateCommittedResource(&h,D3D12_HEAP_FLAG_SHARED,&d,D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&texture)))||
               FAILED(state_->device->CreateSharedHandle(texture.Get(),nullptr,GENERIC_ALL,nullptr,&handle)))
                throw std::runtime_error("cannot create RTX host upload texture");
            state_->host_import.reset();
            if(state_->host_texture_handle)CloseHandle(state_->host_texture_handle);
            state_->host_texture=std::move(texture);state_->host_texture_handle=handle;
            state_->host_width=request.width;state_->host_height=request.height;state_->host_format=format;++state_->host_identity;
        }
        if(!state_->host_import)state_->host_import=std::make_unique<inferbridge::rtx::SharedTexture>(state_->device.Get(),state_->host_texture_handle,format,request.width,request.height);
        // Input is already on the CPU for image/host-video sources. Upload once
        // on the inference worker; resize, inference and output remain on GPU.
        check_cuda(cudaMemcpy2DToArray(state_->host_import->array(),0,0,pixels,stride,request.width*4,request.height,cudaMemcpyHostToDevice),"upload host image");
        state_->upload_bytes+=static_cast<uint64_t>(request.width)*request.height*4;
        if(FAILED(state_->host_fence->Signal(++state_->host_fence_value)))throw std::runtime_error("cannot signal RTX host input readiness");
        auto gpu=request;gpu.shared_texture_handle=reinterpret_cast<uintptr_t>(state_->host_texture_handle);
        gpu.shared_texture_identity=state_->host_identity;gpu.wait_fence_handle=reinterpret_cast<uintptr_t>(state_->host_fence_handle);
        gpu.wait_fence_value=state_->host_fence_value;return submit_gpu_texture(gpu);
    }
    void transfer_counters(uint64_t& upload,uint64_t& download) const override {upload=state_->upload_bytes.load();download=0;}
};
std::unique_ptr<Executor> create_rtx_executor(const std::string& path,int index){return std::make_unique<RtxExecutor>(path,index);}
}
