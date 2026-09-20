#include "distill_any_depth.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>

static dad_status DAD_CALL mock_submit(dad_context*,const dad_d3d12_texture_binding_request*,dad_gpu_job**);
static dad_status DAD_CALL mock_poll(const dad_gpu_job*,dad_gpu_job_status*);
static dad_status DAD_CALL mock_cancel(dad_gpu_job*);
static void DAD_CALL mock_release(dad_gpu_job*);
// Exercise the real worker and ABI state transitions with deterministic GPU
// submission/completion barriers; no GPU, model, or shared texture is needed.
#define dad_submit_d3d12_texture_binding mock_submit
#define dad_gpu_job_poll mock_poll
#define dad_gpu_job_cancel mock_cancel
#define dad_gpu_job_release mock_release
#include "../src/inferbridge_harness.cpp"
#undef dad_submit_d3d12_texture_binding
#undef dad_gpu_job_poll
#undef dad_gpu_job_cancel
#undef dad_gpu_job_release

struct dad_gpu_job { std::atomic<bool> cancelled{false}; };
namespace {
std::mutex gate_mutex;
std::condition_variable gate_condition;
bool entered=false,allow_submit=false;
std::atomic<bool> completed{false};
std::atomic<int> submissions{0},releases{0};
}
static dad_status DAD_CALL mock_submit(dad_context*,const dad_d3d12_texture_binding_request*,dad_gpu_job** output) {
    ++submissions;
    std::unique_lock<std::mutex> lock(gate_mutex);
    entered=true;gate_condition.notify_all();
    gate_condition.wait(lock,[]{return allow_submit;});
    *output=new dad_gpu_job;return DAD_STATUS_OK;
}
static dad_status DAD_CALL mock_poll(const dad_gpu_job* job,dad_gpu_job_status* status) {
    status->state=!completed?DAD_GPU_JOB_RUNNING:job->cancelled?DAD_GPU_JOB_CANCELLED:DAD_GPU_JOB_COMPLETE;
    return DAD_STATUS_OK;
}
static dad_status DAD_CALL mock_cancel(dad_gpu_job* job) {job->cancelled=true;return DAD_STATUS_OK;}
static void DAD_CALL mock_release(dad_gpu_job* job) {++releases;delete job;}
int main() {
    auto worker=std::make_shared<DadGpuWorker>(nullptr);
    auto counts=std::make_shared<std::atomic<uint32_t>>(0u);
    auto make_job=[&](){auto* j=new ibrh_job;counts->fetch_add(1u);j->gpu_admission=std::make_shared<DadGpuAdmission>(counts);j->gpu_worker=worker;j->gpu_state=IBRH_JOB_QUEUED;return j;};
    auto* first=make_job();worker->enqueue(first);
    bool okay=true;
    {
        std::unique_lock<std::mutex> lock(gate_mutex);
        okay=gate_condition.wait_for(lock,std::chrono::seconds(5),[]{return entered;});
    }
    ibrh_job_status status{};
    if(!okay){std::cerr<<"Worker did not enter submission\n";}
    else {
        job_cancel(first);job_poll(first,sizeof(status),&status);
        if(status.state==IBRH_JOB_CANCELLED||status.state==IBRH_JOB_COMPLETE||status.state==IBRH_JOB_FAILED){
            std::cerr<<"Cancellation released borrowed input while GPU submission was still active\n";okay=false;
        }
        auto* queued=make_job();worker->enqueue(queued);job_cancel(queued);job_poll(queued,sizeof(status),&status);
        if(status.state!=IBRH_JOB_CANCELLED||submissions!=1){std::cerr<<"Queued cancellation was not immediate\n";okay=false;}
        job_release(queued);
    }
    {std::lock_guard<std::mutex> lock(gate_mutex);allow_submit=true;}gate_condition.notify_all();
    auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
    for(;;){
        bool published=false;{std::lock_guard<std::mutex> lock(first->gpu_mutex);published=first->gpu_job!=nullptr;}
        if(published)break;
        if(std::chrono::steady_clock::now()>deadline){okay=false;std::cerr<<"Native job was not published\n";break;}
        std::this_thread::yield();
    }
    job_poll(first,sizeof(status),&status);
    if(status.state!=IBRH_JOB_RUNNING){std::cerr<<"Cancelled native job became terminal before GPU completion\n";okay=false;}
    completed=true;job_poll(first,sizeof(status),&status);
    if(status.state!=IBRH_JOB_CANCELLED){std::cerr<<"Completed cancellation was not reported\n";okay=false;}
    job_release(first);worker->stop();
    if(releases!=1||*counts!=0){std::cerr<<"Job lifetime did not retire exactly once\n";okay=false;}
    if(okay)std::cout<<"PASS: dequeued cancellation retains input until GPU completion; queued cancellation remains immediate\n";
    return okay?0:1;
}
