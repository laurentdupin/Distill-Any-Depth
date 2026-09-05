#pragma once
#if defined(__linux__) && !defined(__ANDROID__)
#include <inferbridge/linux_capture_vulkan.h>
struct dad_context;
ibr_linux_capture_capabilities dad_linux_capture_capabilities(dad_context *);
void dad_infer_linux_capture(
    dad_context *, const inferbridge::linux_capture::LinuxDmaBufImage &,
    uint32_t, float *);
#endif
