#if defined(__linux__) && !defined(__ANDROID__)
#include "linux_capture.h"
#include <inferbridge/linux_capture_harness.h>
ibr_linux_capture_capabilities
dad_linux_capture_capabilities(dad_context *context) {
  return context->executor->linux_capture_capabilities();
}
void dad_infer_linux_capture(
    dad_context *context,
    const inferbridge::linux_capture::LinuxDmaBufImage &source, uint32_t size,
    float *output) {
  const auto upstream = dad::network_shape(source.width, source.height, size);
  const dad::ImageShape shape{upstream.height, upstream.width};
  std::vector<float> depth(uint64_t(shape.width) * shape.height);
  context->executor->infer_linux_capture(source, shape.width, shape.height,
                                         depth.data());
  const auto bounds = std::minmax_element(depth.begin(), depth.end());
  const float minimum = *bounds.first, span = *bounds.second - minimum;
  for (auto &value : depth)
    value = span > 0 ? (value - minimum) / span : 0.f;
  inferbridge::linux_capture::resize_nearest(depth.data(), shape.width,
                                             shape.height, output, shape.width,
                                             shape.height);
}
#endif
