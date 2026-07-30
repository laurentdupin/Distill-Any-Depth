#include "distill_any_depth.h"

#include <assert.h>
#include <string.h>

int main(void) {
    assert(dad_abi_version() == DAD_ABI_VERSION);
    assert(strlen(dad_version_string()) != 0);
    assert(strcmp(dad_status_string(DAD_STATUS_OK), "ok") == 0);
    assert(
        strcmp(
            dad_status_string(DAD_STATUS_CANCELLED),
            "cancelled") == 0);

    dad_image_shape shape = {0, 0};
    assert(dad_get_network_shape(640, 480, 518, &shape) == DAD_STATUS_OK);
    assert(shape.width == 686);
    assert(shape.height == 518);
    assert(dad_get_network_shape(480, 640, 518, &shape) == DAD_STATUS_OK);
    assert(shape.width == 518);
    assert(shape.height == 686);

    assert(
        dad_get_network_shape(0, 480, 518, &shape) ==
        DAD_STATUS_INVALID_ARGUMENT);
    assert(strlen(dad_last_error()) != 0);
    dad_gpu_capabilities capabilities = {0};
    capabilities.struct_size = sizeof(capabilities);
    assert(
        dad_probe_gpu_capabilities(0, &capabilities) ==
        DAD_STATUS_OK);
    assert(capabilities.abi_version == DAD_ABI_VERSION);
    capabilities.struct_size = sizeof(capabilities);
    assert(
        dad_get_gpu_capabilities(NULL, &capabilities) ==
        DAD_STATUS_INVALID_ARGUMENT);
    dad_d3d12_texture_submit_request texture_request = {0};
    texture_request.struct_size = sizeof(texture_request);
    texture_request.abi_version = DAD_ABI_VERSION;
    dad_gpu_job* job = NULL;
    assert(
        dad_submit_d3d12_texture(
            NULL, &texture_request, &job) ==
        DAD_STATUS_INVALID_ARGUMENT);
    return 0;
}
