#include "inferbridge_harness.h"

#include <string.h>

#define CHECK(expression) \
    do { if (!(expression)) return __LINE__; } while (0)

int main(void) {
    ibrh_api api = {0};
    CHECK(
        ibrh_get_api(
            IBRH_CURRENT_API_VERSION, sizeof(api), &api) == IBRH_OK);
    CHECK(api.struct_size == sizeof(api));
    CHECK(api.runtime_create != NULL);
    CHECK(api.model_load != NULL);
    CHECK(api.submit != NULL);
    CHECK(
        ibrh_get_api(
            IBRH_MAKE_API_VERSION(2, 0), sizeof(api), &api) ==
        IBRH_ERROR_UNSUPPORTED_API);

    ibrh_capabilities capabilities = {0};
    CHECK(
        api.query_capabilities(sizeof(capabilities), &capabilities) ==
        IBRH_OK);
    CHECK(capabilities.flags ==
        (IBRH_CAP_HOST_MEMORY | IBRH_CAP_ASYNC_SUBMIT |
         IBRH_CAP_CANCELLATION | IBRH_CAP_GPU_RESOURCES |
         IBRH_CAP_EXTERNAL_SYNCHRONIZATION |
         IBRH_CAP_GPU_RESIDENT_OUTPUT));
    CHECK(
        capabilities.input_domain_mask ==
        ((1ull << IBRH_RESOURCE_DOMAIN_HOST) |
         (1ull << IBRH_RESOURCE_DOMAIN_D3D12)));
    CHECK(
        capabilities.output_domain_mask ==
        ((1ull << IBRH_RESOURCE_DOMAIN_HOST) |
         (1ull << IBRH_RESOURCE_DOMAIN_D3D12)));
    CHECK(capabilities.synchronization_mask ==
        (1ull << IBRH_SYNC_D3D12_FENCE));
    CHECK(capabilities.maximum_inputs == 1u);
    CHECK(capabilities.maximum_outputs == 1u);
    CHECK(capabilities.maximum_in_flight_jobs == 3u);
    CHECK(
        capabilities.harness_id.size ==
        strlen("inferbridge.distill-any-depth.native"));

    ibrh_runtime_create_request request = {0};
    request.struct_size = sizeof(request);
    request.api_version = IBRH_CURRENT_API_VERSION;
    request.backend.data = "native";
    request.backend.size = 6u;
    ibrh_runtime* runtime = NULL;
    CHECK(
        api.runtime_create(
            sizeof(request), &request, &runtime) == IBRH_OK);
    CHECK(runtime != NULL);
    api.runtime_destroy(runtime);
    return 0;
}
