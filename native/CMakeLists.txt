cmake_minimum_required(VERSION 3.22)

project(distill_any_depth_c VERSION 0.1.0 LANGUAGES C CXX)

if(MSVC)
    # The distributable must be one DLL, so do not require the separately
    # installed Visual C++ runtime DLLs.
    set(
        CMAKE_MSVC_RUNTIME_LIBRARY
        "MultiThreaded$<$<CONFIG:Debug>:Debug>"
    )
endif()

option(DAD_BUILD_TESTS "Build the C ABI and preprocessing tests" ON)
option(DAD_WITH_VULKAN "Build the dependency-free Vulkan executor" ON)
option(DAD_BUILD_TOOLS "Build native command-line tools" ON)

if(DAD_WITH_VULKAN)
    find_package(Vulkan REQUIRED COMPONENTS glslc)

    set(DAD_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
    file(MAKE_DIRECTORY "${DAD_GENERATED_DIR}")

    function(dad_compile_shader NAME)
        set(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/${NAME}.glsl")
        set(SPIRV "${DAD_GENERATED_DIR}/${NAME}.spv")
        set(HEADER "${DAD_GENERATED_DIR}/${NAME}_spv.h")
        add_custom_command(
            OUTPUT "${SPIRV}"
            COMMAND Vulkan::glslc -fshader-stage=compute
                --target-env=vulkan1.3 --target-spv=spv1.6 -Werror
                "-I${CMAKE_CURRENT_SOURCE_DIR}/src/shaders"
                -o "${SPIRV}" "${SOURCE}"
            DEPENDS
                "${SOURCE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/indexing.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/conv2d_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/linear_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/project_tokens_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/conv_transpose_nonoverlap_common.glsl"
            VERBATIM
        )
        add_custom_command(
            OUTPUT "${HEADER}"
            COMMAND "${CMAKE_COMMAND}"
                "-DINPUT=${SPIRV}"
                "-DOUTPUT=${HEADER}"
                "-DVARIABLE=dad_${NAME}_spv"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake"
            DEPENDS "${SPIRV}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake"
            VERBATIM
        )
        set(DAD_SHADER_HEADERS ${DAD_SHADER_HEADERS} "${HEADER}" PARENT_SCOPE)
    endfunction()

    dad_compile_shader(elementwise)
    dad_compile_shader(linear)
    dad_compile_shader(linear16)
    dad_compile_shader(linear_half)
    dad_compile_shader(linear16_half)
    dad_compile_shader(linear_vec8)
    dad_compile_shader(gelu)
    dad_compile_shader(layer_norm)
    dad_compile_shader(add_scaled)
    dad_compile_shader(prepare_tokens)
    dad_compile_shader(position_bicubic)
    dad_compile_shader(add_position)
    dad_compile_shader(add)
    dad_compile_shader(bmm)
    dad_compile_shader(bmm_score_half)
    dad_compile_shader(bmm_value_half)
    dad_compile_shader(softmax_lastdim)
    dad_compile_shader(softmax_lastdim_half)
    dad_compile_shader(project_tokens)
    dad_compile_shader(project_tokens_half)
    dad_compile_shader(conv2d)
    dad_compile_shader(conv2d8)
    dad_compile_shader(conv2d_half)
    dad_compile_shader(conv2d8_half)
    dad_compile_shader(conv2d_tiled)
    dad_compile_shader(conv_transpose_nonoverlap)
    dad_compile_shader(conv_transpose_nonoverlap_half)
    dad_compile_shader(bilinear_align_false)
    dad_compile_shader(bilinear_align_false_image)
    dad_compile_shader(bilinear_align_true)
    dad_compile_shader(bilinear_align_true_image)
    dad_compile_shader(relu)
    dad_compile_shader(preprocess_rgba)
    dad_compile_shader(preprocess_texture)
endif()

add_library(distill_any_depth SHARED
    src/dad.cpp
    src/image.cpp
    src/inferbridge_harness.cpp
)
add_library(distill_any_depth::distill_any_depth ALIAS distill_any_depth)

target_compile_features(distill_any_depth PRIVATE cxx_std_17)
target_include_directories(distill_any_depth
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_compile_definitions(distill_any_depth
    PRIVATE
        DAD_BUILD_DLL
        INFERBRIDGE_HARNESS_BUILDING_LIBRARY
        $<$<BOOL:${DAD_WITH_VULKAN}>:DAD_WITH_VULKAN=1>
)
set_target_properties(distill_any_depth PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)

if(DAD_WITH_VULKAN)
    target_sources(distill_any_depth PRIVATE
        src/model.cpp
        src/gpu_model.cpp
        src/encoder.cpp
        src/dpt.cpp
        src/operators.cpp
        src/vulkan.cpp
        src/gpu_preprocess.cpp
        src/vulkan_executor.cpp
        ${DAD_SHADER_HEADERS}
    )
    target_include_directories(distill_any_depth PRIVATE "${DAD_GENERATED_DIR}")
    target_link_libraries(distill_any_depth PRIVATE Vulkan::Vulkan)
    if(WIN32)
        target_link_libraries(distill_any_depth PRIVATE d3d12 dxgi)
    endif()
endif()

include(GNUInstallDirs)
install(TARGETS distill_any_depth
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)
install(FILES include/distill_any_depth.h DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(FILES include/inferbridge_harness.h DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

if(DAD_BUILD_TOOLS)
    add_executable(dad_run tools/dad_run.c)
    target_link_libraries(dad_run PRIVATE distill_any_depth)
    install(TARGETS dad_run RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")

    if(DAD_WITH_VULKAN)
        add_executable(dad_encoder_probe
            tools/dad_encoder_probe.cpp
            src/encoder.cpp
            src/dpt.cpp
            src/gpu_model.cpp
            src/model.cpp
            src/operators.cpp
            src/vulkan.cpp
            src/gpu_preprocess.cpp
            ${DAD_SHADER_HEADERS}
        )
        target_compile_features(dad_encoder_probe PRIVATE cxx_std_17)
        target_include_directories(dad_encoder_probe PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${DAD_GENERATED_DIR}"
        )
        target_link_libraries(dad_encoder_probe PRIVATE Vulkan::Vulkan)
    endif()
endif()

if(DAD_BUILD_TESTS)
    enable_testing()
    add_executable(dad_c_abi_smoke tests/c_abi_smoke.c)
    target_link_libraries(dad_c_abi_smoke PRIVATE distill_any_depth)
    add_test(NAME dad_c_abi_smoke COMMAND dad_c_abi_smoke)

    add_executable(dad_harness_abi_smoke tests/harness_abi_smoke.c)
    target_link_libraries(dad_harness_abi_smoke PRIVATE distill_any_depth)
    add_test(NAME dad_harness_abi_smoke COMMAND dad_harness_abi_smoke)

    add_executable(dad_harness_full_graph tests/harness_full_graph.cpp)
    target_compile_features(dad_harness_full_graph PRIVATE cxx_std_17)
    target_link_libraries(dad_harness_full_graph PRIVATE distill_any_depth)
    add_test(NAME dad_harness_full_graph COMMAND dad_harness_full_graph)
    set_tests_properties(
        dad_harness_full_graph PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(dad_image_tests tests/image_tests.cpp src/image.cpp)
    target_include_directories(dad_image_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    add_test(NAME dad_image_tests COMMAND dad_image_tests)

    add_executable(dad_preprocess_probe tools/dad_preprocess_probe.cpp src/image.cpp)
    target_include_directories(
        dad_preprocess_probe PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

    if(DAD_WITH_VULKAN)
        add_executable(dad_vulkan_smoke
            tests/vulkan_smoke.cpp
            src/gpu_model.cpp
            src/model.cpp
            src/operators.cpp
            src/vulkan.cpp
            src/gpu_preprocess.cpp
            ${DAD_SHADER_HEADERS}
        )
        target_compile_features(dad_vulkan_smoke PRIVATE cxx_std_17)
        target_include_directories(dad_vulkan_smoke PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${DAD_GENERATED_DIR}"
        )
        target_link_libraries(dad_vulkan_smoke PRIVATE Vulkan::Vulkan)
        add_test(NAME dad_vulkan_smoke COMMAND dad_vulkan_smoke)

        if(WIN32)
            add_executable(
                dad_model_metadata_tests
                tests/model_metadata.cpp
                src/model.cpp)
            target_compile_features(
                dad_model_metadata_tests PRIVATE cxx_std_17)
            target_include_directories(
                dad_model_metadata_tests PRIVATE
                    "${CMAKE_CURRENT_SOURCE_DIR}/include"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src")
            add_test(
                NAME dad_model_metadata_tests
                COMMAND dad_model_metadata_tests)

            add_executable(dad_vulkan_external_smoke
                tests/vulkan_external_smoke.cpp
                src/gpu_preprocess.cpp
                src/image.cpp
                src/vulkan.cpp
                ${DAD_SHADER_HEADERS}
            )
            target_compile_features(
                dad_vulkan_external_smoke PRIVATE cxx_std_17)
            target_include_directories(
                dad_vulkan_external_smoke PRIVATE
                    "${CMAKE_CURRENT_SOURCE_DIR}/include"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src"
                    "${DAD_GENERATED_DIR}"
            )
            target_link_libraries(
                dad_vulkan_external_smoke PRIVATE
                    Vulkan::Vulkan d3d12 dxgi)
            add_test(
                NAME dad_vulkan_external_smoke
                COMMAND dad_vulkan_external_smoke)
            set_tests_properties(
                dad_vulkan_external_smoke
                PROPERTIES SKIP_RETURN_CODE 77)

            add_executable(
                dad_d3d12_full_graph
                tests/d3d12_full_graph.cpp)
            target_compile_features(
                dad_d3d12_full_graph PRIVATE cxx_std_17)
            target_link_libraries(
                dad_d3d12_full_graph PRIVATE
                    distill_any_depth d3d12 dxgi)
            target_compile_definitions(
                dad_d3d12_full_graph PRIVATE
                    "DAD_DEFAULT_VITS_MODEL=\"${CMAKE_CURRENT_SOURCE_DIR}/validation/distill_any_depth_vits.dad\"")
            add_test(
                NAME dad_d3d12_full_graph
                COMMAND dad_d3d12_full_graph)
            set_tests_properties(
                dad_d3d12_full_graph
                PROPERTIES SKIP_RETURN_CODE 77)
        endif()
    endif()
endif()
