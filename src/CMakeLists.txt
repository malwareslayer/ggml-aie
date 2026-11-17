find_package(PkgConfig REQUIRED)
pkg_check_modules(XRT REQUIRED IMPORTED_TARGET xrt=2.21.75)
if(NOT EXISTS "${GGML_SOURCE_DIR}/src/ggml-backend-impl.h" OR
   NOT EXISTS "${GGML_SOURCE_DIR}/include/ggml.h")
    message(FATAL_ERROR "GGML_SOURCE_DIR must point at ggml 0.29.0 source")
endif()
foreach(library ggml ggml-base)
    if(NOT EXISTS "${GGML_LIBRARY_DIR}/lib${library}.so")
        message(FATAL_ERROR "Missing GGML library: ${GGML_LIBRARY_DIR}/lib${library}.so")
    endif()
endforeach()

add_library(ggml-aie SHARED ggml_backend.cpp xrt_context.cpp xrt_matmul.cpp xrt_ops.cpp xrt_softmax.cpp)
set_target_properties(ggml-aie PROPERTIES
    OUTPUT_NAME ggml-aie
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_compile_features(ggml-aie PRIVATE cxx_std_20)
target_compile_definitions(ggml-aie PRIVATE GGML_BACKEND_DL GGML_BACKEND_SHARED GGML_BACKEND_BUILD)
target_compile_options(ggml-aie PRIVATE -Wall -Wextra -Werror -fvisibility=hidden
    -fvisibility-inlines-hidden
    "-ffile-prefix-map=${GGML_SOURCE_DIR}=."
    "-ffile-prefix-map=${PROJECT_SOURCE_DIR}=." )
target_link_options(ggml-aie PRIVATE
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/ggml-aie.ver")
target_include_directories(ggml-aie PRIVATE
    "${GGML_SOURCE_DIR}/include" "${GGML_SOURCE_DIR}/src")
target_link_libraries(ggml-aie PRIVATE PkgConfig::XRT
    "${GGML_LIBRARY_DIR}/libggml.so"
    "${GGML_LIBRARY_DIR}/libggml-base.so")
install(TARGETS ggml-aie LIBRARY DESTINATION lib/ggml)
