find_package(PkgConfig REQUIRED)
pkg_check_modules(XRT REQUIRED IMPORTED_TARGET xrt=2.21.75)
if(NOT EXISTS "${GGML_SOURCE_DIR}/src/ggml-backend-impl.h" OR
   NOT EXISTS "${GGML_SOURCE_DIR}/include/ggml.h")
    message(FATAL_ERROR "GGML_SOURCE_DIR must point at llama.cpp b12177/ggml")
endif()
foreach(library ggml ggml-base ggml-cpu-x64)
    if(NOT EXISTS "${OLLAMA_LIBRARY_DIR}/lib${library}.so")
        message(FATAL_ERROR "Missing pinned Ollama library: lib${library}.so")
    endif()
endforeach()

add_library(ggml-aie2 SHARED ggml_backend.cpp xrt_context.cpp xrt_matmul.cpp xrt_ops.cpp xrt_softmax.cpp)
set_target_properties(ggml-aie2 PROPERTIES
    OUTPUT_NAME ggml-aie2
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "${OLLAMA_LIBRARY_DIR}")
target_compile_features(ggml-aie2 PRIVATE cxx_std_20)
target_compile_definitions(ggml-aie2 PRIVATE GGML_BACKEND_DL GGML_BACKEND_SHARED GGML_BACKEND_BUILD)
target_compile_options(ggml-aie2 PRIVATE -Wall -Wextra -Werror -fvisibility=hidden
    -fvisibility-inlines-hidden
    "-ffile-prefix-map=${GGML_SOURCE_DIR}=."
    "-ffile-prefix-map=${PROJECT_SOURCE_DIR}=." )
target_link_options(ggml-aie2 PRIVATE
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/ggml-aie2.ver")
target_include_directories(ggml-aie2 PRIVATE
    "${GGML_SOURCE_DIR}/include" "${GGML_SOURCE_DIR}/src")
target_link_libraries(ggml-aie2 PRIVATE PkgConfig::XRT
    "${OLLAMA_LIBRARY_DIR}/libggml.so"
    "${OLLAMA_LIBRARY_DIR}/libggml-base.so"
    "${OLLAMA_LIBRARY_DIR}/libggml-cpu-x64.so")
install(TARGETS ggml-aie2 LIBRARY DESTINATION lib/ollama)
