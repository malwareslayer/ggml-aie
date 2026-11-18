cmake_minimum_required(VERSION 3.24)
project(ggml-aie VERSION 0.3.3 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(GGML_SOURCE_DIR "" CACHE PATH "GGML source matching ggml 0.29.0")
set(GGML_LIBRARY_DIR "/usr/lib" CACHE PATH "System GGML runtime libraries")
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/CMakeLists.txt")
    message(FATAL_ERROR
        "The full-graph AIE2 backend is not implemented. "
        "The GEMM/CPU hybrid was retired to work/hybrid-reference; "
        "it must not be built or packaged as a full-graph backend.")
endif()
add_subdirectory(src)
