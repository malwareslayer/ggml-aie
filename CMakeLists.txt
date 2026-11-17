cmake_minimum_required(VERSION 3.24)
project(ollama-aie2 VERSION 0.2.2 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(GGML_SOURCE_DIR "" CACHE PATH "GGML source matching Ollama 0.40.16 / llama.cpp b12177")
set(OLLAMA_LIBRARY_DIR "/usr/lib/ollama" CACHE PATH "Pinned Ollama GGML runtime libraries")
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/CMakeLists.txt")
    message(FATAL_ERROR
        "The full-graph AIE2 backend is not implemented. "
        "The GEMM/CPU hybrid was retired to work/hybrid-reference; "
        "it must not be built or packaged as a full-graph backend.")
endif()
add_subdirectory(src)
