cmake_minimum_required(VERSION 3.22.1)
project(h33_llama LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
# The installable artifact intentionally remains the same Debug variant as the
# existing app, but the inference runtime itself must never use -O0.
set(CMAKE_C_FLAGS_DEBUG "-O3 -DNDEBUG" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_DEBUG "-O3 -DNDEBUG" CACHE STRING "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP ON CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(GGML_CPU_KLEIDIAI ON CACHE BOOL "" FORCE)

add_subdirectory(llama.cpp llama-build)
add_library(h33_llama SHARED h33_llama.cpp)
target_include_directories(h33_llama PRIVATE llama.cpp/include llama.cpp/ggml/include)
target_link_libraries(h33_llama PRIVATE llama android log)
