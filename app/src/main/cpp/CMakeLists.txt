cmake_minimum_required(VERSION 3.31.6)
project(h33native LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# llama.cpp is fetched by CI into <repo>/third_party/llama.cpp.
set(LLAMA_SRC "${CMAKE_CURRENT_LIST_DIR}/../../../../third_party/llama.cpp")
if(NOT EXISTS "${LLAMA_SRC}/CMakeLists.txt")
    message(FATAL_ERROR "Pinned llama.cpp source is missing: ${LLAMA_SRC}")
endif()

add_subdirectory("${LLAMA_SRC}" "${CMAKE_BINARY_DIR}/llama" EXCLUDE_FROM_ALL)

add_library(h33native SHARED native_bridge.cpp)

target_include_directories(h33native PRIVATE
    "${LLAMA_SRC}/include"
    "${LLAMA_SRC}/ggml/include"
)

target_link_libraries(h33native PRIVATE
    llama
    log
    android
)
