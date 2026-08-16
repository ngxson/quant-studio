cmake_minimum_required(VERSION 3.14)
project(quant-studio-cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# build against the llama.cpp submodule at the repo root
set(LLAMA_BUILD_COMMON ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp ${CMAKE_BINARY_DIR}/llama.cpp)

add_executable(gmatrix gmatrix.cpp)
target_link_libraries(gmatrix PRIVATE llama-common llama ggml)
