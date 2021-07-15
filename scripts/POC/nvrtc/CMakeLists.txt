cmake_minimum_required(VERSION 3.17)
project(cudatest LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)

add_executable(cudatest)
target_sources(cudatest PRIVATE nvrtc.cu)
target_include_directories(cudatest PRIVATE inc)
target_include_directories(cudatest PRIVATE ../external)
target_link_libraries(cudatest PRIVATE cuda nvrtc)
