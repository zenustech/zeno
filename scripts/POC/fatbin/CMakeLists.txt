cmake_minimum_required(VERSION 3.17)
project(cudatest LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --keep")

add_executable(cudatest)
target_sources(cudatest PRIVATE main.cu)
target_include_directories(cudatest PRIVATE inc)
target_include_directories(cudatest PRIVATE ../external)
target_link_libraries(cudatest PRIVATE cuda nvrtc)
