cmake_minimum_required(VERSION 3.17)
project(cudatest LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
set(CMAKE_BUILD_TYPE Release)

add_executable(cudatest)
target_sources(cudatest PRIVATE main.cu)
target_include_directories(cudatest PRIVATE inc)
