cmake_minimum_required(VERSION 3.17)
project(cudatest LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

file(GLOB PROJECT_SOURCE *.cpp *.h *.cu *.cuh)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

add_executable(cudatest)
target_sources(cudatest PRIVATE ${PROJECT_SOURCE})
target_include_directories(cudatest PRIVATE .)
target_include_directories(cudatest PRIVATE ../external)
