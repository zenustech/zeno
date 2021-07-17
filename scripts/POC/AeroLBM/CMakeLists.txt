cmake_minimum_required(VERSION 3.17)
project(LBMSolver LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)

add_executable(LBMSolver)
target_sources(LBMSolver PRIVATE main.cu)
target_include_directories(LBMSolver PRIVATE inc)
