cmake_minimum_required(VERSION 3.17)

set(WITH_CUDA ON)
set(WITH_OPENMP OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

if (WITH_CUDA)
    project(cudatest LANGUAGES CXX CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 75)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr")
else()
    project(cudatest LANGUAGES CXX)
endif()

add_executable(cudatest)
target_sources(cudatest PRIVATE main.cu)
target_include_directories(cudatest PRIVATE inc .)

if (WITH_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(cudatest OpenMP::OpenMP_CXX)
endif()
