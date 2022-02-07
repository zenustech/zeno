cmake_minimum_required(VERSION 3.17)

set(FDB_WITH_CUDA ON)
set(FDB_WITH_OPENMP OFF)

set(CMAKE_CXX_STANDARD 17)

if (FDB_WITH_CUDA)
    project(OpenFDB LANGUAGES CXX CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 75)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr")
else()
    project(OpenFDB LANGUAGES CXX)
endif()

add_executable(OpenFDB)
if (FDB_WITH_CUDA)
    target_sources(OpenFDB PRIVATE main.cu)
else()
    target_sources(OpenFDB PRIVATE main.cpp)
endif()
target_include_directories(OpenFDB PRIVATE inc .)

if (FDB_WITH_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(OpenFDB OpenMP::OpenMP_CXX)
endif()
