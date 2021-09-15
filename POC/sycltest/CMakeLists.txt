cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
project(sycltest LANGUAGES CXX)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

add_definitions(-DCL_TARGET_OPENCL_VERSION=300)

if (COMPUTECPP_BITCODE)
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
    find_package(ComputeCpp REQUIRED)
endif()

set(PROJECT_SOURCES main.cpp)
add_executable(sycltest ${PROJECT_SOURCES})
if (COMPUTECPP_BITCODE)
    add_sycl_to_target(TARGET sycltest SOURCES ${PROJECT_SOURCES})
endif()
target_include_directories(sycltest PRIVATE .)

if (COMPUTECPP_BITCODE)
    target_link_libraries(sycltest PRIVATE ComputeCpp::ComputeCpp)
endif()
