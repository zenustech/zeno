cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
project(sycltest LANGUAGES CXX)

add_definitions(-DCL_TARGET_OPENCL_VERSION=300)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
find_package(ComputeCpp REQUIRED)

set(PROJECT_SOURCES main.cpp)
add_executable(sycltest ${PROJECT_SOURCES})
add_sycl_to_target(TARGET sycltest SOURCES ${PROJECT_SOURCES})
target_include_directories(sycltest PRIVATE .)

target_link_libraries(sycltest PRIVATE ComputeCpp::ComputeCpp)
