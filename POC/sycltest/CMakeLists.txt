cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
project(sycltest LANGUAGES CXX)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
find_package(ComputeCpp REQUIRED)

add_executable(sycltest main.cpp)
add_sycl_to_target(TARGET sycltest SOURCES main.cpp)
target_include_directories(sycltest PRIVATE .)

target_link_libraries(sycltest PRIVATE ComputeCpp::ComputeCpp)
