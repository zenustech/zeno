cmake_minimum_required(VERSION 3.10)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} cmake)
project(sycltest LANGUAGES C CXX)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

find_package(ComputeCpp REQUIRED)

add_executable(sycltest main.cpp)
add_sycl_to_target(TARGET sycltest SOURCES main.cpp)
target_include_directories(sycltest PRIVATE .)

target_link_libraries(sycltest PRIVATE ComputeCpp::ComputeCpp)
