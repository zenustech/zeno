cmake_minimum_required(VERSION 3.10)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} cmake)
project(sycltest LANGUAGES C CXX)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

add_executable(sycltest main.cpp)
target_include_directories(sycltest PRIVATE .)

find_package(ComputeCpp REQUIRED)
target_link_libraries(sycltest PRIVATE ComputeCpp::ComputeCpp)
