cmake_minimum_required(VERSION 3.10)
project(sycltest LANGUAGES C CXX)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

add_executable(sycltest main.cpp)
target_include_directories(sycltest PRIVATE .)

#set(CMAKE_MODULE_PATH /usr/lib/cmake/hipSYCL)
find_package(ComputeCpp REQUIRED)
target_link_libraries(sycltest PRIVATE ComputeCpp::sycl)
