cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_CXX_COMPILER dpcpp)

project(main LANGUAGES CXX)

set(PROJECT_SOURCES usm.cpp)
add_executable(main ${PROJECT_SOURCES})
target_include_directories(main PRIVATE .)
