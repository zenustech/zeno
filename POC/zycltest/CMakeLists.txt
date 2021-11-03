cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_CXX_COMPILER dpcpp)

project(main LANGUAGES CXX)

set(PROJECT_SOURCES main.cpp)
add_executable(main ${PROJECT_SOURCES})
target_include_directories(main PRIVATE .)
