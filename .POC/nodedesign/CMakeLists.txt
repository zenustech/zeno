cmake_minimum_required(VERSION 3.10)
# GCC 11, Clang 12, MSVC 19.28
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Debug)
project(ZenoEditor LANGUAGES C CXX)

file(GLOB PROJECT_SOURCE *.cpp *.h)
add_executable(ZenoEditor ${PROJECT_SOURCE})

target_include_directories(ZenoEditor PRIVATE .)
