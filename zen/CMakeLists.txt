cmake_minimum_required(VERSION 3.18)
project(zen LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

file(GLOB PROJECT_SOURCE *.cpp include/zen/*.h)

add_library(zen SHARED libzen.cpp)
target_include_directories(zen PUBLIC include)

add_library(zenpy SHARED libzenpy.cpp)
target_include_directories(zenpy PRIVATE include)
target_link_libraries(zenpy PRIVATE zen)

include(../cmake/Pybind11.cmake)
do_python_bindings()
