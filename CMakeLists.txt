cmake_minimum_required(VERSION 3.19)
project(zeno LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(external)
add_subdirectory(zen)
add_subdirectory(zenvis)
add_subdirectory(zeneditor)
#add_subdirectory(FastFLIP)
