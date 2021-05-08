cmake_minimum_required(VERSION 3.19)
project(zeno LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_definitions(-D__TBB_CPP20_COMPARISONS_PRESENT=0)

add_subdirectory(external)
add_subdirectory(zen)
add_subdirectory(zenvis)
add_subdirectory(partio)
add_subdirectory(FastFLIP)
add_subdirectory(QuickOCT)
