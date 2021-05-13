cmake_minimum_required(VERSION 3.19)
project(zeno LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# core
add_subdirectory(zen)
add_subdirectory(zenvdb)
add_subdirectory(zenvis)

# apps
add_subdirectory(FastFLIP)
add_subdirectory(QuickOCT)

# deps
add_subdirectory(glad)
add_subdirectory(partio)
