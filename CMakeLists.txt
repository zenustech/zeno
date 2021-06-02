cmake_minimum_required(VERSION 3.12)
project(zeno LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_ROOT_DIR}/cmake)

add_subdirectory(external/pybind11)
add_subdirectory(external/glad)
add_subdirectory(zen)
add_subdirectory(zenvis)