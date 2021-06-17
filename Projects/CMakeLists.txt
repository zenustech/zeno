cmake_minimum_required(VERSION 3.12)
project(zeno_projects LANGUAGES C CXX)
#cmake_policy(SET CMP0079 NEW)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../zen/cmake ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(ENV{PYTHONPATH} ${CMAKE_CURRENT_SOURCE_DIR}/..)

include_directories(include)

add_subdirectory(zenbase)
add_subdirectory(ZMS)
find_package(OpenVDB QUIET)
if (${OpenVDB_FOUND})
    add_subdirectory(zenvdb)
    add_subdirectory(FastFLIP)
endif()
add_subdirectory(Rigid)
