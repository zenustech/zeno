cmake_minimum_required(VERSION 3.12)
#project(zeno_projects LANGUAGES C CXX)
#cmake_policy(SET CMP0079 NEW)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../zen/cmake ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(ENV{PYTHONPATH} ${CMAKE_CURRENT_SOURCE_DIR}/..)

include_directories(include)

add_subdirectory(zenbase)
#add_subdirectory(Rigid)
#add_subdirectory(ZMS)
#if (NOT WIN32)
    #add_subdirectory(zenvdb)
    #add_subdirectory(FastFLIP)
#endif()
