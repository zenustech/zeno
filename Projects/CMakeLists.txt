cmake_minimum_required(VERSION 3.12)
#project(zeno_projects LANGUAGES C CXX)
#cmake_policy(SET CMP0079 NEW)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../zen/cmake ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(ENV{PYTHONPATH} ${CMAKE_CURRENT_SOURCE_DIR}/..)

include_directories(include)

set(EXTENSION_LIST Rigid ZMS zenbase zenvdb FastFLIP)

foreach (name IN ITEMS ${EXTENSION_LIST})
    option(EXTENSION_${name} "Build ZENO with extension: ${name}" ON)
endforeach()

if (WIN32)
    set(EXTENSION_zenvdb OFF)
    set(EXTENSION_FastFLIP OFF)
endif()

foreach (name IN ITEMS ${EXTENSION_LIST})
    message("Building ZENO with extension: ${name}")
    add_subdirectory(${name})
endforeach()
