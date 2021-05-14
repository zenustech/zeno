cmake_minimum_required(VERSION 3.19)
project(zeno LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)#Release)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# openvdb
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /usr/local/lib/cmake/OpenVDB/)
find_package(OpenVDB QUIET)
set(ZS_ENABLE_OPENVDB ON)
if (NOT ${OpenVDB_FOUND})
    set(ZS_ENABLE_OPENVDB OFF)
    message("OpenVDB turned off mandatorily")
endif()
# message("\n\n\nopenvdb find status: ${OpenVDB_FOUND}\n\n\n")
# message("\n\n\nopenvdb status: ${ZS_ENABLE_OPENVDB}\n\n\n")

add_subdirectory(zen)
add_subdirectory(glad)
add_subdirectory(zenvis)

add_subdirectory(partio)
if(ZS_ENABLE_OPENVDB)
    add_subdirectory(FastFLIP)
endif()
add_subdirectory(QuickOCT)
