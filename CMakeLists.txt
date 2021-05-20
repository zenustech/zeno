cmake_minimum_required(VERSION 3.16)
project(zeno LANGUAGES C CXX)


set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)#Debug)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /usr/lib/cmake/OpenVDB /usr/local/lib/cmake/OpenVDB)
find_package(OpenVDB QUIET)
option(ZS_ENABLE_OPENVDB "Enable OpenVDB" ON)
if (NOT ${OpenVDB_FOUND})
    set(ZS_ENABLE_OPENVDB OFF)
endif()
message("==> OpenVDB find status: ${OpenVDB_FOUND}")
message("==> OpenVDB enable status: ${ZS_ENABLE_OPENVDB}")


add_subdirectory(zen)
add_subdirectory(zenvis)
add_subdirectory(zenbase)
add_subdirectory(external/glad)
add_subdirectory(QuickOCT)
if(ZS_ENABLE_OPENVDB)
    add_subdirectory(zenvdb)
    add_subdirectory(FastFLIP)
    add_subdirectory(external/partio)
endif()
