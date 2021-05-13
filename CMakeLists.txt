cmake_minimum_required(VERSION 3.19)
project(zeno LANGUAGES C CXX)


option(ZS_WITH_OPENVDB "Enable OpenVDB" ON)
option(ZS_RELEASE_MODE "Enable Release Build" ON)


if (ZS_RELEASE_MODE)
    set(CMAKE_BUILD_TYPE Release)
else()
    set(CMAKE_BUILD_TYPE Debug)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


# core
add_subdirectory(zen)
add_subdirectory(zenvis)
if (ZS_WITH_OPENVDB)
    add_subdirectory(zenvdb)
endif()

# apps
add_subdirectory(QuickOCT)
if (ZS_WITH_OPENVDB)
    add_subdirectory(FastFLIP)
endif()

# deps
add_subdirectory(glad)
add_subdirectory(partio)
