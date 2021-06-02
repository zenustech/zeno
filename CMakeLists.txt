cmake_minimum_required(VERSION 3.12)
project(zeno LANGUAGES C CXX)


set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


option(ZS_ENABLE_OPENVDB "Enable OpenVDB" ON)
option(ZS_ENABLE_ZENBASE "Enable ZenBase" ON)


set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /usr/lib/cmake/OpenVDB /usr/local/lib/cmake/OpenVDB)
find_package(OpenVDB QUIET)
if (NOT ${OpenVDB_FOUND})
    set(ZS_ENABLE_OPENVDB OFF)
endif()
message("==> OpenVDB find status: ${OpenVDB_FOUND}")
message("==> OpenVDB enable status: ${ZS_ENABLE_OPENVDB}")

if (MSVC)
	message("MSVC found, disabling ZenBase...")
	set(ZS_ENABLE_ZENBASE OFF)
endif()


add_subdirectory(zen)
add_subdirectory(zenvis)
add_subdirectory(external/glad)
if(ZS_ENABLE_ZENBASE)
	add_subdirectory(zenbase)
	add_subdirectory(QuickOCT)
endif()
if(ZS_ENABLE_OPENVDB)
    add_subdirectory(zenvdb)
    add_subdirectory(FastFLIP)
    add_subdirectory(external/partio)
endif()
