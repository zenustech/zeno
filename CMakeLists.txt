cmake_minimum_required(VERSION 3.18)

option(ZENO_WITH_SYCL "Enable SYCL support for Zeno" OFF)
option(ZENO_WITH_LEGACY "Build Zeno With Legacy Nodes" OFF)
option(ZENO_WITH_BACKWARD "Enable stack backtrace for Zeno" OFF)
option(ZENO_BUILD_EDITOR "Enable building Zeno Editor" ON)
option(ZENO_BUILD_DESIGNER "Enable building Zeno Designer" ON)
option(ZENO_BUILD_TESTS "Enable building Zeno Tests" OFF)
option(ZENO_BUILD_BENCHMARK "Enable building Zeno Benchmark" OFF)

############### BEGIN ADHOC ###############
if (UNIX)  # these are only used by archibate and zhxx1987

    if ($ENV{HOME} STREQUAL "/home/bate")
        message("-- BATE detected, making him happy")
        include(cmake/BATE.cmake)
    elseif ($ENV{HOME} STREQUAL "/home/dilei")
        message("-- ZHXX detected, making him happy")
        include(cmake/ZHXX.cmake)
    endif()

endif()  # normal users won't be affected
################ END ADHOC ################

if (UNIX)
    find_program(CCACHE_PROGRAM ccache)
    if (CCACHE_PROGRAM)
        message("-- Found CCache: ${CCACHE_PROGRAM}")
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
        set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CCACHE_PROGRAM}")
    endif()
endif()

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message("-- Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILD_SHARED_LIBS OFF)
if (WIN32)
    add_definitions(-DNOMINMAX -D_USE_MATH_DEFINES)
endif()

project(zeno VERSION 0.1 LANGUAGES CXX DESCRIPTION "Zeno node system for simulations")

if (ZENO_WITH_BACKWARD)
    message(STATUS "Building Backward-cpp")
    add_subdirectory(3rdparty/mods/backward-cpp)
endif()

message(STATUS "Building Zeno Core Library")
add_subdirectory(core)

if (ZENO_BUILD_WORKER)
    message(STATUS "Building Zeno Worker")
    add_subdirectory(worker)
endif()

if (ZENO_BUILD_EDITOR OR ZENO_BUILD_DESIGNER)
    message(STATUS "Building Zeno UI Library")
    add_subdirectory(3rdparty/qtmod/KDDockWidgets)
    add_subdirectory(zenoui)
endif()

if (ZENO_BUILD_EDITOR)
    message(STATUS "Building Zeno Editor")
    add_subdirectory(editor)
endif()

if (ZENO_BUILD_DESIGNER)
    message(STATUS "Building Zeno Designer")
    add_subdirectory(designer)
endif()

if (ZENO_BUILD_TESTS)
    message(STATUS "Building Zeno Tests")
    add_subdirectory(tests)
endif()

if (ZENO_BUILD_BENCHMARK)
    message(STATUS "Building Zeno Benchmark")
    add_subdirectory(benchmark)
endif()
