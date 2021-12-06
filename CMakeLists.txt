cmake_minimum_required(VERSION 3.18)

option(ZENO_WITH_LEGACY "Build with legacy nodes" OFF)
option(ZENO_WITH_BACKWARD "Enable stack backtrace" OFF)
option(ZENO_BUILD_EDITOR "Build Zeno Editor" ON)
option(ZENO_BUILD_DESIGNER "Build Zeno Designer" ON)
option(ZENO_BUILD_WORKER "Build Zeno Worker" OFF)
option(ZENO_BUILD_TESTS "Build Zeno Tests" OFF)
option(ZENO_BUILD_BENCHMARK "Build Zeno Benchmark" OFF)

############### BEGIN ADHOC ###############
if (UNIX)  # these are only used by archibate and zhxx1987

    if ($ENV{HOME} STREQUAL "/home/bate")
        message(STATUS "BATE detected, making him happy")
        include(cmake/BATE.cmake)
    elseif ($ENV{HOME} STREQUAL "/home/dilei")
        message(STATUS "ZHXX detected, making him happy")
        include(cmake/ZHXX.cmake)
    endif()

endif()  # normal users won't be affected
################ END ADHOC ################

if (UNIX)
    find_program(CCACHE_PROGRAM ccache)
    if (CCACHE_PROGRAM)
        message(STATUS "Found CCache: ${CCACHE_PROGRAM}")
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
        set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CCACHE_PROGRAM}")
    endif()
endif()

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILD_SHARED_LIBS OFF)
if (WIN32)
    message(STATUS "Adding flags: -DNOMINMAX -D_USE_MATH_DEFINES")
    add_definitions(-DNOMINMAX -D_USE_MATH_DEFINES)
endif()

project(zeno
    VERSION 0.1
    LANGUAGES CXX
    HOMEPAGE_URL "https://github.com/zenustech/zeno"
    DESCRIPTION "Zeno node system for simulations"
    )

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
    add_subdirectory(3rdparty/mods/googletest)
    add_subdirectory(tests)
endif()

if (ZENO_BUILD_BENCHMARK)
    message(STATUS "Building Zeno Benchmark")
    add_subdirectory(3rdparty/mods/benchmark)
    add_subdirectory(benchmark)
endif()
