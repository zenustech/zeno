cmake_minimum_required(VERSION 3.18)

# https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
option(ZENO_WITH_ZPM "Use ZPM to manage Zeno dependencies" ON)
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

if (ZENO_WITH_ZPM)
    include(cmake/ZPM.cmake)
    
    #zpm_requires(3rdparty/pkgs/fmt -DFMT_DOC:BOOL=OFF -DFMT_TEST:BOOL=OFF)
    #zpm_requires(3rdparty/pkgs/spdlog -DSPDLOG_FMT_EXTERNAL:BOOL=ON -DSPDLOG_BUILD_EXAMPLE:BOOL=OFF)
    #zpm_requires(3rdparty/pkgs/range-v3 -DRANGE_V3_DOCS:BOOL=OFF -DRANGE_V3_TESTS:BOOL=OFF -DRANGE_V3_EXAMPLES:BOOL=OFF)
    #zpm_requires(3rdparty/pkgs/python-cmake-buildsystem)
    #zpm_requires(3rdparty/pkgs/BoostBuilder)
    #zpm_requires(3rdparty/pkgs/eigen)
    #zpm_requires(3rdparty/pkgs/zlib)
    #zpm_requires(3rdparty/pkgs/c-blosc -DBUILD_TESTS:BOOL=OFF)
    zpm_requires(3rdparty/pkgs/tbb -DTBB_BUILD_TESTS:BOOL=OFF -DTBB_BUILD_STATIC:BOOL=OFF)
    #zpm_requires(3rdparty/pkgs/openvdb)
    zpm_finalize()
endif()

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
