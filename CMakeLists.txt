cmake_minimum_required(VERSION 3.18)

# https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
set(HIPSYCL_TARGETS "omp;cuda:sm_52;cuda:sm_61;cuda:sm_70;cuda:sm_75;cuda:sm_86" CACHE STRING "Specify the hipSYCL targets to build against")
set(ZENO_TARGET "editor" CACHE STRING "Specify the hipSYCL targets to build against")
option(ZENO_WITH_SYCL "Enable SYCL support for Zeno" OFF)
option(ZENO_WITH_BACKWARD "Enable stack backtrace for Zeno" OFF)
option(ZENO_WITH_LEGACY "Build Zeno With Legacy Nodes" OFF)

###################
### BEGIN ADHOC ###
###################
if (UNIX)
    if ($ENV{HOME} STREQUAL "/home/dilei")
        message("-- ZHXX detected, making him happy")
        set(ZHXX ON)
    elseif ($ENV{HOME} STREQUAL "/home/bate")
        message("-- BATE detected, making him happy")
        set(BATE ON)
    endif()
endif()
if (BATE)
    set(HIPSYCL_TARGETS "omp")
    set(ZENO_WITH_SYCL OFF)
    set(ZENO_WITH_BACKWARD ON)
    set(ZENO_WITH_LEGACY OFF)
    set(ZENO_TARGET editor)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /usr/local/lib/cmake/hipSYCL)
    set(CMAKE_BUILD_TYPE Debug)
endif()
if (ZHXX)
    set(HIPSYCL_TARGETS "omp;cuda:sm_86")
    set(ZENO_WITH_SYCL OFF)
    set(ZENO_WITH_BACKWARD ON)
    set(ZENO_WITH_LEGACY ON)
    set(ZENO_TARGET editor)
    set(CMAKE_BUILD_TYPE Release)
endif()
###################
#### END ADHOC ####
###################

if (UNIX)
    find_program(CCACHE_PROGRAM ccache)
    if (CCACHE_PROGRAM)
        message("-- Found CCache: ${CCACHE_PROGRAM}")
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_PROGRAM})
        set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ${CCACHE_PROGRAM})
    endif()
endif()

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message("-- Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(BUILD_SHARED_LIBS OFF)
if (WIN32)
    add_definitions(-DNOMINMAX -D_USE_MATH_DEFINES)
endif()

project(zeno VERSION 0.1 LANGUAGES C CXX)

add_subdirectory(tpls)

if (ZENO_WITH_SYCL)
    message("-- Building Zeno with hipSYCL targets: [${HIPSYCL_TARGETS}]")
    find_package(hipSYCL CONFIG REQUIRED)
else()
    message(WARNING "Building Zeno with host emulated SYCL")
endif()

message("-- Loading Rules for Zeno Target: [${ZENO_TARGET}]")
add_subdirectory(${ZENO_TARGET})

if (ZENO_WITH_SYCL)
    target_compile_definitions(zeno PUBLIC -DZENO_WITH_SYCL)
endif()

if (ZENO_WITH_BACKWARD)
    message("-- Building Zeno with Stack Traceback")
    target_sources(zeno PRIVATE ${BACKWARD_ENABLE})
    add_backward(zeno)
endif()

message("-- Building Zeno Core Library")
add_subdirectory(src)

if (ZENO_WITH_LEGACY)
    message("-- Building Zeno Legacy Library")
    add_subdirectory(legacy)
endif()
