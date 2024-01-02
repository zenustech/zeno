cmake_minimum_required(VERSION 3.15)

if (DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    message(STATUS "VCPKG_ROOT: $ENV{VCPKG_ROOT}")
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

if (DEFINED ENV{VCPKG_TRIPLET} AND NOT DEFINED VCPKG_TARGET_TRIPLET)
    message(STATUS "VCPKG_TRIPLET: $ENV{VCPKG_TRIPLET}")
    set(VCPKG_TARGET_TRIPLET "$ENV{VCPKG_TRIPLET}" CACHE STRING "")
endif()

# Please use VS2019+ or GCC 9+
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(Zeno
    LANGUAGES CXX C
    DESCRIPTION "Open-source node system framework for simulation and others"
    HOMEPAGE_URL https://github.com/zenustech/zeno
    )

if (PROJECT_BINARY_DIR STREQUAL PROJECT_SOURCE_DIR)
    message(FATAL_ERROR "The binary directory of CMake cannot be the same as source directory!"
        "Please consider use 'cmake -B build' to specify a different binary directory."
        "Otherwise ZENO may fail to build, or make Git hard to exclude binary files.")
endif()

option(ZENO_BUILD_EDITOR "Build ZENO editor" ON)
option(ZENO_BUILD_DESIGNER "Build ZENO designer" OFF)
option(ZENO_BUILD_PLAYER "Build ZENO player" OFF)
option(ZENO_MULTIPROCESS "Enable multiprocessing for ZENO" ON)
option(ZENO_IPC_USE_TCP "Use TCP for inter-process communication" ON)
option(ZENO_OUT_TO_BIN "Output all target files to build/bin" ON)
option(ZENO_BUILD_SHARED "Build shared library for ZENO" ON)
option(ZENO_USE_CCACHE "Use CCache if found in path" ON)
option(ZENO_NO_WARNING "Use customized compiler warnings" ON)
option(ZENO_INSTALL_TARGET "Enable installation targets" OFF)
option(ZENO_WITH_CUDA "Build ZENO with CUDA support" OFF)
option(ZENO_MARCH_NATIVE "Build ZENO with -march=native" OFF)
option(ZENO_USE_FAST_MATH "Build ZENO with -ffast-math" OFF)
option(ZENO_OPTIX_PROC "Optix with a new proc" OFF)

if (NOT DEFINED CMAKE_POSITION_INDEPENDENT_CODE)
    # Otherwise we can't link .so libs with .a libs
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()
if (MSVC AND NOT DEFINED CMAKE_MSVC_RUNTIME_LIBRARY)
    # MSVC sucks /MD /MT /MDd /MTd options....
    # MultiThreadedDLL = /MD, MultiThreadedDebugDLL = /MDd
    # MultiThreaded = /MT, MultiThreadedDebug = /MTd
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build directory: ${PROJECT_BINARY_DIR}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if (WIN32)
    # get rid of 'min', 'max' macros; get back 'M_PI', 'M_E'
    add_definitions(-DNOMINMAX -D_USE_MATH_DEFINES)
endif()

if (ZENO_USE_CCACHE AND NOT MSVC)  # speed up compilation by cache (https://ccache.dev)
    find_program(CCACHE_PROGRAM ccache)
    if (CCACHE_PROGRAM)
        message(STATUS "Found CCache: ${CCACHE_PROGRAM}")
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_PROGRAM})
        set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ${CCACHE_PROGRAM})
    endif()
endif()

if (ZENO_OUT_TO_BIN)
    # Otherwise Windows (which doesn't support RPATH) cannot find the DLLs at runtime
    set(ZENO_OUTPUT_DIR ${PROJECT_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${ZENO_OUTPUT_DIR})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${ZENO_OUTPUT_DIR})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${ZENO_OUTPUT_DIR})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${ZENO_OUTPUT_DIR})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${ZENO_OUTPUT_DIR})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${ZENO_OUTPUT_DIR})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${ZENO_OUTPUT_DIR})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${ZENO_OUTPUT_DIR})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${ZENO_OUTPUT_DIR})
    set(CMAKE_DEBUG_POSTFIX "")  # No more annoying zeno_d.dll
endif()

if (ZENO_NO_WARNING)  # Use our customized warning flags
    set(ZENO_OLD_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-all -Wno-cpp -Wno-deprecated-declarations -Wno-enum-compare -Wno-ignored-attributes -Wno-extra -Wreturn-type -Wmissing-declarations -Wnon-virtual-dtor -Wsuggest-override -Wconversion-null -Wuninitialized -Wno-format-security -Wnarrowing -Wno-maybe-uninitialized")
    endif()
    if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
    endif()
endif()

## --- begin cihou pyb nowarn
macro (zeno_disable_warning)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set_source_files_properties(${ARGV} DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} TARGET_DIRECTORY zeno PROPERTIES COMPILE_OPTIONS $<IF:$<COMPILE_LANGUAGE:CXX>,-w,>)
    endif()
endmacro()
## --- end cihou pyb nowarn

macro (zeno_dbg_msvc project)
    if (WIN32)
        target_compile_options(${project} PRIVATE /Zi /O2 /Ob2)
        # Tell linker to include symbol data
        set_target_properties(${project} PROPERTIES 
            LINK_FLAGS "/INCREMENTAL:NO /DEBUG /OPT:REF /OPT:ICF"
        )
        # Set file name & location
        set_target_properties(${project} PROPERTIES 
            COMPILE_PDB_NAME ${project}
            COMPILE_PDB_OUTPUT_DIR ${project}
        )
    endif()
endmacro()

## --- begin cihou wxl cuda
if (ZENO_WITH_CUDA)        # must move enable_language ahead to here
    message(STATUS "CUDA_PATH: "$ENV{CUDA_PATH})
    if (NOT DEFINED CUDAToolkit_ROOT AND NOT DEFINED ENV{CUDAToolkit_ROOT} AND DEFINED ENV{CUDA_PATH})
        set(CUDAToolkit_ROOT $ENV{CUDA_PATH} CACHE FILEPATH "ZENOCIHOUCUDA" FORCE)
        if (NOT DEFINED CUDAToolkit_NVCC_EXECUTABLE)
            set(CUDAToolkit_ROOT_DIR ${CUDAToolkit_ROOT} CACHE FILEPATH "ZENOCIHOUCUDA" FORCE)
            set(CUDAToolkit_BIN_DIR ${CUDAToolkit_ROOT}/bin CACHE FILEPATH "ZENOCIHOUCUDA" FORCE)
            set(CUDAToolkit_NVCC_EXECUTABLE ${CUDAToolkit_BIN_DIR}/nvcc${CMAKE_EXECUTABLE_SUFFIX} CACHE FILEPATH "ZENOCIHOUCUDA" FORCE)
            set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE} CACHE FILEPATH "ZENOCIHOUCUDA" FORCE)
        endif()
    endif()
    message(STATUS "CUDAToolkit_ROOT: "${CUDAToolkit_ROOT})
    message(STATUS "CUDAToolkit_ROOT_DIR: "${CUDAToolkit_ROOT_DIR})
    message(STATUS "CUDAToolkit_NVCC_EXECUTABLE: "${CUDAToolkit_NVCC_EXECUTABLE})
    message(STATUS "CUDAToolkit_BIN_DIR: "${CUDAToolkit_BIN_DIR})
    message(STATUS "Now invoking FindCUDAToolkit.cmake")
    find_package(CUDAToolkit REQUIRED COMPONENTS cufft REQUIRED)
    message(STATUS "CUDAToolkit_ROOT: "${CUDAToolkit_ROOT})
    message(STATUS "CUDAToolkit_ROOT_DIR: "${CUDAToolkit_ROOT_DIR})
    message(STATUS "CUDAToolkit_NVCC_EXECUTABLE: "${CUDAToolkit_NVCC_EXECUTABLE})
    message(STATUS "CUDAToolkit_BIN_DIR: "${CUDAToolkit_BIN_DIR})
    message(STATUS "CUDAToolkit_INCLUDE_DIRS: "${CUDAToolkit_INCLUDE_DIRS})
    message(STATUS "CUDAToolkit_LIBRARY_DIR: "${CUDAToolkit_LIBRARY_DIR})
    message(STATUS "CUDAToolkit_VERSION: "${CUDAToolkit_VERSION})
    message(STATUS "CMAKE_CUDA_COMPILER: "${CMAKE_CUDA_COMPILER})
    message(STATUS "CMAKE_CUDA_COMPILER_VERSION: "${CMAKE_CUDA_COMPILER_VERSION})
    enable_language(CUDA)  # otherwise projects/CUDA will fail to compile
endif()
## --- end cihou wxl cuda

## --- begin cihou asset dir
set(zeno_asset_dirs_txt ${CMAKE_CURRENT_BINARY_DIR}/zeno_asset_dirs.txt)
if (EXISTS ${zeno_asset_dirs_txt})
    file(REMOVE ${zeno_asset_dirs_txt})
endif()
function (zeno_add_asset_dir dir)
    if (ZENO_INSTALL_TARGET)
        install(
            DIRECTORY ${dir}
            DESTINATION share/Zeno/assets/${dir}
            )
    endif()
    file(APPEND ${zeno_asset_dirs_txt} "${dir}\n")
endfunction()
## --- end cihou asset dir

add_subdirectory(zenocore)
add_subdirectory(zeno)

## --- begin cihou perf-geeks
if (ZENO_MARCH_NATIVE)
    if (MSVC)
        target_compile_options(zeno PRIVATE /arch:AVX)
    else()
        target_compile_options(zeno PRIVATE "-march=native")
    endif()
endif()
if (ZENO_USE_FAST_MATH)
    if (MSVC)
        target_compile_options(zeno PRIVATE /fp:fast)
    else()
        target_compile_options(zeno PRIVATE "-ffast-math")
    endif()
endif()
## --- end cihou perf-geeks

target_compile_options(zeno
            PUBLIC $<$<COMPILE_LANGUAGE:CXX>: $<IF:$<CXX_COMPILER_ID:MSVC>, /utf-8, >>
            )
if (ZENO_BUILD_EDITOR OR ZENO_BUILD_DESIGNER)
    message(STATUS "Building Zeno UI Library")
    add_subdirectory(ui/zenoui)
    add_subdirectory(ui/zenoio)
    add_subdirectory(ui/zenomodel)
endif()

if (ZENO_BUILD_EDITOR)
    message(STATUS "Building Zeno Editor")
    add_subdirectory(ui/zenoedit)
    add_subdirectory(zenovis)
endif()

add_subdirectory(projects)

if (ZENO_BUILD_DESIGNER)
    message(STATUS "Building Zeno Designer")
    add_subdirectory(ui/zenodesign)
endif()

if (ZENO_BUILD_PLAYER)
    message(STATUS "Building Zeno Player")
    add_subdirectory(ui/zenoplayer)
endif()

#add_subdirectory(embed)

if (ZENO_INSTALL_TARGET)
    install(
        EXPORT ZenoTargets
        FILE ZenoConfig.cmake
        NAMESPACE Zeno::
        DESTINATION lib/cmake/Zeno
        )
    #configure_file(misc/ZenoConfig.cmake.in ZenoConfig.cmake @ONLY)
    #install(
        #FILES ${CMAKE_CURRENT_BINARY_DIR}/ZenoConfig.cmake
        #DESTINATION lib/cmake/Zeno
        #)
endif()
