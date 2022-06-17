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
option(ZENO_MULTIPROCESS "Enable multiprocessing for ZENO" ON)
option(ZENO_IPC_USE_TCP "Use TCP for inter-process communication" ON)
option(ZENO_OUT_TO_BIN "Output files to a fixed directory" ON)
option(ZENO_BUILD_SHARED "Build shared library for ZENO" OFF)
option(ZENO_USE_CCACHE "Use CCache if found in path" ON)
option(ZENO_NO_WARNING "Use customized compiler warnings" ON)
option(ZENO_INSTALL_TARGET "Enable installation targets" OFF)

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

if (ZENO_NO_WARNING)
    set(ZENO_OLD_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-all -Wno-cpp -Wno-deprecated-declarations -Wno-enum-compare -Wno-ignored-attributes -Wno-extra -Wreturn-type -Wmissing-declarations -Wnon-virtual-dtor -Wsuggest-override -Wconversion-null -Wuninitialized -Wmaybe-uninitialized -Wno-format-security -Wnarrowing")
    endif()
    if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
    endif()
endif()

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

add_subdirectory(zeno)
add_subdirectory(projects)

if (ZENO_BUILD_EDITOR OR ZENO_BUILD_DESIGNER)
    message(STATUS "Building Zeno UI Library")
    add_subdirectory(ui/zenoui)
    add_subdirectory(ui/zenoio)
endif()

if (ZENO_BUILD_EDITOR)
    message(STATUS "Building Zeno Editor")
    add_subdirectory(ui/zenoedit)
    add_subdirectory(zenovis)
endif()

if (ZENO_BUILD_DESIGNER)
    message(STATUS "Building Zeno Designer")
    add_subdirectory(ui/zenodesign)
endif()

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
