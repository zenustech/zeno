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

project(zeno
    LANGUAGES CXX
    DESCRIPTION "Open-source node system framework for simulation and others"
    HOMEPAGE_URL https://github.com/zenustech/zeno
    )

if (PROJECT_BINARY_DIR STREQUAL PROJECT_SOURCE_DIR)
    message(FATAL_ERROR "The binary directory of CMake cannot be the same as source directory!"
        "Please consider use 'cmake -B build' to specify a different binary directory."
        "Otherwise ZENO may fail to build, or make Git hard to exclude binary files.")
endif()

# Otherwise we can't link SHARED libs with STATIC libs
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build directory: ${PROJECT_BINARY_DIR}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if (WIN32)
    # get rid of 'min', 'max' macros; get back 'M_PI', 'M_E'
    add_definitions(-DNOMINMAX -D_USE_MATH_DEFINES)
endif()

option(ZENO_BUILD_EDITOR "Build ZENO editor" ON)
option(ZENO_BUILD_DESIGNER "Build ZENO designer" OFF)
option(ZENO_BENCHMARKING "Enable ZENO benchmarking timer" ON)
option(ZENO_MULTIPROCESS "Enable multiprocessing for ZENO" ON)
option(ZENO_PARALLEL_STL "Enable parallel STL in ZENO" OFF)
option(ZENO_ENABLE_BACKWARD "Enable ZENO fault handler for traceback" OFF)
option(ZENO_ENABLE_OPENMP "Enable OpenMP in ZENO for parallelism" ON)
option(ZENO_ENABLE_SPDLOG "Enable spdlog in ZENO for logging" OFF)
option(ZENO_ENABLE_MAGICENUM "Enable magicenum in ZENO for enum reflection" ON)
option(ZENO_OUT_TO_BIN "Output files to a fixed directory" ON)
option(ZENO_BUILD_SHARED "Build shared library for ZENO" ON)
option(ZENO_USE_CCACHE "Use CCache if found in path" ON)
option(ZENO_NO_WARNING "Use customized compiler warnings" ON)
option(ZENO_INSTALL_TARGET "Enable installation targets" OFF)

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
    if (WIN32)
        set(CMAKE_DEBUG_POSTFIX "")  # No more annoying zeno_d.dll
    endif()
endif()

if (ZENO_NO_WARNING)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        add_compile_options(-Wno-all -Wno-cpp -Wno-deprecated-declarations -Wno-enum-compare -Wno-ignored-attributes -Wno-extra -Werror=return-type -Wnon-virtual-dtor -Wsuggest-override -Wmissing-declarations -Wconversion-null)
    endif()
    if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        add_compile_options(-w)
    endif()
endif()

if (ZENO_ENABLE_BACKWARD)
    add_subdirectory(zeno/tpls/backward-cpp)
endif()

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
        FILE ZenoTargets.cmake
        NAMESPACE Zeno::
        DESTINATION lib/cmake/Zeno
        )
    configure_file(misc/ZenoConfig.cmake.in ZenoConfig.cmake @ONLY)
    install(
        FILES ${CMAKE_CURRENT_BINARY_DIR}/ZenoConfig.cmake
        DESTINATION lib/cmake/Zeno
        )
endif()
