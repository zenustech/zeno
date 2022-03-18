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
    LANGUAGES C CXX
    VERSION 0.0.1
    DESCRIPTION "Open-source node system framework for simulation and others"
    HOMEPAGE_URL https://github.com/zenustech/zeno
    )

if (PROJECT_BINARY_DIR STREQUAL PROJECT_SOURCE_DIR)
    message(WARNING "The binary directory of CMake cannot be the same as source directory!"
        "Please consider use 'cmake -B build' to specify a different binary directory."
        "Otherwise ZENO may fail to build, or make Git hard to exclude binary files.")
endif()

# Otherwise we can't link SHARED libs with STATIC libs
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if (WIN32)
    # get rid of 'min', 'max' macros; get back 'M_PI', 'M_E'
    add_definitions(-DNOMINMAX -D_USE_MATH_DEFINES)
endif()

option(ZENO_BUILD_EDITOR "Build ZENO Editor" ON)
option(ZENO_BUILD_DESIGNER "Build ZENO Designer" OFF)
option(ZENO_BUILD_TESTER "Build ZENO Quick Tester" OFF)
option(ZENO_FAULTHANDLER "Enable ZENO fault handler for traceback" ON)
option(ZENO_BENCHMARKING "Enable ZENO benchmarking timer" ON)
option(ZENO_ENABLE_OPENMP "Enable OpenMP in ZENO for parallelism" ON)
option(ZENO_ENABLE_SPDLOG "Enable spdlog in ZENO for logging" OFF)
option(ZENO_ENABLE_MAGICENUM "Enable magicenum in ZENO for enum reflection" ON)
option(ZENO_MULTIPROCESS "Enable multiprocessing for ZENO" ON)
option(ZENO_OUT_TO_BIN "Output files to a fixed directory" ON)
option(ZENO_USE_CCACHE "Use CCache if found in path" ON)
option(ZENO_NO_WARNING "Disable compiler warnings" ON)
option(ZENO_INSTALL_TARGET "Enable installation targets" ON)

if (ZENO_USE_CCACHE AND NOT MSVC)  # speed up compilation by cache (https://ccache.dev)
    find_program(CCACHE_PROGRAM ccache)
    if (CCACHE_PROGRAM)
        message(STATUS "Found CCache: ${CCACHE_PROGRAM}")
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_PROGRAM})
        set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ${CCACHE_PROGRAM})
    endif()
endif()

if (ZENO_NO_WARNING)
    if (CMAKE_COMPILER_IS_GNUCC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -Werror=return-type")
    endif()
    if (MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w")
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

if (ZENO_FAULTHANDLER)
    add_subdirectory(external/backward-cpp)
endif()

if (ZENO_ENABLE_SPDLOG)
    add_subdirectory(external/spdlog)
endif()

add_subdirectory(external/include)
add_subdirectory(zeno)
add_subdirectory(projects)

if (ZENO_MULTIPROCESS)
    message(STATUS "Building Zeno Runner")
    add_subdirectory(ui/zenorunner)
endif()

if (ZENO_BUILD_EDITOR OR ZENO_BUILD_DESIGNER)
    message(STATUS "Building Zeno UI Library")
    add_subdirectory(ui/zenoui)
    add_subdirectory(ui/zenoio)
endif()

if (ZENO_BUILD_EDITOR)
    message(STATUS "Building Zeno Editor")
    add_subdirectory(external/glad)
    add_subdirectory(ui/zenovis)
    add_subdirectory(ui/zenoedit)
endif()

if (ZENO_BUILD_DESIGNER)
    message(STATUS "Building Zeno Designer")
    add_subdirectory(ui/zenodesign)
endif()

if (ZENO_BUILD_TESTER)
    message(STATUS "Building Zeno Quick Tester")
    add_subdirectory(ui/zenotest)
endif()

if (ZENO_INSTALL_TARGET AND NOT WIN32)
    message(STATUS "Adding Zeno Installation Target")
    set(targets zenodes zenocore zenoinclude zenoedit)
    if (ZENO_MULTIPROCESS)
        list(APPEND targets zenorunner)
    endif()
    install(
        TARGETS ${targets}
        EXPORT ZenoTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        PUBLIC_HEADER DESTINATION include
        )
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
