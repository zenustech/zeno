cmake_minimum_required(VERSION 3.15)

# Please use VS2019+ or GCC 9+
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (PROJECT_BINARY_DIR STREQUAL PROJECT_SOURCE_DIR)
    message(WARNING "The binary directory of CMake cannot be the same as source directory!"
        "Please consider use 'cmake -B build' to specify a different binary directory."
        "Otherwise ZENO may fail to build, or make Git hard to exclude binary files.")
endif()

project(zeno LANGUAGES C CXX)

# Otherwise we can't link SHARED libs with STATIC libs
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if (WIN32)
    # get rid of 'min', 'max' macros; get back 'M_PI', 'M_E'
    add_definitions(-DNOMINMAX -D_USE_MATH_DEFINES)
endif()

option(ZENO_BUILD_EDITOR "Build ZENO Editor" ON)
option(ZENO_BUILD_DESIGNER "Build ZENO Designer" OFF)
option(ZENO_BUILD_EXTENSIONS "Build extension modules for ZENO" ON)
option(ZENO_FAULTHANDLER "Enable ZENO fault handler for traceback" ON)
option(ZENO_BENCHMARKING "Enable ZENO benchmarking timer" ON)
option(ZENO_ENABLE_OPENMP "Enable OpenMP in ZENO for parallelism" ON)
option(ZENO_ENABLE_SPDLOG "Enable spdlog in ZENO for logging" ON)
option(ZENO_MULTIPROCESS "Enable multiprocessing for ZENO" ON)
option(ZENO_OUT_TO_BIN "Output files to a fixed directory" ON)
option(ZENO_USE_CCACHE "Use CCache if found in path" ON)
option(ZENO_NO_WARNING "Disable compiler warnings" ON)

if (ZENO_USE_CCACHE)  # speed up compilation by cache
    find_program(CCACHE_PROGRAM ccache)
    if (CCACHE_PROGRAM)
        message(STATUS "Found CCache: ${CCACHE_PROGRAM}")
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_PROGRAM})
        set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ${CCACHE_PROGRAM})
    endif()
endif()

if (ZENO_NO_WARNING)
    if (CMAKE_COMPILER_IS_GNUCC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
    endif()
    if (MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w")
    endif()
endif()

if (ZENO_OUT_TO_BIN)
    # Otherwise Windows (which doesn't support RPATH) cannot find the DLLs at runtime
    set(OUTPUT_DIR ${PROJECT_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_DIR})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_DIR})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR})
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
add_subdirectory(src/zeno)

if (ZENO_BUILD_EXTENSIONS)
    message(STATUS "Building Zeno Extensions")
    add_subdirectory(projects)
endif()

if (ZENO_MULTIPROCESS)
    message(STATUS "Building Zeno Runner")
    add_subdirectory(src/zenorunner)
endif()

if (ZENO_BUILD_EDITOR OR ZENO_BUILD_DESIGNER)
    message(STATUS "Building Zeno UI Library")
    add_subdirectory(src/zenoui)
    add_subdirectory(src/zenoio)
endif()

if (ZENO_BUILD_EDITOR)
    message(STATUS "Building Zeno Editor")
    add_subdirectory(external/glad)
    add_subdirectory(src/zenovis)
    add_subdirectory(src/zenoedit)
endif()

if (ZENO_BUILD_DESIGNER)
    message(STATUS "Building Zeno Designer")
    add_subdirectory(src/zenodesign)
endif()
