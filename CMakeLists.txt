cmake_minimum_required(VERSION 3.18)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    message("-- Setting CMAKE_BUILD_TYPE to Release")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if (WIN32)
    add_definitions(-DNOMINMAX -D_USE_MATH_DEFINES)
endif()

find_program(CCACHE_PROGRAM ccache)
if (CCACHE_PROGRAM)
    message("-- Found CCache: ${CCACHE_PROGRAM}")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_PROGRAM})
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ${CCACHE_PROGRAM})
endif()

project(zeno LANGUAGES C CXX)

add_subdirectory(tpls)
add_subdirectory(src)

option(ZENO_WITH_EDITOR "Build Zeno Qt5 editor" ON)
if (ZENO_WITH_EDITOR)
    message("-- Building Zeno Qt5 Editor")
    add_subdirectory(editor)
endif()

option(ZENO_WITH_TESTS "Build Zeno Google Tests" ON)
if (ZENO_WITH_TESTS)
    message("-- Building Zeno Google Tests")
    enable_testing()
    add_subdirectory(tests)
endif()
