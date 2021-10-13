cmake_minimum_required(VERSION 3.18)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    message("-- Setting CMAKE_BUILD_TYPE to Release")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_program(CCACHE_FOUND ccache)
if (CCACHE_FOUND)
    message("-- Using CCache for compile and link")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif()

project(zeno LANGUAGES C CXX)

add_subdirectory(tpls)
add_subdirectory(src)
add_subdirectory(editor)
add_subdirectory(tests)
