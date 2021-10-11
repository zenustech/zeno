cmake_minimum_required(VERSION 3.18)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILD_SHARED_LIBS OFF)

project(zeno LANGUAGES C CXX)

add_subdirectory(tpls)
add_subdirectory(src)
add_subdirectory(main)
add_subdirectory(tests)
