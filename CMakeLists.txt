cmake_minimum_required(VERSION 3.10)
project(zeno LANGUAGES C CXX)

if (NOT CMAKE_BUILD_TYPE)
    message("Auto-setting CMAKE_BUILD_TYPE to RelWithDebInfo")
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(include)


option(ZENO_ENABLE_PYTHON "Enable ZENO Python bindings" ON)
if (ZENO_ENABLE_PYTHON)
    add_subdirectory(external/pybind11)
endif()

option(ZENO_GLOBALSTATE "Enable ZENO frame and substep support" ON)
option(ZENO_VISUALIZATION "Enable ZENO visualization support" ON)
if (ZENO_VISUALIZATION)
    add_subdirectory(external/glad)
    add_subdirectory(zenvis)
endif()

option(ZENO_ENABLE_OPENMP "Enable OpenMP in ZENO for parallelism" ON)
add_subdirectory(zeno)

option(ZENO_BUILD_EXTENSIONS "Build extensions for Zeno" ON)
if (ZENO_BUILD_EXTENSIONS)
    add_subdirectory(projects)
endif()

option(ZENO_BUILD_ZFX "Build ZFX module for Zeno" ON)
if (ZENO_BUILD_ZFX)
    add_subdirectory(ZFX)
endif()

option(ZENO_BUILD_TESTS "Build tests for Zeno" ON)
if (ZENO_BUILD_TESTS)
    add_subdirectory(tests)
endif()
