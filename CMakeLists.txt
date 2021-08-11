cmake_minimum_required(VERSION 3.10)
project(zeno LANGUAGES C CXX)

if (NOT CMAKE_BUILD_TYPE)
    if (WIN32)
        message("Auto-setting CMAKE_BUILD_TYPE to Release")
        set(CMAKE_BUILD_TYPE Release)
    else()
        message("Auto-setting CMAKE_BUILD_TYPE to RelWithDebInfo")
        set(CMAKE_BUILD_TYPE RelWithDebInfo)
    endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if (WIN32)
	set(CMAKE_DEBUG_POSTFIX "")
endif()

include_directories(include)

option(ZENO_BUILD_EXTENSIONS "Build extension modules for Zeno" ON)
option(ZENO_BUILD_ZFX "Build ZFX module for Zeno" ON)
option(ZENO_BUILD_TESTS "Build tests for Zeno" OFF)
option(ZENO_GLOBALSTATE "Enable ZENO frame and substep support" ON)
option(ZENO_VISUALIZATION "Enable ZENO visualization support" ON)
option(ZENO_FAULTHANDLER "Enable ZENO fault handler for traceback" ON)
option(ZENO_BENCHMARKING "Enable ZENO benchmarking timer" ON)
option(ZENO_ENABLE_PYTHON "Enable ZENO Python bindings" ON)
option(ZENO_ENABLE_OPENMP "Enable OpenMP in ZENO for parallelism" ON)

function(add_zeno_autoload target)
    if (ZENO_ENABLE_PYTHON)
        set(OUTPUT_DIR ${PROJECT_SOURCE_DIR}/zenqt/lib)
        set_target_properties(${target} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_DIR}
            RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR}
            ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_DIR}
            LIBRARY_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR}
            RUNTIME_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR}
            ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR}
            LIBRARY_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR}
            RUNTIME_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR}
            ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR}
            )
    endif()
endfunction()

if (ZENO_ENABLE_PYTHON)
    add_subdirectory(external/pybind11)
endif()

add_subdirectory(external/spdlog)
add_subdirectory(zeno)

if (ZENO_ENABLE_PYTHON)
    add_subdirectory(pyzeno)
endif()

if (ZENO_VISUALIZATION)
    add_subdirectory(external/glad)
    add_subdirectory(zenvis)
endif()

if (ZENO_BUILD_ZFX)
    add_subdirectory(ZFX)
endif()

if (ZENO_BUILD_EXTENSIONS)
    add_subdirectory(projects)
endif()

if (ZENO_BUILD_TESTS)
    add_subdirectory(tests)
endif()
