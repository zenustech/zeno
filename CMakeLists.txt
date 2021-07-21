cmake_minimum_required(VERSION 3.10)
project(zeno LANGUAGES C CXX)

if (NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(include)

add_subdirectory(external/pybind11)
add_subdirectory(external/glad)
add_subdirectory(zenvis)


file(GLOB ZENO_SOURCE zeno/*.cpp zeno/nodes/*.cpp zeno/nodes/*/*.cpp zeno/include/*/*.h)
add_library(zeno SHARED ${ZENO_SOURCE})
if (UNIX)
	target_link_libraries(zeno PRIVATE stdc++fs)
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

target_compile_definitions(zeno PRIVATE -DDLL_ZENO)
target_include_directories(zeno PUBLIC zeno/include)
set(OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/zeno/lib)
set_target_properties(zeno PROPERTIES
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

file(GLOB PYZENO_SOURCE pyzeno/*.cpp)
pybind11_add_module(pyzeno ${PYZENO_SOURCE})
target_link_libraries(pyzeno PRIVATE zeno)
set(OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/zeno)
set_target_properties(pyzeno PROPERTIES
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
