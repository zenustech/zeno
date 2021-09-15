cmake_minimum_required(VERSION 3.10)
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 17)
project(openfdb LANGUAGES CXX)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
find_package(OpenCL REQUIRED)

add_executable(openfdb main.cpp test.cl)
set_target_properties(openfdb PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(openfdb PRIVATE .)
target_include_directories(openfdb PRIVATE ${OPENCL_INCLUDE_DIRS})
target_link_libraries(openfdb PRIVATE ${OPENCL_LIBRARIES})
