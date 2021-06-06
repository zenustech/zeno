cmake_minimum_required(VERSION 3.12)
project(demo_project LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE RelWithDbgInfo)  # WARNING: YOU MUST USE RELEASE BUILD ON WINDOWS THANKS TO THE STUPID MECHINISM OF MSVC!
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../share/cmake)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(zen REQUIRED)

add_library(demo_project SHARED MyNode.cpp)
target_link_libraries(demo_project PRIVATE zen)
set_property(TARGET demo_project PROPERTY LIBRARY_OUTPUT_DIRECTORY ${zen_INSTALL_DIR}/autoload)
set_property(TARGET demo_project PROPERTY RUNTIME_OUTPUT_DIRECTORY ${zen_INSTALL_DIR}/autoload)
