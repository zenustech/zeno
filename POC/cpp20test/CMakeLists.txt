cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_COMPILER clang)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} cmake)
include(Cpp20Modules)

project(cpp20test LANGUAGES C CXX)

include_directories(.)

add_module(hello)
add_module(world)

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /usr/local/lib/cmake/OpenVDB)
#find_package(OpenVDB REQUIRED)
#target_link_libraries(world PRIVATE OpenVDB::openvdb)

add_executable(main main.cc)
target_link_libraries(main PRIVATE hello world)
