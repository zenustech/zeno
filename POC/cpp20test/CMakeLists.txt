cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_COMPILER clang)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} cmake)
include(Cpp20Modules)

project(cpp20test LANGUAGES C CXX)


add_module(helloworld__hello)
add_module(helloworld__world)

add_executable(main main.cc)
target_link_libraries(main helloworld__hello helloworld__world)
