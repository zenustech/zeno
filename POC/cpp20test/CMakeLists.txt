cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_COMPILER clang)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} cmake)
project(cpp20test LANGUAGES C CXX)

include(Cpp20Modules)


add_library(helloworld OBJECT helloworld.cpp)
target_as_module(helloworld helloworld.cc)
add_executable(main main.cpp)
target_link_libraries(main helloworld)
