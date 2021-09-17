cmake_minimum_required(VERSION 3.10)
project(oneapitest LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_COMPILER dpcpp)

add_executable(oneapitest main.cpp)
