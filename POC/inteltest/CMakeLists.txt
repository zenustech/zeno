cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_COMPILER "/opt/sycl/bin/clang++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath=/opt/sycl/lib -std=c++17 -fsycl -fsycl-targets=nvptx64-nvidia-cuda")

project(inteltest LANGUAGES CXX)

add_executable(main main.cpp)
