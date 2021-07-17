cmake_minimum_required(VERSION 3.17)
project(adaptivegrid LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
set(CMAKE_BUILD_TYPE Release)

add_executable(adaptivegrid)
target_sources(adaptivegrid PRIVATE main.cu)
target_include_directories(adaptivegrid PRIVATE inc)
target_link_libraries(adaptivegrid PRIVATE OpenGL glut)
