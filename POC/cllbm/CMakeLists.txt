cmake_minimum_required(VERSION 3.10)
project(cllbm LANGUAGES C CXX)

add_executable(cllbm main.cpp stbi.cpp)
target_link_libraries(cllbm PRIVATE OpenCL)
