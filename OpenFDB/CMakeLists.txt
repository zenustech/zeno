cmake_minimum_required(VERSION 3.10)
project(openfdb LANGUAGES C CXX)

add_executable(openfdb main.cpp openvdb.cpp)
target_link_libraries(openfdb PRIVATE tbb)
target_include_directories(openfdb PRIVATE .)
