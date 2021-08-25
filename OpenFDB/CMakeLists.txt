cmake_minimum_required(VERSION 3.10)
project(openfdb LANGUAGES C CXX)

add_executable(openfdb main.cpp openvdb.cpp)
target_include_directories(openfdb PRIVATE .)

set(CMAKE_MODULE_PATH /usr/local/lib/cmake/OpenVDB)
find_package(OpenVDB REQUIRED)
target_link_libraries(openfdb PRIVATE OpenVDB::openvdb)
