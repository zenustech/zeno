cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
project(openfdb LANGUAGES C CXX)
#set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)

add_executable(openfdb main.cpp)
target_include_directories(openfdb PRIVATE .)

set(CMAKE_MODULE_PATH /usr/local/lib/cmake/OpenVDB)
find_package(OpenVDB REQUIRED)
target_link_libraries(openfdb PRIVATE OpenVDB::openvdb)
