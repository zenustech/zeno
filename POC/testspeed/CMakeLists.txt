cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(ZenoEditor LANGUAGES C CXX)

file(GLOB PROJECT_SOURCE *.cpp)
add_executable(ZenoEditor ${PROJECT_SOURCE})

set(CMAKE_MODULE_PATH /usr/local/lib/cmake/OpenVDB)
find_package(OpenVDB REQUIRED)
target_link_libraries(ZenoEditor PRIVATE OpenVDB::openvdb)
