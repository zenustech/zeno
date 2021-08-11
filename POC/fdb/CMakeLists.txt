cmake_minimum_required(VERSION 3.10)
project(adflip LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

add_executable(adflip mgflip.cpp vdbio.cpp timer.cpp)

find_package(OpenMP REQUIRED)
target_link_libraries(adflip OpenMP::OpenMP_CXX)

if (UNIX)
    set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../projects/cmake ${CMAKE_MODULE_PATH})
endif()

find_package(OpenVDB REQUIRED)
target_link_libraries(adflip OpenVDB::openvdb)
target_include_directories(adflip PRIVATE ../../include)
