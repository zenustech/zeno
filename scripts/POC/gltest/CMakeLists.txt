cmake_minimum_required(VERSION 3.10)
project(gltest LANGUAGES C CXX)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)

add_executable(gltest)
target_include_directories(gltest PRIVATE .)

if (NOT TARGET spdlog)
    find_package(spdlog REQUIRED)
endif()
target_link_libraries(gltest PRIVATE spdlog::spdlog)
target_link_libraries(gltest PRIVATE GLESv2 glut)
target_sources(gltest PRIVATE main.cpp)
