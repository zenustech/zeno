cmake_minimum_required(VERSION 3.10)
# GCC 11, Clang 12, MSVC 19.28
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Debug)
project(ZenoEditor LANGUAGES C CXX)

add_subdirectory(glad)
add_subdirectory(glfw)
add_subdirectory(ftgl)
add_subdirectory(backward-cpp)

file(GLOB PROJECT_SOURCE zui/*.cpp zui/*.h)
add_executable(ZenoEditor ${PROJECT_SOURCE} ${BACKWARD_ENABLE})
add_backward(ZenoEditor)

target_include_directories(ZenoEditor PRIVATE .)
target_link_libraries(ZenoEditor PRIVATE glad glfw ftgl)
