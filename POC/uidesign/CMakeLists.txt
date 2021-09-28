cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
project(ZenoEditor LANGUAGES C CXX)

add_subdirectory(glad)
add_subdirectory(glfw)
add_subdirectory(ftgl)
add_subdirectory(backward-cpp)

file(GLOB PROJECT_SOURCE *.cpp)
add_executable(ZenoEditor ${PROJECT_SOURCE} ${BACKWARD_ENABLE})
add_backward(ZenoEditor)

target_link_libraries(ZenoEditor PRIVATE GL glfw ftgl)
target_compile_options(ZenoEditor PRIVATE -lbfd -liberty)
