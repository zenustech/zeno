cmake_minimum_required(VERSION 3.10)
project(ZenoEditor LANGUAGES C CXX)

file(GLOB PROJECT_SOURCE *.cpp)
add_executable(ZenoEditor ${PROJECT_SOURCE})
target_link_libraries(ZenoEditor PRIVATE GL glfw)
