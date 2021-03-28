cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 20)
project(zenvis)

file(GLOB PROJECT_SOURCE "*.cpp" "*.hpp" "*.h")

add_executable(zenvis)
target_sources(zenvis PRIVATE ${PROJECT_SOURCE})
target_include_directories(zenvis PRIVATE .)
target_include_directories(zenvis PRIVATE ../include ../external)
target_link_libraries(zenvis PRIVATE OpenGL GLEW glfw pthread)
