cmake_minimum_required(VERSION 3.18)
project(zeneditor LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

file(GLOB PROJECT_SOURCE *.cpp *.h)

add_library(zeneditor SHARED)
target_sources(zeneditor PRIVATE ${PROJECT_SOURCE})
target_include_directories(zeneditor PRIVATE .)
target_include_directories(zeneditor PRIVATE ../include ../external)
target_link_libraries(zeneditor PRIVATE OpenGL GLEW glfw)
target_link_libraries(zeneditor PRIVATE imgui imnodes)

include(../cmake/Pybind11.cmake)
do_python_bindings()
