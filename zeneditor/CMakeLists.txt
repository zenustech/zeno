cmake_minimum_required(VERSION 3.18)
project(zeneditor LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

file(GLOB PROJECT_SOURCE *.cpp *.h imgui/*.cpp imgui/*.h
    imnodes/*.cpp imnodes/*.h)

add_library(zeneditor SHARED)
target_sources(zeneditor PRIVATE ${PROJECT_SOURCE})
target_include_directories(zeneditor PRIVATE . imgui imnodes)
target_include_directories(zeneditor PRIVATE ../include)
target_link_libraries(zeneditor PRIVATE OpenGL GLEW glfw)

include(../../CMake/Pybind11.cmake)
do_python_bindings()
