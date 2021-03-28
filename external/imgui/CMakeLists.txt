project(imgui LANGUAGES CXX)

file(GLOB source *.cpp)

add_library(imgui ${source})
target_include_directories(imgui PUBLIC .)
