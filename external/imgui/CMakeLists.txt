project(imgui LANGUAGES CXX)

file(GLOB source *.cpp)

add_library(imgui SHARED ${source})
target_include_directories(imgui PRIVATE .)
