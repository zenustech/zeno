project(imnodes LANGUAGES CXX)

file(GLOB source *.cpp)

add_library(imnodes ${source})
target_include_directories(imnodes PUBLIC .)
target_include_directories(imnodes PUBLIC ../imgui)
