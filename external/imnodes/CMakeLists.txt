project(imnodes LANGUAGES CXX)

file(GLOB source *.cpp)

add_library(imnodes SHARED ${source})
target_include_directories(imnodes PRIVATE .)
target_include_directories(imnodes PRIVATE ../imgui)
