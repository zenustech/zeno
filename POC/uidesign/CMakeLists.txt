cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
project(ZenoEditor LANGUAGES C CXX)

add_subdirectory(ftgl)

file(GLOB PROJECT_SOURCE *.cpp)
add_executable(ZenoEditor ${PROJECT_SOURCE})
target_link_libraries(ZenoEditor PRIVATE GL glfw)
target_compile_options(ZenoEditor PRIVATE -lbfd -liberty)


target_link_libraries(ZenoEditor PRIVATE ftgl)
find_package(Freetype REQUIRED)
target_include_directories(ZenoEditor PRIVATE ftgl/src ftgl/src/FTFont ftgl/src/FTGlyph ftgl/src/FTLayout ${FREETYPE_INCLUDE_DIRS})
