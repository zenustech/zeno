cmake_minimum_required(VERSION 3.10)
project(FLTKtest CXX)

add_executable(main main.cpp)
find_package(FLTK REQUIRED)
target_link_libraries(main ${FLTK_LIBRARIES})
