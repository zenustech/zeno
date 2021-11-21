cmake_minimum_required(VERSION 3.10)
project(nanatest CXX)

add_subdirectory(nana)

add_executable(main main.cpp)
target_include_directories(main PUBLIC nana/include)
target_link_libraries(main PUBLIC nana)
