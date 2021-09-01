cmake_minimum_required(VERSION 3.10)
project(zengledit LANGUAGES C CXX)

add_subdirectory(glad)

file(GLOB PROJECT_SOURCE *.cpp)
add_executable(zengledit ${PROJECT_SOURCE})
target_link_libraries(zengledit PRIVATE glad dl)
