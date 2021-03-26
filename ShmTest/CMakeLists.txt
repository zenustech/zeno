cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 20)
project(shmtest)

file(GLOB PROJECT_SOURCE "*.cpp" "*.hpp" "*.h")

add_executable(shmtest)
target_sources(shmtest PRIVATE ${PROJECT_SOURCE})
target_include_directories(shmtest PRIVATE .)
target_include_directories(shmtest PRIVATE ../include)
target_link_libraries(shmtest PRIVATE dl)
