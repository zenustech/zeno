cmake_minimum_required(VERSION 3.19)
set(CMAKE_CXX_STANDARD 20)
project(FuckTBB)

find_package(TBB CONFIG REQUIRED COMPONENTS tbb tbbmalloc tbbmalloc_proxy)

add_library(Fucker SHARED Fucker.cpp)
add_executable(Fuckee Fuckee.cpp)

target_link_libraries(Fucker PRIVATE TBB::tbb TBB::tbbmalloc TBB::tbbmalloc_proxy)
target_link_libraries(Fuckee PRIVATE Fucker)
