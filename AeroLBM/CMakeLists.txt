cmake_minimum_required(VERSION 3.16)
project(cudatest LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

#find_package(OpenMP)

file(GLOB PROJECT_SOURCE *.cpp *.h *.hpp)

add_executable(simdtest)
target_sources(simdtest PRIVATE ${PROJECT_SOURCE})
target_include_directories(simdtest PRIVATE .)
target_include_directories(simdtest PRIVATE ../external)
#target_compile_options(simdtest PRIVATE -march=native -O3)
#target_link_libraries(simdtest PRIVATE OpenMP::OpenMP_CXX)
