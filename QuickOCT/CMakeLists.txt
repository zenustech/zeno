cmake_minimum_required(VERSION 3.19)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
project(OCTlib)

file(GLOB OCTLIB_SOURCE *.cpp *.h)

add_library(OCTlib SHARED ${OCTLIB_SOURCE})
target_compile_options(OCTlib PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -march=native -Wno-ignored-attributes>)
target_compile_definitions(OCTlib PRIVATE FMT_HEADER_ONLY)
target_include_directories(OCTlib PRIVATE .)

find_package(OpenMP REQUIRED)
target_link_libraries(OCTlib PRIVATE OpenMP::OpenMP_CXX)

find_package(TBB CONFIG REQUIRED COMPONENTS tbb tbbmalloc)
target_link_libraries(OCTlib PRIVATE TBB::tbb TBB::tbbmalloc)

target_link_libraries(OCTlib PRIVATE zen)
