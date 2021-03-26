cmake_minimum_required(VERSION 3.18)
project(zenbase LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

file(GLOB PROJECT_SOURCE *.cpp *.h)
add_library(zenbase SHARED)
target_sources(zenbase PRIVATE ${PROJECT_SOURCE})
target_include_directories(zenbase PUBLIC include)
target_include_directories(zenbase PRIVATE ../include)
target_compile_options(zenbase PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -O3 -fopenmp>)
target_link_libraries(zenbase PRIVATE openvdb Half IlmImf tbb)
target_compile_definitions(zenbase
    PRIVATE   FMT_HEADER_ONLY
)

find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
    set_target_properties(OpenMP::OpenMP_CXX PROPERTIES IMPORTED_GLOBAL TRUE)
    target_link_libraries(zenbase PRIVATE OpenMP::OpenMP_CXX)
endif()

target_include_directories(zenbase PRIVATE ../zen/include)
target_link_directories(zenbase PRIVATE ../zen/build)
target_link_libraries(zenbase PRIVATE zen)
