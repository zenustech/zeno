cmake_minimum_required(VERSION 3.19)
set(CMAKE_CXX_STANDARD 17)
project(NBodyLib)
add_definitions(-D__TBB_CPP20_COMPARISONS_PRESENT=0)  
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

find_package(TBB CONFIG REQUIRED COMPONENTS tbb tbbmalloc)

file(GLOB NBODY_SOURCE *.cpp *.h)

add_library(NBodyLib SHARED ${NBODY_SOURCE})

if (WIN32)
target_compile_options(NBodyLib PRIVATE "/arch:AVX")
else()
target_compile_options(NBodyLib PRIVATE "-march=native")
endif()

target_compile_options(NBodyLib PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -fopenmp>)
target_link_libraries(NBodyLib PRIVATE openvdb Half IlmImf tbb)
target_compile_definitions(NBodyLib PRIVATE FMT_HEADER_ONLY)
find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
    set_target_properties(OpenMP::OpenMP_CXX PROPERTIES IMPORTED_GLOBAL TRUE)
    target_link_libraries(NBodyLib PRIVATE OpenMP::OpenMP_CXX)
endif()


target_include_directories(NBodyLib PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(NBodyLib PRIVATE TBB::tbb TBB::tbbmalloc)
target_link_libraries(NBodyLib PRIVATE pthread)

