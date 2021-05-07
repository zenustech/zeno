cmake_minimum_required(VERSION 3.19)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
project(NBody)
add_definitions(-D__TBB_CPP20_COMPARISONS_PRESENT=0)  
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

find_package(TBB CONFIG REQUIRED COMPONENTS tbb tbbmalloc)

file(GLOB NBODY_SOURCE *.cpp *.h)

#add_library(NBody SHARED ${NBODY_SOURCE})
add_executable(NBody ${NBODY_SOURCE})
target_compile_options(NBody PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -Wno-ignored-attributes>)

if (WIN32)
target_compile_options(NBody PRIVATE "/arch:AVX")
else()
target_compile_options(NBody PRIVATE "-march=native")
endif()

target_compile_options(NBody PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -fopenmp>)
target_link_libraries(NBody PRIVATE openvdb tbb)
target_compile_definitions(NBody PRIVATE FMT_HEADER_ONLY)
find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
    set_target_properties(OpenMP::OpenMP_CXX PROPERTIES IMPORTED_GLOBAL TRUE)
    target_link_libraries(NBody PRIVATE OpenMP::OpenMP_CXX)
endif()


target_include_directories(NBody PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(NBody PRIVATE TBB::tbb TBB::tbbmalloc)
target_link_libraries(NBody PRIVATE pthread)

target_link_libraries(FLIPlib PRIVATE zen)
