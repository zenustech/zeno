cmake_minimum_required(VERSION 3.19)
set(CMAKE_CXX_STANDARD 17)
project(zeno_mesher)
add_definitions(-D__TBB_CPP20_COMPARISONS_PRESENT=0)  

# add cmake file
include (common)


find_package(Eigen3 REQUIRED)


add_subdirectory(mpmmeshing)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

file (GLOB ZEN_MESHER_SOURCE *.cpp)
add_library(zeno_mesher SHARED ${ZEN_MESHER_SOURCE})
target_include_directories(zeno_mesher PRIVATE mpmmeshing/)


target_link_libraries(zeno_mesher PRIVATE mpmmeshing )
target_link_libraries(zeno_mesher PRIVATE Eigen3::Eigen)
target_link_libraries(zeno_mesher PUBLIC zeno)
target_link_libraries(zeno_mesher PRIVATE zeno_nodep)
