cmake_minimum_required(VERSION 3.19)
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
target_sources(zeno PRIVATE ${ZEN_MESHER_SOURCE})
target_include_directories(zeno PRIVATE mpmmeshing/)


target_link_libraries(zeno PRIVATE mpmmeshing)
target_link_libraries(zeno PRIVATE Eigen3::Eigen)
