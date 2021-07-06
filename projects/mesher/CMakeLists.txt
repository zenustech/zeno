cmake_minimum_required(VERSION 3.19)
set(CMAKE_CXX_STANDARD 17)
project(zenMesher)
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
add_library(zenMesher SHARED ${ZEN_MESHER_SOURCE})
target_include_directories(zenMesher PRIVATE mpmmeshing/)


target_link_libraries(zenMesher PRIVATE mpmmeshing )

find_package(zeno REQUIRED)
target_link_libraries(zenMesher PRIVATE zeno Eigen3::Eigen)

set_target_properties(zenMesher PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${zeno_AUTOLOAD_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${zeno_AUTOLOAD_DIR}
    ARCHIVE_OUTPUT_DIRECTORY ${zeno_AUTOLOAD_DIR}
    LIBRARY_OUTPUT_DIRECTORY_DEBUG ${zeno_AUTOLOAD_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${zeno_AUTOLOAD_DIR}
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${zeno_AUTOLOAD_DIR}
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${zeno_AUTOLOAD_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${zeno_AUTOLOAD_DIR}
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${zeno_AUTOLOAD_DIR}
    )
