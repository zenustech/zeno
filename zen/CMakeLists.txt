set(CMAKE_CXX_STANDARD 17)
file(GLOB PROJECT_SOURCE *.cpp include/*/*.h view/*.cpp base/*.cpp)

add_library(zen SHARED)
add_definitions(-D__TBB_CPP20_COMPARISONS_PRESENT=0)  
target_include_directories(zen PRIVATE ../external)
target_link_directories(zen PRIVATE ../external)

target_sources(zen PRIVATE ${PROJECT_SOURCE})
target_include_directories(zen PUBLIC include)

target_link_libraries(zen PRIVATE openvdb Half tbb)
target_compile_definitions(zen PRIVATE FMT_HEADER_ONLY)
find_package(OpenMP)
target_compile_options(zen PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -fopenmp>)
if (TARGET OpenMP::OpenMP_CXX)
    set_target_properties(OpenMP::OpenMP_CXX PROPERTIES IMPORTED_GLOBAL TRUE)
    target_link_libraries(zen PRIVATE OpenMP::OpenMP_CXX)
endif()


add_library(zenpy SHARED python/libzenpy.cpp)
target_include_directories(zenpy PRIVATE include)
target_link_libraries(zenpy PRIVATE zen)

include(../scripts/Pybind11.cmake)
do_python_bindings()
