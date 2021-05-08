set(CMAKE_CXX_STANDARD 17)
file(GLOB PROJECT_SOURCE include/*/*.h *.cpp view/*.cpp base/*/*.cpp)

add_library(zen SHARED)
add_definitions(-D__TBB_CPP20_COMPARISONS_PRESENT=0)  
target_include_directories(zen PRIVATE ../external)
target_link_directories(zen PRIVATE ../external)

target_sources(zen PRIVATE ${PROJECT_SOURCE})
target_include_directories(zen PUBLIC include)

target_link_libraries(zen PRIVATE openvdb tbb)
target_compile_definitions(zen PRIVATE FMT_HEADER_ONLY)
find_package(OpenMP)
target_compile_options(zen PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -fopenmp>)
if (TARGET OpenMP::OpenMP_CXX)
    get_target_property(OPENMP_IMPORT_STATUS OpenMP::OpenMP_CXX IMPORTED_GLOBAL)
    if (NOT OPENMP_IMPORT_STATUS)
        set_target_properties(OpenMP::OpenMP_CXX PROPERTIES IMPORTED_GLOBAL TRUE)
    endif()
    target_link_libraries(zen PRIVATE OpenMP::OpenMP_CXX)
endif()

add_library(zenpy SHARED python/libzenpy.cpp)
target_link_libraries(zenpy PRIVATE zen)

include(../scripts/Pybind11.cmake)
do_python_bindings()
