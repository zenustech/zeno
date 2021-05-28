if(ZS_ENABLE_OPENVDB)
    file(GLOB PROJECT_SOURCE include/*/*.h *.cpp view/*.cpp mesh/*.cpp pars/*.cpp prim/*.cpp vdb/*.cpp) 
else()
    file(GLOB PROJECT_SOURCE include/*/*.h *.cpp view/*.cpp mesh/*.cpp pars/*.cpp prim/*.cpp) 
endif()

add_library(zen SHARED ${PROJECT_SOURCE})
target_include_directories(zen PRIVATE ../include)
target_include_directories(zen PUBLIC include)

find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
    message("found package: OpenMP::OpenMP_CXX")
    target_link_libraries(zen PRIVATE OpenMP::OpenMP_CXX)
endif()

if (ZS_ENABLE_OPENVDB)
    set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/zen/cmake)
    find_package(TBB CONFIG REQUIRED COMPONENTS tbb tbbmalloc)
    find_package(IlmBase REQUIRED COMPONENTS Half REQUIRED)
    target_link_libraries(zen PRIVATE openvdb IlmBase::Half TBB::tbbmalloc TBB::tbb)
endif()

add_library(zenpy SHARED python/libzenpy.cpp)
target_link_libraries(zenpy PRIVATE zen)

include(../scripts/Pybind11.cmake)
do_python_bindings()
