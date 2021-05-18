file(GLOB PROJECT_SOURCE include/*/*.h *.cpp)

add_library(zenvdb SHARED ${PROJECT_SOURCE})
target_include_directories(zenvdb PRIVATE ../include)
target_include_directories(zenvdb PUBLIC include)
target_link_libraries(zenvdb PRIVATE zen)

find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
    message("found package: OpenMP::OpenMP_CXX")
    target_link_libraries(zenvdb PRIVATE OpenMP::OpenMP_CXX)
endif()

find_package(TBB CONFIG REQUIRED COMPONENTS tbb tbbmalloc)
find_package(IlmBase REQUIRED COMPONENTS Half REQUIRED)
target_link_libraries(zenvdb PRIVATE openvdb IlmBase::Half TBB::tbbmalloc TBB::tbb)
