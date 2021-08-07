file(GLOB PROJECT_SOURCE include/*/*.h *.cpp)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /usr/local/lib/cmake/OpenVDB/)

add_library(zeno_zenvdb ${PROJECT_SOURCE})
target_include_directories(zeno_zenvdb PUBLIC include)
target_include_directories(zeno_zenvdb PRIVATE ../oldzenbase/include)

find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
    message("found package: OpenMP::OpenMP_CXX")
    target_link_libraries(zeno_zenvdb PRIVATE OpenMP::OpenMP_CXX)
endif()

find_package(TBB CONFIG REQUIRED COMPONENTS tbb tbbmalloc)
find_package(IlmBase REQUIRED COMPONENTS Half REQUIRED)
find_package(OpenVDB REQUIRED)
target_link_libraries(zeno_zenvdb PRIVATE OpenVDB::openvdb IlmBase::Half TBB::tbbmalloc TBB::tbb)

target_link_libraries(zeno_zenvdb PUBLIC zeno)
