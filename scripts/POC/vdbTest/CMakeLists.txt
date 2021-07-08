cmake_minimum_required(VERSION 3.19)
file(GLOB PROJECT_SOURCE include/*/*.h *.cpp)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /usr/local/lib/cmake/OpenVDB/)

add_executable(vdbTest  ${PROJECT_SOURCE})

find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
    message("found package: OpenMP::OpenMP_CXX")
    target_link_libraries(vdbTest PRIVATE OpenMP::OpenMP_CXX)
endif()

find_package(TBB CONFIG REQUIRED COMPONENTS tbb tbbmalloc)
find_package(IlmBase REQUIRED COMPONENTS Half REQUIRED)
find_package(OpenVDB REQUIRED)
target_link_libraries(vdbTest PRIVATE OpenVDB::openvdb IlmBase::Half TBB::tbbmalloc TBB::tbb)