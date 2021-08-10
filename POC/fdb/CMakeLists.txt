set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

add_executable(adflip mgflip.cpp timer.cpp)

find_package(OpenMP REQUIRED)
target_link_libraries(adflip OpenMP::OpenMP_CXX)

if (GNU)
    set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../projects/cmake ${CMAKE_MODULE_PATH})
endif()

find_package(TBB REQUIRED)
find_package(OpenVDB REQUIRED)
target_link_libraries(adflip OpenVDB::openvdb TBB::tbb)
