cmake_minimum_required(VERSION 3.16)

if (NOT ZENO_WITH_CUDA)
    message(FATAL_ERROR "ZENO_WITH_CUDAmust be ON when CUDA is ON! "
        "Please specify: -ZENO_WITH_CUDA:BOOL=ON")
endif()

set(CMAKE_CXX_STANDARD 17)
find_package(OpenVDB)

set(TEST_OPENVDB_SRC nosys.cpp)

target_sources(zeno PRIVATE ${TEST_OPENVDB_SRC})

target_link_libraries(zeno PRIVATE OpenVDB::openvdb)
target_include_directories(zeno PRIVATE ../zenvdb/include)
target_link_libraries(zeno PRIVATE zshelper)