cmake_minimum_required(VERSION 3.10)

add_subdirectory(LSystem)

target_sources(zeno PRIVATE
    zeno.cpp
)
find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
    message("found package: OpenMP::OpenMP_CXX")
    target_link_libraries(zeno PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(zeno PUBLIC LSystem)
