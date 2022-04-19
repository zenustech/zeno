cmake_minimum_required(VERSION 3.10)

target_sources(zeno PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/SketchTree.cpp
)

target_include_directories(zeno PRIVATE
    include
)

target_sources(zeno PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/TreeSketchMath.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/TreeSketchTreeObj.cpp
)
