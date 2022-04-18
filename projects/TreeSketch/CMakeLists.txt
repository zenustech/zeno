cmake_minimum_required(VERSION 3.10)

target_sources(zeno PUBLIC
    SketchTree.cpp
)

target_include_directories(zeno PRIVATE
    include
)

target_sources(zeno PRIVATE
    src/TreeSketchMath.cpp
    src/TreeSketchTreeObj.cpp
)
