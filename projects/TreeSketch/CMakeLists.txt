cmake_minimum_required(VERSION 3.10)

add_library(zeno_TreeSketch SHARED
    SketchTree.cpp
)

target_link_libraries(zeno_TreeSketch PUBLIC
    zeno
)