cmake_minimum_required(VERSION 3.10)

add_library(zeno_TreeSketch SHARED
    SketchTree.cpp
)

target_include_directories(zeno_TreeSketch PRIVATE
    include
)

target_sources(zeno_TreeSketch PRIVATE
    src/TreeSketchMath.cpp
    src/TreeSketchTreeObj.cpp
)

target_link_libraries(zeno_TreeSketch PUBLIC
    zeno
)
