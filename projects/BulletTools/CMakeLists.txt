add_library(zeno_BulletTools SHARED stub.cpp IslandVoronoi.cpp)
target_link_libraries(zeno_BulletTools PRIVATE zeno)
target_link_libraries(zeno_BulletTools PRIVATE zeno_nodep)
