add_library(zeno_BulletTools SHARED stub.cpp)
target_link_libraries(zeno_BulletTools PRIVATE zeno)
target_link_libraries(zeno_BulletTools PRIVATE zeno_nodep)
