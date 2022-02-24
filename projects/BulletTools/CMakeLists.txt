add_library(zeno_BulletTools OBJECT stub.cpp)
target_link_libraries(zeno_BulletTools PRIVATE zeno)
target_link_libraries(zeno_BulletTools PRIVATE zeno_nodep)
