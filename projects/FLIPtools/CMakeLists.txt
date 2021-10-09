add_library(zeno_FLIPtools SHARED stub.cpp)
target_link_libraries(zeno_FLIPtools PRIVATE zeno)
target_link_libraries(zeno_FLIPtools PRIVATE zeno_nodep)
