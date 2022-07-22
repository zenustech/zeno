add_library(zeno_Audio SHARED Audio.cpp)

add_subdirectory(aquila)

target_link_libraries(zeno_Audio PRIVATE Aquila)
target_link_libraries(zeno_Audio PRIVATE zeno)
target_link_libraries(zeno_Audio PRIVATE zeno_nodep)
