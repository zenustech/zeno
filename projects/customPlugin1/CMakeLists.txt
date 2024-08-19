add_library(customPlugin1 SHARED plugindef.h pluginnode.cpp)

target_link_libraries(customPlugin1 PUBLIC zenocore)