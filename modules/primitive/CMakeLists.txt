add_library(zeno.modules.primitive OBJECT)
target_link_libraries(zeno.modules.primitive PRIVATE zeno.core)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
target_sources(zeno.modules.primitive PRIVATE ${source})
