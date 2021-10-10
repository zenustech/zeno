add_library(zeno.modules.primitive OBJECT)
target_link_libraries(zeno.modules.primitive PUBLIC glm)
target_link_libraries(zeno.modules.primitive PUBLIC zeno.legacy)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
target_sources(zeno.modules.primitive PRIVATE ${source})
