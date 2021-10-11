add_library(zs.modules.primitive OBJECT)
target_link_libraries(zs.modules.primitive PUBLIC glm)
target_link_libraries(zs.modules.primitive PUBLIC zs.legacy)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
target_sources(zs.modules.primitive PRIVATE ${source})
