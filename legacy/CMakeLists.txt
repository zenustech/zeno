add_library(zeno.legacy OBJECT)
target_include_directories(zeno.legacy PUBLIC .)
target_link_libraries(zeno.legacy PUBLIC zeno.core)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
target_sources(zeno.legacy PRIVATE ${source})
