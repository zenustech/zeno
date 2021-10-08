add_library(ZenoModule_primitive OBJECT)
target_link_libraries(ZenoModule_primitive PRIVATE ZenoCore)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
target_sources(ZenoModule_primitive PRIVATE ${source})
