add_library(zs.legacy STATIC)
target_link_libraries(zs.legacy PUBLIC spdlog)
target_link_libraries(zs.legacy PUBLIC zs.ztd)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
target_sources(zs.legacy PRIVATE ${source})
