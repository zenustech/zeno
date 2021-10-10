add_library(zs.zeno STATIC)
target_link_libraries(zs.zeno PUBLIC spdlog)
target_link_libraries(zs.zeno PUBLIC zs.ztd)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
target_sources(zs.zeno PRIVATE ${source})
