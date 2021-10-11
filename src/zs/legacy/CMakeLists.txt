target_link_libraries(zs PUBLIC spdlog)
target_link_libraries(zs PUBLIC zs.ztd)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
target_sources(zs PRIVATE ${source})
