target_link_libraries(zeno PUBLIC spdlog)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
target_sources(zeno PRIVATE ${source})
