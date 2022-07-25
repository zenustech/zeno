file(GLOB_RECURSE source CONFIGURE_DEPENDS *.cpp *.h)

target_sources(zeno PRIVATE ${source})
