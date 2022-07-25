file(GLOB_RECURSE source CONFIGURE_DEPENDS *.cpp *.h)

target_sources(zeno PRIVATE ${source})
target_include_directories(zeno PRIVATE .)
