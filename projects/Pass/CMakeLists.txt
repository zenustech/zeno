file(GLOB_RECURSE source CONFIGURE_DEPENDS *.cpp *.h)
file(GLOB_RECURSE glad_source CONFIGURE_DEPENDS glad/*.c glad/*.h)

target_sources(zeno PRIVATE ${source} ${glad_source})
target_include_directories(zeno PRIVATE . glad/include)
