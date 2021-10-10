add_library(zs.ztd OBJECT)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
target_sources(zs.ztd PRIVATE ${source})
