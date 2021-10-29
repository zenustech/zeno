add_library(zeno_legacy OBJECT)
target_include_directories(zeno_legacy PUBLIC .)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
target_sources(zeno_legacy PRIVATE ${source})
