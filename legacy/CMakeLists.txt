target_include_directories(zeno_core PUBLIC .)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
target_sources(zeno_core PRIVATE ${source})
