file(GLOB_RECURSE source CONFIGURE_DEPENDS src/*.h src/*.cpp)
target_include_directories(zeno_core PUBLIC src)
target_sources(zeno_core PRIVATE ${source})
