target_include_directories(zeno PUBLIC .)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
target_sycl_sources(zeno PRIVATE ${source})
