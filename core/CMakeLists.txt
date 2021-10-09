add_library(zeno.core OBJECT)
target_include_directories(zeno.core PUBLIC .)
target_link_libraries(zeno.core PUBLIC spdlog)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
target_sources(zeno.core PRIVATE ${source})
