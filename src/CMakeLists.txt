add_library(zeno OBJECT)
target_include_directories(zeno PUBLIC .)
target_include_directories(zeno PUBLIC zeno/legacy)

target_link_libraries(zeno PUBLIC spdlog)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
target_sources(zeno PRIVATE ${source})
