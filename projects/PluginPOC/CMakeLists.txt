project(PluginPOC)

cmake_minimum_required(VERSION 3.16)

## plugin loader
find_package(Boost REQUIRED COMPONENTS system filesystem)
target_sources(zeno PRIVATE plugin.cpp)
target_link_libraries(zeno PRIVATE Boost::system Boost::filesystem)

## plugin helper
add_library(zeno_plugin_helper INTERFACE)

get_target_property(ZENO_INC_DIR zeno INCLUDE_DIRECTORIES)
message("demo plugin includes [${ZENO_INC_DIR}]")
target_include_directories(zeno_plugin_helper INTERFACE ${ZENO_INC_DIR})

get_target_property(ZENO_LINK_OPTIONS zeno LINK_OPTIONS)
message("demo plugin link options [${ZENO_LINK_OPTIONS}]")
target_link_options(zeno_plugin_helper INTERFACE ${ZENO_LINK_OPTIONS})

get_target_property(ZENO_LINK_DIRS zeno LINK_DIRECTORIES)
message("demo plugin link dirs [${ZENO_LINK_DIRS}]")
target_link_directories(zeno_plugin_helper INTERFACE ${ZENO_LINK_DIRS})

find_package(OpenMP)
get_target_property(ZENO_LINK_LIBS zeno LINK_LIBRARIES)
message("demo plugin link libs [${ZENO_LINK_LIBS}]")
target_link_libraries(zeno_plugin_helper INTERFACE ${ZENO_LINK_LIBS})

get_target_property(ZENO_COMPILE_DEFS zeno COMPILE_DEFINITIONS)
message("demo plugin compile defs [${ZENO_COMPILE_DEFS}]")
target_compile_definitions(zeno_plugin_helper INTERFACE ${ZENO_COMPILE_DEFS})

# get_target_property(ZENO_COMPILE_FEATURES zeno COMPILE_FEATURES)
# message("demo plugin compile features [${ZENO_COMPILE_FEATURES}]")
# target_compile_features(zeno_plugin_helper INTERFACE ${ZENO_COMPILE_FEATURES})


## plugins
add_library(zeno_plugin_demo SHARED demo_node.cpp)
target_link_libraries(zeno_plugin_demo PRIVATE zeno_plugin_helper)
add_dependencies(zenoedit zeno_plugin_demo)
