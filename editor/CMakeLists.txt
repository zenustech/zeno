add_library(zeno.editor OBJECT)
target_include_directories(zeno.editor PUBLIC .)
target_link_libraries(zeno.editor PUBLIC glad glfw ftgl glm)
target_link_libraries(zeno.editor PUBLIC tinyfiledialogs rapidjson)
target_link_libraries(zeno.editor PUBLIC zeno.core zeno.assets)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
target_sources(zeno.editor PRIVATE ${source})
