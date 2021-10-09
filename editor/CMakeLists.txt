add_library(zeno.editor OBJECT)
target_include_directories(zeno.editor PUBLIC .)
target_link_libraries(zeno.editor PRIVATE glad glfw ftgl glm)
target_link_libraries(zeno.editor PRIVATE rapidjson incbin)
target_link_libraries(zeno.editor PUBLIC zeno.core)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
target_sources(zeno.editor PRIVATE ${source})
