add_library(zeno.editor OBJECT)
target_include_directories(zeno.editor PUBLIC .)
target_link_libraries(zeno.editor PRIVATE glad glfw ftgl glm incbin)
target_link_libraries(zeno.editor PUBLIC zeno.core)

file(GLOB_RECURSE source *.h *.cpp)
target_sources(zeno.editor PRIVATE ${source})
