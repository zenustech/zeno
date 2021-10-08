add_executable(zeno.main ${BACKWARD_ENABLE})
add_backward(zeno.main)
target_link_libraries(zeno.main PRIVATE zeno.core zeno.modules zeno.editor)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
target_sources(zeno.editor PRIVATE ${source})
