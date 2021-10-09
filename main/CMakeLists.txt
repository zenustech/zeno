add_executable(zeno.main)
target_sources(zeno.main PRIVATE ${BACKWARD_ENABLE})
add_backward(zeno.main)

target_link_libraries(zeno.main PRIVATE zeno.core zeno.editor)
#target_link_libraries(zeno.main PRIVATE zeno.legacy zeno.modules)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
target_sources(zeno.editor PRIVATE ${source})
