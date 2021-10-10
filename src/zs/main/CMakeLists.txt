add_executable(zs.main)
target_sources(zs.main PRIVATE ${BACKWARD_ENABLE})
add_backward(zs.main)

target_link_libraries(zs.main PRIVATE zs.zeno zs.editor)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
target_sources(zs.main PRIVATE ${source})
