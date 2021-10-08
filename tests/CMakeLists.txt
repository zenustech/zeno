add_executable(zeno.tests)
target_sources(zeno.tests PRIVATE ${BACKWARD_ENABLE})
add_backward(zeno.tests)

target_link_libraries(zeno.tests PRIVATE zeno.core zeno.modules)
