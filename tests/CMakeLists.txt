add_executable(zeno.tests ${BACKWARD_ENABLE})
add_backward(zeno.tests)
target_link_libraries(zeno.tests PRIVATE zeno.core)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
target_sources(zeno.tests PRIVATE ${source})

enable_testing()
target_link_libraries(zeno.tests PRIVATE gtest_main)
include(GoogleTest)
gtest_discover_tests(zeno.tests)
