file(GLOB_RECURSE source RELATIVE tests CONFIGURE_DEPENDS *.h *.cpp)

add_executable(zeno ${source})

target_link_libraries(zeno PRIVATE gtest_main)

include(GoogleTest)
gtest_discover_tests(zeno)
