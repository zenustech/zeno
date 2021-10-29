add_executable(zeno)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
target_sources(zeno PRIVATE ${source})

target_link_libraries(zeno PRIVATE gtest_main)

include(GoogleTest)
gtest_discover_tests(zeno)
