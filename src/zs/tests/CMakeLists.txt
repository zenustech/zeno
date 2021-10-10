add_executable(zs.tests)

target_link_libraries(zs.tests PRIVATE zs.zeno)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
target_sources(zs.tests PRIVATE ${source})

enable_testing()
target_link_libraries(zs.tests PRIVATE gtest_main)
include(GoogleTest)
gtest_discover_tests(zs.tests)
