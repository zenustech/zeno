add_executable(zeno_tests)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
target_sources(zeno_tests PRIVATE ${source})

target_link_libraries(zeno_tests PRIVATE zeno)
target_link_libraries(zeno_tests PRIVATE gtest_main)

include(GoogleTest)
gtest_discover_tests(zeno_tests)
