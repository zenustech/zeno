add_executable(zs_tests)

target_link_libraries(zs_tests PRIVATE zs)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
target_sources(zs_tests PRIVATE ${source})

enable_testing()
target_link_libraries(zs_tests PRIVATE gtest_main)
include(GoogleTest)
gtest_discover_tests(zs_tests)
