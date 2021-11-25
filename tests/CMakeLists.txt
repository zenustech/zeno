add_subdirectory(googletest)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
add_executable(zeno_tests ${source})

target_link_libraries(zeno_tests PRIVATE gtest_main)

include(GoogleTest)
gtest_discover_tests(zeno_tests)
