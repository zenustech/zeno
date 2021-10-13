add_executable(zs_tests)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
target_sources(zs_tests PRIVATE ${source})

target_link_libraries(zs_tests PRIVATE zs)

target_link_libraries(zs_tests PRIVATE gtest_main)
