file(GLOB TEST_SOURCE *.cpp)
add_executable(zeno_tests ${TEST_SOURCE})
target_link_libraries(zeno_tests PRIVATE zeno)

enable_testing()
add_test(zeno_tests zeno_tests)
