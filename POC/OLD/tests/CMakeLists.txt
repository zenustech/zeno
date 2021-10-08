file(GLOB TEST_SOURCE *.cpp)
add_executable(zeno_tests ${TEST_SOURCE})
target_link_libraries(zeno_tests PRIVATE zeno)
