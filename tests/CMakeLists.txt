file(GLOB TEST_SOURCE *.cpp)
add_executable(zentest ${TEST_SOURCE})
enable_testing()
add_test(NAME zentest COMMAND zentest)
