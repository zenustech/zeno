file(GLOB TEST_SOURCE *.cpp)
add_executable(zentest ${TEST_SOURCE})
target_link_libraries(zentest PRIVATE zeno)
