file(GLOB TEST_SOURCE *.cpp)
add_executable(zeno_tests ${TEST_SOURCE})
target_link_libraries(zeno_tests PRIVATE zeno)

enable_testing()
add_test(zeno_tests zeno_tests)

# Add dll output dir to PATH
if (WIN32)
    set_tests_properties(zeno_tests PROPERTIES
        ENVIRONMENT "PATH=${PROJECT_SOURCE_DIR}/zenqt/lib\;$ENV{PATH}"
    )
endif()
