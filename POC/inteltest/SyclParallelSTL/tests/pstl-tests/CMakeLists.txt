find_package(Threads)

function(compile_test source)
    set(test_name "pstl.${source}")
    set(source "${source}.cpp")
    add_executable(${test_name} ${source})
    target_link_libraries(${test_name} PUBLIC "${gtest_BINARY_DIR}/libgtest.a"
                                       PUBLIC "${gtest_BINARY_DIR}/libgtest_main.a"
                                       PUBLIC "${CMAKE_THREAD_LIBS_INIT}")
    add_dependencies(${test_name} gtest_main)
    add_dependencies(${test_name} gtest)
    add_sycl_to_target(${test_name} ${CMAKE_CURRENT_BINARY_DIR}
                                    ${CMAKE_CURRENT_SOURCE_DIR}/${source})
    add_test(test.${test_name} ${test_name})
endfunction()

file(GLOB files "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
foreach (file ${files})
    get_filename_component(file ${file} NAME_WE)
    compile_test(${file})
endforeach()
