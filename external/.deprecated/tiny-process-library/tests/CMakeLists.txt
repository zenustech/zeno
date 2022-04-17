include_directories(..)

add_executable(io_test io_test.cpp)
target_link_libraries(io_test tiny-process-library)
add_test(io_test io_test)

add_executable(open_close_test open_close_test.cpp)
target_link_libraries(open_close_test tiny-process-library)
add_test(open_close_test open_close_test)

add_executable(multithread_test multithread_test.cpp)
target_link_libraries(multithread_test tiny-process-library)
add_test(multithread_test multithread_test)
