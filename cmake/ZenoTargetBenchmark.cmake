set(GOOGLETEST_PATH "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/googletest")
add_subdirectory(3rdparty/benchmark)

zeno_glob_recurse(source benchmark *.h *.cpp)
add_executable(zeno ${source})

target_link_libraries(zeno PRIVATE benchmark::benchmark_main)
