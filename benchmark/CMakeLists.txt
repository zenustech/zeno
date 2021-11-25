set(GOOGLETEST_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../tests/googletest")
add_subdirectory(benchmark)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
add_executable(zeno_benchmark ${source})

target_link_libraries(zeno_benchmark PRIVATE benchmark::benchmark_main)
