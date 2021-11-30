file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp)
add_executable(zeno_benchmark ${source})

target_link_libraries(zeno_benchmark PRIVATE benchmark::benchmark_main)
