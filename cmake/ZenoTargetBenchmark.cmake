add_subdirectory(depends/CppBenchmark)

zeno_glob_recurse(source benchmark *.h *.cpp)
add_executable(zeno ${source})

target_link_libraries(zeno PRIVATE cppbenchmark)
