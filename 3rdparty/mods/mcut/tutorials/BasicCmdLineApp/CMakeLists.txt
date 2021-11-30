add_executable(BasicCmdLineApp BasicCmdLineApp.cpp)
message(STATUS "extra_libs = ${extra_libs}")
target_link_libraries(BasicCmdLineApp mcut ${extra_libs})
target_include_directories(BasicCmdLineApp PRIVATE ${MCUT_INCLUDE_DIR} ${libigl_include_dir} ${eigen_include_dir})#
target_compile_options(BasicCmdLineApp PRIVATE ${compilation_flags})
target_compile_definitions(BasicCmdLineApp PRIVATE ${preprocessor_defs})