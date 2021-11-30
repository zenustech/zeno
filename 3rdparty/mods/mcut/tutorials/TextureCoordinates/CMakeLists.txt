add_executable(TextureCoordinates TextureCoordinates.cpp)
target_link_libraries(TextureCoordinates mcut ${extra_libs})
target_include_directories(TextureCoordinates PRIVATE ${MCUT_INCLUDE_DIR} ${libigl_include_dir} ${eigen_include_dir})#
target_compile_definitions(TextureCoordinates PRIVATE -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data" -DOUTPUT_DIR="${CMAKE_CURRENT_BINARY_DIR}" ${preprocessor_defs})
target_compile_options(TextureCoordinates PRIVATE ${compilation_flags})