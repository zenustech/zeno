

add_executable(VertexNormals VertexNormals.cpp)
target_link_libraries(VertexNormals mcut ${extra_libs})
target_include_directories(VertexNormals PRIVATE ${MCUT_INCLUDE_DIR} ${libigl_include_dir} ${eigen_include_dir})#
target_compile_definitions(VertexNormals PRIVATE -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data" -DOUTPUT_DIR="${CMAKE_CURRENT_BINARY_DIR}" ${preprocessor_defs})
target_compile_options(VertexNormals PRIVATE ${compilation_flags})