

add_executable(AdjacentFaces AdjacentFaces.cpp)

target_link_libraries(AdjacentFaces mcut )
target_include_directories(AdjacentFaces PRIVATE ${MCUT_INCLUDE_DIR}  ${libigl_include_dir} ${eigen_include_dir})
target_compile_definitions(AdjacentFaces PRIVATE -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data" -DOUTPUT_DIR="${CMAKE_CURRENT_BINARY_DIR}")
target_compile_options(AdjacentFaces PRIVATE ${compilation_flags})
target_compile_definitions(AdjacentFaces PRIVATE ${preprocessor_defs})