file(GLOB ZENVIS_SOURCE "*.cpp" "*.hpp" "*.h")
pybind11_add_module(zenvis_pybind11_module ${ZENVIS_SOURCE})
target_compile_definitions(zenvis_pybind11_module PRIVATE -DZENO_APIFREE)
target_include_directories(zenvis_pybind11_module PRIVATE .)
target_include_directories(zenvis_pybind11_module PRIVATE ${PROJECT_SOURCE_DIR}/zeno/include)
target_link_libraries(zenvis_pybind11_module PRIVATE glad)

set(OUTPUT_DIR ${PROJECT_SOURCE_DIR}/zenqt)
set_target_properties(zenvis_pybind11_module PROPERTIES
	LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_DIR}
	RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR}
	ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_DIR}
	LIBRARY_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR}
	RUNTIME_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR}
	ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR}
	LIBRARY_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR}
	RUNTIME_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR}
	ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR}
	)
