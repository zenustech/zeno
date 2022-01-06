file(GLOB ZENVIS_SOURCE "*.cpp" "*.hpp" "*.h")

pybind11_add_module(pylib_zenvis ${ZENVIS_SOURCE})

target_compile_definitions(pylib_zenvis PRIVATE -DZENO_APIFREE)
target_include_directories(pylib_zenvis PRIVATE .)
target_include_directories(pylib_zenvis PRIVATE ${PROJECT_SOURCE_DIR}/zeno)
target_link_libraries(pylib_zenvis PRIVATE glad)

ADD_LIBRARY(zenvis STATIC ${ZENVIS_SOURCE})
target_include_directories(zenvis PRIVATE .)
target_include_directories(zenvis PRIVATE ${PROJECT_SOURCE_DIR}/zeno)
target_link_libraries(zenvis glad)
target_compile_definitions(zenvis PRIVATE ZENVIS_LIB)