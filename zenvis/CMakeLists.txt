file(GLOB PROJECT_SOURCE "*.cpp" "*.hpp" "*.h")

pybind11_add_module(libzenvis ${PROJECT_SOURCE})
target_include_directories(libzenvis PRIVATE .)
target_include_directories(libzenvis PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(libzenvis PRIVATE ${PROJECT_SOURCE_DIR}/zen/include)
target_include_directories(libzenvis PRIVATE ${PROJECT_SOURCE_DIR}/zenbase/include)
target_link_libraries(libzenvis PRIVATE glad)
#set_property(TARGET libzenvis PROPERTY LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python/zenvis)
