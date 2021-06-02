file(GLOB PROJECT_SOURCE "*.cpp" "*.hpp" "*.h")

add_library(zenvis SHARED)
target_include_directories(zenvis PRIVATE ../include)

target_sources(zenvis PRIVATE ${PROJECT_SOURCE})
target_include_directories(zenvis PRIVATE .)
target_include_directories(zenvis PRIVATE ${PROJECT_SOURCE_DIR}/zen/include)
target_include_directories(zenvis PRIVATE ${PROJECT_SOURCE_DIR}/zenbase/include)
target_link_libraries(zenvis PRIVATE glad)
set_property(TARGET zenvis PROPERTY LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python/zenvis)

include(${PROJECT_SOURCE_DIR}/cmake/ZenPybind11.cmake)
do_python_bindings()
