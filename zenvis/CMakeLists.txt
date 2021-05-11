file(GLOB PROJECT_SOURCE "*.cpp" "*.hpp" "*.h")

add_library(zenvis SHARED)
target_include_directories(zenvis PRIVATE ../include)

target_sources(zenvis PRIVATE ${PROJECT_SOURCE})
target_include_directories(zenvis PRIVATE .)
target_include_directories(zenvis PRIVATE ../zen/include)
target_link_libraries(zenvis PRIVATE OpenGL glad)

include(../scripts/Pybind11.cmake)
do_python_bindings()
