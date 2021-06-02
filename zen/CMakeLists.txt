add_library(zen SHARED libzen.cpp include/zen/zen.h)
target_include_directories(zen PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(zen PUBLIC include)
set_property(TARGET zen PROPERTY LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python/zen/mods)

add_library(zenpy SHARED python/libzenpy.cpp)
target_link_libraries(zenpy PRIVATE zen)
set_property(TARGET zenpy PROPERTY LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python/zen)

include(${PROJECT_SOURCE_DIR}/cmake/ZenPybind11.cmake)
do_python_bindings()
