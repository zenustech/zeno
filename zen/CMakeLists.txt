add_library(zen SHARED libzen.cpp include/zen/zen.h)
target_include_directories(zen PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(zen PUBLIC include)
target_library_output_dir(zen ${PROJECT_SOURCE_DIR}/python/zen)

pybind11_add_module(libzenpy python/libzenpy.cpp)
target_link_libraries(libzenpy PRIVATE zen)
target_library_output_dir(libzenpy ${PROJECT_SOURCE_DIR}/python/zen)
