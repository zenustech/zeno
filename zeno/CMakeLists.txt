add_library(zen SHARED libzen.cpp include/zen/zen.h)
target_include_directories(zen PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(zen PUBLIC include)
target_library_output_dir(zen ${PROJECT_SOURCE_DIR}/python/zen/usr/lib)

pybind11_add_module(libzenpy python/libzenpy.cpp)
target_link_libraries(libzenpy PRIVATE zen)
target_library_output_dir(libzenpy ${PROJECT_SOURCE_DIR}/python/zen)

add_custom_command(TARGET zen POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/share ${PROJECT_SOURCE_DIR}/python/zen/usr/share
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/python/zen/usr/include
    )
