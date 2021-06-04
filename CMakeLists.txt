cmake_minimum_required(VERSION 3.12)
project(zeno LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(common)

add_subdirectory(external/pybind11)


add_library(zen SHARED zensession.cpp include/zen/zen.h)
target_include_directories(zen PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(zen PUBLIC include)
target_library_output_dir(zen ${PROJECT_SOURCE_DIR}/zen/usr/lib)

pybind11_add_module(libzenpy libzenpy.cpp)
target_link_libraries(libzenpy PRIVATE zen)
target_library_output_dir(libzenpy ${PROJECT_SOURCE_DIR}/zen)

add_custom_command(TARGET zen POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/share ${PROJECT_SOURCE_DIR}/python/zen/usr/share
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/python/zen/usr/include
    )
