cmake_minimum_required(VERSION 3.12)
project(zeno LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(external/pybind11)


add_library(zensession SHARED zensession.cpp include/zen/zen.h)
target_include_directories(zensession PUBLIC include)
target_link_libraries(zensession PUBLIC dl)
set_property(TARGET zensession PROPERTY LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/zen/usr/lib)

pybind11_add_module(libzenpy libzenpy.cpp)
target_link_libraries(libzenpy PRIVATE zensession)
set_property(TARGET libzenpy PROPERTY LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/zen)

add_custom_command(TARGET libzenpy POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/zen/autoload
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/share ${CMAKE_CURRENT_SOURCE_DIR}/zen/usr/share
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/zen/usr/include
    )
