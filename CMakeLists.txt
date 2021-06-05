cmake_minimum_required(VERSION 3.12)
project(zeno LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(external/pybind11)


add_library(zensession SHARED zensession.cpp include/zen/zen.h)
target_include_directories(zensession PRIVATE include)
set(OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/zen/usr/lib)
set_target_properties(zensession PROPERTIES
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

pybind11_add_module(libzenpy libzenpy.cpp)
target_include_directories(libzenpy PRIVATE include)
target_link_libraries(libzenpy PUBLIC zensession)
set_property(TARGET libzenpy PROPERTY LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/zen)

add_custom_command(TARGET libzenpy PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/zen/usr
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/zen/usr/lib
    )

add_custom_command(TARGET libzenpy POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_SOURCE_DIR}/zen/autoload
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/zen/autoload
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/README.md ${CMAKE_CURRENT_SOURCE_DIR}/zen/autoload/README.md
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/share ${CMAKE_CURRENT_SOURCE_DIR}/zen/usr/share
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/zen/usr/include
    )

#add_library(demo_project SHARED demo_project/MyNode.cpp)
#target_include_directories(demo_project PRIVATE include)
#target_link_libraries(demo_project PUBLIC zensession)
#set_property(TARGET demo_project PROPERTY LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/zen/autoload)
#set_property(TARGET demo_project PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/zen/autoload)
