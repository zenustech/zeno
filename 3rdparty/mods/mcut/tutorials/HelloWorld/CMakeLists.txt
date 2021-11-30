#get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
#project(${PROJECT_NAME})

add_executable(HelloWorld HelloWorld.cpp)
target_link_libraries(HelloWorld mcut ${extra_libs})
target_include_directories(HelloWorld PRIVATE ${MCUT_INCLUDE_DIR})
target_compile_options(HelloWorld PRIVATE ${compilation_flags})
target_compile_definitions(HelloWorld PRIVATE ${preprocessor_defs})