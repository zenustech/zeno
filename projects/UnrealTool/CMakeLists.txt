cmake_minimum_required(VERSION 3.10)

file(GLOB_RECURSE PROJECT_SOURCE *.h *.cpp)

target_compile_definitions(zeno PRIVATE -DZENO_WITH_UnrealTool)

target_include_directories(zeno PUBLIC ./include)

target_sources(zeno PRIVATE ${PROJECT_SOURCE})
