cmake_minimum_required(VERSION 3.10)

file(GLOB_RECURSE PROJECT_SOURCE *.h *.cpp)
FILE(GLOB_RECURSE STATIC_DEFINITION "StaticDefinition.cxx")

target_compile_definitions(zeno PUBLIC -DZENO_WITH_UnrealTool)
# Fix for windows.h and winsock2.h conflict
target_compile_definitions(zeno PUBLIC -DWIN32_LEAN_AND_MEAN)

target_include_directories(zeno PUBLIC ./include)

target_sources(zeno PRIVATE ${PROJECT_SOURCE})
target_sources(zeno PRIVATE ${STATIC_DEFINITION})
