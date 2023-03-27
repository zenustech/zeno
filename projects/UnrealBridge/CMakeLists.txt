cmake_minimum_required(VERSION 3.10)

file(GLOB_RECURSE PROJECT_SOURCE *.h *.cpp)

target_compile_definitions(zenoedit PRIVATE -DZENO_WITH_UnrealBridge)

target_include_directories(zenoedit PRIVATE .)

target_sources(zenoedit PRIVATE ${PROJECT_SOURCE})
