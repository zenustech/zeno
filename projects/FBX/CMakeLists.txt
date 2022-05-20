cmake_minimum_required(VERSION 3.16)

file(GLOB FBX_SOURCE *.cpp *.h)

target_sources(zeno PRIVATE ${FBX_SOURCE})

target_include_directories(zeno PRIVATE .)

add_subdirectory(assimp)

target_link_libraries(zeno PRIVATE assimp)
