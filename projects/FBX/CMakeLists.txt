cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)
project(zeno_FBX)

file(GLOB FBX_SOURCE *.cpp *.h)

add_library(zeno_FBX SHARED ${FBX_SOURCE})

target_include_directories(zeno_FBX PRIVATE .)

add_subdirectory(assimp)

target_link_libraries(zeno_FBX PRIVATE assimp)
target_link_libraries(zeno_FBX PRIVATE zeno)