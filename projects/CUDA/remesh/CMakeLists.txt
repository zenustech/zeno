
project(remesh)

cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)

message( "-----remesh------")

target_sources(zeno PRIVATE 
  remesh.cpp
  SurfaceMesh.cpp
)
add_subdirectory(algorithms)
