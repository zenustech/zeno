
project(remesh)

cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)


target_sources(zeno PRIVATE 
  simplification.cpp
  remesh.cpp
  repair.cpp
  SurfaceMesh.cpp
  TriSubdiv.cpp
)
add_subdirectory(algorithms)
