project(Geometry)

cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)

target_sources(zeno PRIVATE quadmesh.cpp)
target_link_libraries(zeno PRIVATE InstantMeshesApi)
add_subdirectory(instant_meshes)
