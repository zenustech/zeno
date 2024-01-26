project(Geometry)

cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)

target_sources(zeno PRIVATE quadmesh.cpp tetmesh.cpp)
target_link_libraries(zeno PRIVATE InstantMeshesApi)
target_link_libraries(zeno PRIVATE FTetwildApi)
add_subdirectory(instant_meshes)
add_subdirectory(fTetWild)
