project(Geometry)

cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)

target_sources(zeno PRIVATE quadmesh.cpp tetmesh.cpp)# stitchmesh.cpp)
target_link_libraries(zeno PRIVATE InstantMeshesApi)
target_link_libraries(zeno PRIVATE FTetwildApi)
# target_link_libraries(zeno PRIVATE StitchMeshingApi)
add_subdirectory(instant_meshes)
add_subdirectory(fTetWild)
# add_subdirectory(stitchMeshing)
