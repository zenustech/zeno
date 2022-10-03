target_sources(zeno PRIVATE 
PBDSoftBody.cpp 
PBDSolveDistanceConstraint.cpp 
PBDSolveVolumeConstraint.cpp
PBDSoftBodyInit.cpp
)
add_subdirectory(PBDCloth)
add_subdirectory(PBF)
add_subdirectory(BunnyMesh)
add_subdirectory(ReadVtkTet)