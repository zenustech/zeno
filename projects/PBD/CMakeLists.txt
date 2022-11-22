target_sources(zeno PRIVATE 
PBDSoftBody.cpp 
PBDSolveDistanceConstraint.cpp 
PBDSolveVolumeConstraint.cpp
PBDSoftBodyInit.cpp
PBDPostSolve.cpp
PBDPreSolve.cpp
PBDCollision.cpp
PBDRestPos.cpp
)
add_subdirectory(PBDCloth)
# add_subdirectory(PBF)
add_subdirectory(BunnyMesh)
add_subdirectory(ReadVtkTet)

# # please comment out the test source files in production environment
add_subdirectory(test)