target_sources(zeno PRIVATE 
# PBF.cpp 
# NeighborSearch_BVH.cpp
# NeighborSearch.cpp
# PBFWorld_Setup.cpp
# PBFWorld_PreSolve.cpp
# PBFWorld_NeighborhoodSearch.cpp
# PBFWorld_Solve.cpp
# PBFWorld_PostSolve.cpp
# PBFWorld_Step.cpp
# SPHKernels.cpp
# PBFWorld_testCube.cpp
# PBFWorld_testCube2.cpp
# PBFWorld_Debug.cpp
testPBFCube.cpp
PBF_BVH.cpp
PBF_BVH.h
)

target_include_directories(zeno PRIVATE ${PROJECT_SOURCE_DIR})

if(NOT ZENO_WITH_ZenoFX)
  message(FATAL_ERROR "ZenoFX must be ON when PBD is ON! "
    "Please specify: -DZENO_WITH_ZenoFX:BOOL=ON")
endif()