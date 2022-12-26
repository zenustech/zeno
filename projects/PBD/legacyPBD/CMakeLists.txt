add_library(legacyPBD INTERFACE "")

add_subdirectory(Cloth)
add_subdirectory(Fluid)

target_sources(legacyPBD INTERFACE
${CMAKE_CURRENT_LIST_DIR}/Soft/PBDSoftBody.cpp
${CMAKE_CURRENT_LIST_DIR}/Soft/PBDSoftBodyInit.cpp
${CMAKE_CURRENT_LIST_DIR}/Soft/BunnyMesh/BunnyMesh.cpp
${CMAKE_CURRENT_LIST_DIR}/Soft/ReadVtkTet/ReadVtkTet.cpp
)

target_sources(legacyPBD INTERFACE
${CMAKE_CURRENT_LIST_DIR}/Constraints/PBDSolveDistanceConstraint.cpp
${CMAKE_CURRENT_LIST_DIR}/Constraints/PBDSolveVolumeConstraint.cpp
)


option(PBD_TEST "Build the PBD test" OFF)
if(PBD_TEST)
    add_subdirectory(tests)
endif()

target_include_directories(legacyPBD INTERFACE .)
target_link_libraries(zeno PRIVATE legacyPBD)