cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)
project(zeno_FEM)

find_package(Eigen3 REQUIRED NO_MODULE)

set(FEM_MUSCLE_SOURCE_FILES
    src/matrix_helper.hpp
    src/differentiable_SVD.h
    src/differentiable_SVD.cpp
    src/tetrahedra_mesh.h
    src/tetrahedra_mesh.cpp
    FEMNode.cpp
)

add_library(zeno_FEM SHARED ${FEM_MUSCLE_SOURCE_FILES})
target_link_libraries(zeno_FEM PRIVATE Eigen3::Eigen)
target_include_directories(zeno_FEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(zeno_FEM PUBLIC zeno)
target_link_libraries(zeno_FEM PRIVATE zeno_nodep)
target_include_directories(zeno_FEM PRIVATE ../oldzenbase/include)
target_include_directories(zeno_FEM PRIVATE ../zenvdb/include)