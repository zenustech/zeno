cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)
project(zeno_FEM)

find_package(Eigen3 REQUIRED NO_MODULE)

find_package(OpenMP)
if(NOT OpenMP_CXX_FOUND)

endif(NOT OpenMP_CXX_FOUND)

set(FEM_MUSCLE_SOURCE_FILES
    src/math/matrix_helper.hpp
    src/math/differentiable_SVD.h
    src/math/differentiable_SVD.cpp
    src/force_model/stable_anisotropic_NH.h
    src/force_model/stable_isotropic_NH.h
    src/force_model/base_force_model.h
    src/integrator/base_integrator.h
    src/integrator/backward_euler_integrator.h
    src/integrator/backward_euler_integrator.cpp
    src/integrator/quasi_static_solver.h
    src/integrator/quasi_static_solver.cpp
    nodesys.cpp
    testopensim.cpp
)

add_library(zeno_FEM SHARED ${FEM_MUSCLE_SOURCE_FILES})
find_package(Eigen3 REQUIRED)
target_link_libraries(zeno_FEM PRIVATE Eigen3::Eigen)
find_package(OpenMP REQUIRED)
target_link_libraries(zeno_FEM PUBLIC OpenMP::OpenMP_CXX)

target_include_directories(zeno_FEM PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/math
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mesh
    ${CMAKE_CURRENT_SOURCE_DIR}/src/force_model
    ${CMAKE_CURRENT_SOURCE_DIR}/src/integrator)

target_link_libraries(zeno_FEM PUBLIC zeno)
target_link_libraries(zeno_FEM PRIVATE zeno_nodep)
target_include_directories(zeno_FEM PRIVATE ../oldzenbase/include)
target_include_directories(zeno_FEM PRIVATE ../zenvdb/include)

add_subdirectory(OpenSim)
target_link_libraries(zeno_FEM PRIVATE OpenSim)
target_include_directories(zeno_FEM PRIVATE OpenSim
        OpenSim/dependencies/simbody/SimTKcommon/include              OpenSim/dependencies/simbody/SimTKcommon/Simulation/include
 OpenSim/dependencies/simbody/Simbody/include                 OpenSim/dependencies/simbody/SimTKcommon/Mechanics/include    OpenSim/dependencies/simbody/SimTKcommon/SmallMatrix/include
 OpenSim/dependencies/simbody/Simbody/Visualizer/include      OpenSim/dependencies/simbody/SimTKcommon/Polynomial/include   OpenSim/dependencies/simbody/SimTKmath/Geometry/include
 OpenSim/dependencies/simbody/SimTKcommon/BigMatrix/include   OpenSim/dependencies/simbody/SimTKcommon/Random/include       OpenSim/dependencies/simbody/SimTKmath/include
 OpenSim/dependencies/simbody/SimTKcommon/Geometry/include    OpenSim/dependencies/simbody/SimTKcommon/Scalar/include       OpenSim/dependencies/simbody/SimTKmath/Integrators/include
)
