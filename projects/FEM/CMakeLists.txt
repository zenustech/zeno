cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)

if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
endif(WIN32)

project(zeno_FEM)

find_package(Eigen3 REQUIRED NO_MODULE)

if(WIN32)
    if(TARGET igl::core)
    else()
        find_package(libigl REQUIRED)
    endif()
endif(WIN32)


find_package(OpenMP)
if(NOT OpenMP_CXX_FOUND)

endif(NOT OpenMP_CXX_FOUND)

set(FEM_MUSCLE_SOURCE_FILES
    src/math/matrix_helper.hpp
    src/math/differentiable_SVD.h
    src/math/differentiable_SVD.cpp
    src/force_model/stable_anisotropic_NH.h
    src/force_model/stable_isotropic_NH.h
    src/force_model/base_elastic_model.h
    src/force_model/stable_Stvk.h
    src/force_model/bspline_isotropic_model.h
    src/integrator/base_integrator.h
    src/integrator/backward_euler_integrator.h
    src/integrator/backward_euler_integrator.cpp
    src/integrator/quasi_static_solver.h
    src/integrator/quasi_static_solver.cpp
    src/bspline/cubicBspline.cpp
    src/bspline/cubicBspline.h
    equa_solver.cpp
    integrator.cpp
    fem_tools.cpp
    force_model.cpp
)

add_library(zeno_FEM SHARED ${FEM_MUSCLE_SOURCE_FILES})
target_link_libraries(zeno_FEM PRIVATE Eigen3::Eigen)
target_link_libraries(zeno_FEM PUBLIC OpenMP::OpenMP_CXX)

target_include_directories(zeno_FEM PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/math
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mesh
    ${CMAKE_CURRENT_SOURCE_DIR}/src/force_model
    ${CMAKE_CURRENT_SOURCE_DIR}/src/integrator
    ${CMAKE_CURRENT_SOURCE_DIR}/src/bspline   
    ${CMAKE_CURRENT_SOURCE_DIR}/LBFGSpp/include
)

target_link_libraries(zeno_FEM PUBLIC zeno)
target_link_libraries(zeno_FEM PRIVATE zeno_nodep)
target_link_libraries(zeno_FEM PRIVATE igl::core)
target_include_directories(zeno_FEM PRIVATE ../oldzenbase/include)
target_include_directories(zeno_FEM PRIVATE ../zenvdb/include)