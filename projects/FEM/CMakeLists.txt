cmake_minimum_required(VERSION 3.16)
if (NOT ZENO_WITH_cgmesh)
    message(FATAL_ERROR "ZENO_WITH_cgmesh must be ON when ZENO_WITH_FEM is ON")
endif()

if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
endif(WIN32)

#project(zenoFEM)

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

target_sources(zeno PRIVATE ${FEM_MUSCLE_SOURCE_FILES})
target_link_libraries(zeno PRIVATE Eigen3::Eigen)
target_link_libraries(zeno PUBLIC OpenMP::OpenMP_CXX)

target_include_directories(zeno PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/math
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mesh
    ${CMAKE_CURRENT_SOURCE_DIR}/src/force_model
    ${CMAKE_CURRENT_SOURCE_DIR}/src/integrator
    ${CMAKE_CURRENT_SOURCE_DIR}/src/bspline   
)

target_link_libraries(zeno PRIVATE igl::core)
target_include_directories(zeno PRIVATE ../oldzenbase/include)
target_include_directories(zeno PRIVATE ../zenvdb/include)
