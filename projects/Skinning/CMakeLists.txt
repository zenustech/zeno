cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)
if (NOT ZENO_WITH_cgmesh)
    message(FATAL_ERROR "ZENO_WITH_cgmesh must be ON when ZENO_WITH_Skinning is ON")
endif()

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenMP)
if(NOT OpenMP_CXX_FOUND)

endif(NOT OpenMP_CXX_FOUND)

set(SKINNING_SOURCE_FILES
    bone_animation.cpp
    bone_system.cpp
    igl_file_reader.cpp
    skinning_bbw.cpp
    skinning.cpp
    tetrahedralize.cpp
    laplace_equa.cpp
    gradient.cpp
    deformation_field.cpp
    extract_tetrahedra_surface.cpp
    triangularize_quads_surf.cpp
    arap.cpp
)

target_sources(zeno PRIVATE ${SKINNING_SOURCE_FILES})
target_link_libraries(zeno PRIVATE Eigen3::Eigen)
target_link_libraries(zeno PUBLIC OpenMP::OpenMP_CXX)

target_link_libraries(zeno PRIVATE igl::core $<TARGET_NAME_IF_EXISTS:igl::cgal> igl::tetgen)
# target_link_libraries(zeno PRIVATE libTetWild)

target_include_directories(zeno PRIVATE ../oldzenbase/include)
target_include_directories(zeno PRIVATE ../zenvdb/include)
