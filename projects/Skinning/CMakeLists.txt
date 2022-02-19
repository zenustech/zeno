cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)
project(zeno_SKINNING)

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
)

add_library(zeno_SKINNING SHARED ${SKINNING_SOURCE_FILES})
target_link_libraries(zeno_SKINNING PRIVATE Eigen3::Eigen)
target_link_libraries(zeno_SKINNING PUBLIC OpenMP::OpenMP_CXX)

target_link_libraries(zeno_SKINNING PRIVATE igl::core igl::cgal igl::tetgen)
# target_link_libraries(zeno_SKINNING PRIVATE libTetWild)

target_link_libraries(zeno_SKINNING PUBLIC zeno)
target_link_libraries(zeno_SKINNING PRIVATE zeno_nodep)
target_include_directories(zeno_SKINNING PRIVATE ../oldzenbase/include)
target_include_directories(zeno_SKINNING PRIVATE ../zenvdb/include)