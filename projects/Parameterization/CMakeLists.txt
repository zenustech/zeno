cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)
# if (NOT ZENO_WITH_cgmesh)
#     message(FATAL_ERROR "ZENO_WITH_cgmesh must be ON when ZENO_WITH_Paramterization is ON")
# endif()

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenMP)

set(PARAMETERIZATION_SOURCE_FILES
    harmonic_parameterization.cpp
    arap_parameterization.cpp
    nrosy.cpp
    read_off.cpp
    lscm_parameterization.cpp
    MIQ.cpp
)

target_sources(zeno PRIVATE ${PARAMETERIZATION_SOURCE_FILES})
target_link_libraries(zeno PRIVATE Eigen3::Eigen)
target_link_libraries(zeno PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(zeno PRIVATE igl_copyleft::core igl_copyleft::cgal)
if(TARGET igl_copyleft::comiso)
    message(STATUS "Found CoMiSo")
    target_link_libraries(zeno PRIVATE igl_copyleft::comiso)
else()
    message(FATAL_ERROR "Failed find CoMiSo Lib Target")
endif()


target_include_directories(zeno PRIVATE ../oldzenbase/include)
target_include_directories(zeno PRIVATE ../zenvdb/include)