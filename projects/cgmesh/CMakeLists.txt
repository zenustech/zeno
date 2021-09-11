add_library(zeno_cgmesh SHARED)
target_include_directories(zeno_cgmesh PRIVATE .)

option(ZENO_CGMESH_WITH_VOROXX "Build with Voro++ nodes" ON)
if (ZENO_CGMESH_WITH_VOROXX)
    target_sources(zeno_cgmesh PRIVATE VoronoiFracture.cpp)
    add_subdirectory(voro)
    target_link_libraries(zeno_cgmesh PRIVATE voro++)
endif()

option(ZENO_CGMESH_WITH_MESHFIX "Build with MeshFix nodes" ON)
if (ZENO_CGMESH_WITH_MESHFIX)
    target_sources(zeno_cgmesh PRIVATE PrimitiveMeshingFix.cpp)
    add_subdirectory(meshfix)
    target_include_directories(zeno_cgmesh PRIVATE meshfix/JMeshExt-1.0alpha_src/include /home/bate/Codes/zeno-blender/external/zeno/projects/cgmesh/meshfix/JMeshExt-1.0alpha_src/JMeshLib-1.2/include)
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    add_definitions(-DUSE_FORTRAN_BLAS)
    if(APPLE)
      set(CMAKE_CXX_LINK_FLAGS "-framework Accelerate")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    endif()
    target_link_libraries(
        zeno_cgmesh PRIVATE
      meshfix
      jmesh
      jmeshext
      nl
      superlu
      ${BLAS_LIBRARIES}
      ${LAPACK_LIBRARIES}
      )
endif()

option(ZENO_CGMESH_WITH_LIBIGL "Build with LIBIGL nodes" ON)
if (ZENO_CGMESH_WITH_LIBIGL)
    #set(CMAKE_LINK_WHAT_YOU_USE TRUE)
    target_sources(zeno_cgmesh PRIVATE igl_sink.cpp igl_dummy.cpp PrimitiveBooleanOp.cpp)
    if (UNIX)
        find_package(LIBIGL QUIET)
        if (NOT LIBIGL_FOUND)
            message(FATAL_ERROR "libigl not found -- install it from source: "
            "https://github.com/libigl/libigl.git")
        endif()
        find_package(Eigen3 REQUIRED)
        set(LIBIGL_INCLUDE_DIR /home/bate/Codes/libigl/include)
        set(LIBIGL_EIGEN_INCLUDE_DIR /usr/include/eigen3)
        if (NOT LIBIGL_INCLUDE_DIR)
            message(FATAL_ERROR "please specify LIBIGL_INCLUDE_DIR")
        endif()
        if (NOT LIBIGL_EIGEN_INCLUDE_DIR)
            message(FATAL_ERROR "please specify LIBIGL_EIGEN_INCLUDE_DIR")
        endif()
        target_include_directories(zeno_cgmesh PRIVATE ${LIBIGL_INCLUDE_DIR} ${LIBIGL_EIGEN_INCLUDE_DIR})
        target_link_libraries(zeno_cgmesh PRIVATE igl::core igl igl_cgal Eigen3::Eigen gmp mpfr boost_system boost_log)
    else()
        find_package(LIBIGL CONFIG REQUIRED)
        find_package(Eigen3 CONFIG REQUIRED)
        target_link_libraries(zeno_cgmesh PRIVATE igl::core igl::cgal Eigen3::Eigen)
    endif()
endif()

find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
    message("found package: OpenMP::OpenMP_CXX")
    target_link_libraries(zeno_cgmesh PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(zeno_cgmesh PUBLIC zeno)
target_link_libraries(zeno_cgmesh PRIVATE zeno_nodep)
