add_subdirectory(voro)

set(LIBIGL_INCLUDE_DIRS /home/bate/Codes/libigl/include)
find_package(LIBIGL QUIET)
if (NOT LIBIGL_FOUND)
    message(FATAL_ERROR "libigl not found -- install it using: "
    "https://github.com/libigl/libigl.git")
endif()
find_package(Eigen3 REQUIRED)

file(GLOB PROJECT_SOURCE *.cpp *.h)

add_library(zeno_cgmesh SHARED ${PROJECT_SOURCE})
target_include_directories(zeno_cgmesh PRIVATE .)

target_link_libraries(zeno_cgmesh PRIVATE voro++)
target_link_libraries(zeno_cgmesh PRIVATE igl::core igl_cgal Eigen3::Eigen)

find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
    message("found package: OpenMP::OpenMP_CXX")
    target_link_libraries(zeno_cgmesh PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(zeno_cgmesh PUBLIC zeno)
target_link_libraries(zeno_cgmesh PRIVATE zeno_nodep)
