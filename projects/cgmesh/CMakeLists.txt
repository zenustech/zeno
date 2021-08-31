add_subdirectory(voro)

set(LIBIGL_WITH_CGAL ON CACHE BOOL "we need igl::cgal")
add_subdirectory(libigl)

file(GLOB PROJECT_SOURCE *.cpp *.h)

add_library(zeno_cgmesh SHARED ${PROJECT_SOURCE})
target_include_directories(zeno_cgmesh PRIVATE .)

target_link_libraries(zeno_cgmesh PRIVATE voro++)
target_link_libraries(zeno_cgmesh PRIVATE igl::core igl::cgal)

find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
    message("found package: OpenMP::OpenMP_CXX")
    target_link_libraries(zeno_cgmesh PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(zeno_cgmesh PUBLIC zeno)
target_link_libraries(zeno_cgmesh PRIVATE zeno_nodep)
