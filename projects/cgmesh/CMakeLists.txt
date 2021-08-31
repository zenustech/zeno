add_subdirectory(voro)

file(GLOB PROJECT_SOURCE *.cpp *.h)

add_library(zeno_cgmesh SHARED ${PROJECT_SOURCE})
target_include_directories(zeno_cgmesh PRIVATE .)

target_link_libraries(zeno_cgmesh PRIVATE voro++)

find_package(CGAL REQUIRED)
target_link_libraries(zeno_cgmesh PRIVATE CGAL::CGAL)

find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
    message("found package: OpenMP::OpenMP_CXX")
    target_link_libraries(zeno_cgmesh PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(zeno_cgmesh PUBLIC zeno)
target_link_libraries(zeno_cgmesh PRIVATE zeno_nodep)
