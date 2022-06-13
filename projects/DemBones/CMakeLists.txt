file(GLOB DEM_SOURCE *.cpp *.h)

find_package(Eigen3 REQUIRED)

target_sources(zeno PRIVATE ${DEM_SOURCE})

target_include_directories(zeno PRIVATE .)
target_include_directories(zeno PRIVATE ./include)
target_link_libraries(zeno PRIVATE Eigen3::Eigen)