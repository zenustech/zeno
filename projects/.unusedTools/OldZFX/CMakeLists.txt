file (GLOB WRANGLE_SOURCE *.cpp)
target_sources(zeno PRIVATE ${WRANGLE_SOURCE})
target_include_directories(zeno PRIVATE .)

find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
    message("found package: OpenMP::OpenMP_CXX")
    target_link_libraries(zeno PRIVATE OpenMP::OpenMP_CXX)
endif()
