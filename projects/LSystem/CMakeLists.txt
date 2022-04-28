add_subdirectory(LSystem)

target_sources(zeno PRIVATE
    zeno_LSystem.cpp
)
find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
    message("found package: OpenMP::OpenMP_CXX")
    target_link_libraries(zeno PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(zeno PRIVATE LSystem)
