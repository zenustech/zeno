cmake_minimum_required(VERSION 3.10)

project(zeno_LSystem)

add_subdirectory(LSystem)

# ZENO extension must be a SHARED library with name starting with `zeno_`
# otherwise it won't be automcatically loaded on ZENO startup
add_library(zeno_LSystem SHARED
    zeno_LSystem.cpp
)
find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
    message("found package: OpenMP::OpenMP_CXX")
    target_link_libraries(zeno_LSystem PRIVATE OpenMP::OpenMP_CXX)
endif()

# Finally don't forget to link zeno.dll
target_link_libraries(zeno_LSystem PUBLIC zeno LSystem)
