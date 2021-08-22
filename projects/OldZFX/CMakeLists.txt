file (GLOB WRANGLE_SOURCE *.cpp)
add_library(zeno_OldZFX SHARED ${WRANGLE_SOURCE})
target_include_directories(zeno_OldZFX PRIVATE .)

target_link_libraries(zeno_OldZFX PUBLIC zeno)
target_link_libraries(zeno_OldZFX PRIVATE zeno_nodep)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
