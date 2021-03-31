cmake_minimum_required(VERSION 3.19)
set(CMAKE_CXX_STANDARD 20)
project(FLIPlib)

# add directory with macros
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

# add cmake file
include (common)

find_package(TBB CONFIG REQUIRED COMPONENTS tbb tbbmalloc tbbmalloc_proxy)
find_package(IlmBase REQUIRED COMPONENTS Half REQUIRED)
find_package(Eigen3 REQUIRED)

#add_library(FLIPlib SHARED FLIPwrapper.cpp FLIPwrapper.h FLIP_particle fluidsim.cpp fluidsim_fusion_p2g.cpp fluidsim.h GeometricLevelGen.cpp GeometricLevelGen.h
#        levelset_util.cpp Sparse_buffer.cpp volumeMeshTools.cpp amg2.h morton_encoding.h FLIP_vdb.h FLIP_vdb.cpp simd_vdb_poisson.h simd_vdb_poisson.cpp)
set(FLIP_SOURCE levelset_util.cpp volumeMeshTools.cpp FLIP_vdb.h FLIP_vdb.cpp simd_vdb_poisson.h simd_vdb_poisson.cpp sdf_source_manager.h sdf_source_manager.cpp)
file(GLOB LOSTOPOS_SOURCE ltcommon/*.h ltcommon/*.cpp ltcommon/tunicate/*.h ltcommon/tunicate/*.cpp ltcommon/newsparse/*.h ltcommon/newsparse/*.cpp lostopos/*.h lostopos/*.cpp d3dutils/*.cpp d3dutils/*.h)
file(GLOB NOSYS_SOURCE nosys/*.cpp nosys/*.h)

add_library(FLIPlib SHARED ${FLIP_SOURCE} ${LOSTOPOS_SOURCE} ${NOSYS_SOURCE})
#add_executable(FLIPlib ${FLIP_SOURCE} ${LOSTOPOS_SOURCE})

#simd compiler flags
if (WIN32)
#target_compile_options(FLIPlib PRIVATE "/arch:AVX")
else()
target_compile_options(FLIPlib PRIVATE "-march=native")
endif()


target_include_directories(FLIPlib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(FLIPlib PUBLIC ltcommon ltcommon/tunicate ltcommon/newsparse lostopos d3dutils)
##target_include_directories(FLIPlib PRIVATE "LosTopos/common")
##target_include_directories(FLIPlib PRIVATE "Droplet3D")

target_link_libraries(FLIPlib PUBLIC TBB::tbb pthread  openblas f2c lapack blas)
target_link_libraries(FLIPlib PUBLIC TBB::tbbmalloc TBB::tbbmalloc_proxy)

# zen link flags
target_link_libraries(FLIPlib PRIVATE zen)

#disable default malloc options
if (WIN32)
target_compile_options(FLIPlib PUBLIC -DNOMINMAX)
endif()

target_link_libraries(FLIPlib PUBLIC openvdb IlmBase::Half)
target_link_libraries(FLIPlib PUBLIC Eigen3::Eigen)
#target_link_libraries(FLIPlib PUBLIC SimEnvironment)
#target_link_libraries(FLIPlib PUBLIC partio)
