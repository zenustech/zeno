cmake_minimum_required(VERSION 3.19)
set(CMAKE_CXX_STANDARD 17)
project(FLIPlib)
add_definitions(-D__TBB_CPP20_COMPARISONS_PRESENT=0)  
# add directory with macros
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

# add cmake file
include (common)

find_package(TBB CONFIG REQUIRED COMPONENTS tbb tbbmalloc)
find_package(IlmBase REQUIRED COMPONENTS Half REQUIRED)
find_package(Eigen3 REQUIRED)

#add_library(FLIPlib SHARED FLIPwrapper.cpp FLIPwrapper.h FLIP_particle fluidsim.cpp fluidsim_fusion_p2g.cpp fluidsim.h GeometricLevelGen.cpp GeometricLevelGen.h
#        levelset_util.cpp Sparse_buffer.cpp volumeMeshTools.cpp amg2.h morton_encoding.h FLIP_vdb.h FLIP_vdb.cpp simd_vdb_poisson.h simd_vdb_poisson.cpp)
set(FLIP_SOURCE levelset_util.cpp volumeMeshTools.cpp FLIP_vdb.h FLIP_vdb.cpp simd_vdb_poisson.h simd_vdb_poisson.cpp )
file(GLOB LOSTOPOS_SOURCE ltcommon/*.h ltcommon/*.cpp ltcommon/tunicate/*.h ltcommon/tunicate/*.cpp ltcommon/newsparse/*.h ltcommon/newsparse/*.cpp lostopos/*.h lostopos/*.cpp d3dutils/*.cpp d3dutils/*.h)
file(GLOB NOSYS_SOURCE nosys/*.cpp nosys/*.h)

add_library(FLIPlib SHARED ${FLIP_SOURCE} ${LOSTOPOS_SOURCE} ${NOSYS_SOURCE})

#simd compiler flags
if (WIN32)
#target_compile_options(FLIPlib PRIVATE "/arch:AVX")
else()
target_compile_options(FLIPlib PRIVATE "-march=native")
endif()


target_include_directories(FLIPlib PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(FLIPlib PRIVATE ltcommon ltcommon/tunicate ltcommon/newsparse lostopos d3dutils)

target_link_libraries(FLIPlib PRIVATE TBB::tbb TBB::tbbmalloc)
target_link_libraries(FLIPlib PRIVATE pthread openblas lapack cblas)

# zen link flags
target_link_libraries(FLIPlib PRIVATE zen)

#disable default malloc options
if (WIN32)
  target_compile_options(FLIPlib PRIVATE -DNOMINMAX)
endif()

target_link_directories(FLIPlib PUBLIC /usr/lib/x86_64-linux-gnu:/usr/local/lib)
target_link_libraries(FLIPlib PRIVATE openvdb IlmBase::Half)
target_link_libraries(FLIPlib PRIVATE Eigen3::Eigen)
#target_link_libraries(FLIPlib PRIVATE SimEnvironment)
target_link_libraries(FLIPlib PRIVATE partio)


