cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)
project(FLIPlib)
add_definitions(-D__TBB_CPP20_COMPARISONS_PRESENT=0)  

# add cmake file
include (common)

find_package(TBB CONFIG REQUIRED COMPONENTS tbb tbbmalloc)
find_package(IlmBase REQUIRED COMPONENTS Half REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenVDB REQUIRED)

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
target_link_libraries(FLIPlib PRIVATE pthread openblas lapack)

#disable default malloc options
if (WIN32)
  target_compile_options(FLIPlib PRIVATE -DNOMINMAX)
endif()

target_link_directories(FLIPlib PUBLIC /usr/lib/x86_64-linux-gnu:/usr/local/lib)
target_link_libraries(FLIPlib PRIVATE OpenVDB::openvdb IlmBase::Half)
target_link_libraries(FLIPlib PRIVATE Eigen3::Eigen)
#target_link_libraries(FLIPlib PRIVATE SimEnvironment)

add_subdirectory(partio)
target_link_libraries(FLIPlib PRIVATE partio)


find_package(zen REQUIRED)
target_link_libraries(FLIPlib PRIVATE zen)
set_target_properties(FLIPlib PROPERTIES
	LIBRARY_OUTPUT_DIRECTORY ${zen_AUTOLOAD_DIR}
	RUNTIME_OUTPUT_DIRECTORY ${zen_AUTOLOAD_DIR}
	ARCHIVE_OUTPUT_DIRECTORY ${zen_AUTOLOAD_DIR}
	LIBRARY_OUTPUT_DIRECTORY_DEBUG ${zen_AUTOLOAD_DIR}
	RUNTIME_OUTPUT_DIRECTORY_DEBUG ${zen_AUTOLOAD_DIR}
	ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${zen_AUTOLOAD_DIR}
	LIBRARY_OUTPUT_DIRECTORY_RELEASE ${zen_AUTOLOAD_DIR}
	RUNTIME_OUTPUT_DIRECTORY_RELEASE ${zen_AUTOLOAD_DIR}
	ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${zen_AUTOLOAD_DIR}
	)
target_include_directories(FLIPlib PRIVATE ../zenbase/include)
target_include_directories(FLIPlib PRIVATE ../zenvdb/include)
