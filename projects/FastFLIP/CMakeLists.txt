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

set(FLIP_SOURCE levelset_util.cpp volumeMeshTools.cpp FLIP_vdb.h FLIP_vdb.cpp simd_vdb_poisson.h simd_vdb_poisson.cpp SimOptions.cpp Timer.cpp )
file(GLOB NOSYS_SOURCE nosys/*.cpp nosys/*.h)

add_library(FLIPlib SHARED ${FLIP_SOURCE} ${NOSYS_SOURCE})

#simd compiler flags
if (WIN32)
#target_compile_options(FLIPlib PRIVATE "/arch:AVX")
else()
target_compile_options(FLIPlib PRIVATE "-march=native")
endif()


target_include_directories(FLIPlib PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(FLIPlib PRIVATE TBB::tbb TBB::tbbmalloc)
target_link_libraries(FLIPlib PRIVATE pthread openblas lapack)

#disable default malloc options
if (WIN32)
  target_compile_options(FLIPlib PRIVATE -DNOMINMAX)
endif()

target_link_directories(FLIPlib PUBLIC /usr/lib/x86_64-linux-gnu:/usr/local/lib)
target_link_libraries(FLIPlib PRIVATE OpenVDB::openvdb IlmBase::Half)
target_link_libraries(FLIPlib PRIVATE Eigen3::Eigen)

add_subdirectory(partio)
target_link_libraries(FLIPlib PRIVATE partio)


find_package(zeno REQUIRED)
target_link_libraries(FLIPlib PRIVATE zeno)
set_target_properties(FLIPlib PROPERTIES
	LIBRARY_OUTPUT_DIRECTORY ${zeno_AUTOLOAD_DIR}
	RUNTIME_OUTPUT_DIRECTORY ${zeno_AUTOLOAD_DIR}
	ARCHIVE_OUTPUT_DIRECTORY ${zeno_AUTOLOAD_DIR}
	LIBRARY_OUTPUT_DIRECTORY_DEBUG ${zeno_AUTOLOAD_DIR}
	RUNTIME_OUTPUT_DIRECTORY_DEBUG ${zeno_AUTOLOAD_DIR}
	ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${zeno_AUTOLOAD_DIR}
	LIBRARY_OUTPUT_DIRECTORY_RELEASE ${zeno_AUTOLOAD_DIR}
	RUNTIME_OUTPUT_DIRECTORY_RELEASE ${zeno_AUTOLOAD_DIR}
	ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${zeno_AUTOLOAD_DIR}
	)
target_include_directories(FLIPlib PRIVATE ../zenbase/include)
target_include_directories(FLIPlib PRIVATE ../zenvdb/include)
