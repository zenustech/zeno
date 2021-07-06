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
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(FLIP_SOURCE levelset_util.cpp FLIP_vdb.h FLIP_vdb.cpp simd_vdb_poisson.h simd_vdb_poisson.cpp )
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
target_link_libraries(FLIPlib PRIVATE pthread)

#disable default malloc options
if (WIN32)
  target_compile_options(FLIPlib PRIVATE -DNOMINMAX)
endif()

target_link_directories(FLIPlib PUBLIC /usr/lib/x86_64-linux-gnu:/usr/local/lib)
target_link_libraries(FLIPlib PRIVATE OpenVDB::openvdb IlmBase::Half)
target_link_libraries(FLIPlib PRIVATE Eigen3::Eigen)

add_subdirectory(partio)
IF (NOT WIN32)
    INSTALL(TARGETS partio DESTINATION lib)
ENDIF()
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
target_include_directories(FLIPlib PRIVATE ../oldzenbase/include)
target_include_directories(FLIPlib PRIVATE ../zenvdb/include)
