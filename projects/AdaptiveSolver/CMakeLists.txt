cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)
project(zeno_AdaptiveSolver)
add_definitions(-D__TBB_CPP20_COMPARISONS_PRESENT=0)  

# add cmake file
include (common)

find_package(TBB CONFIG REQUIRED COMPONENTS tbb tbbmalloc)
find_package(IlmBase REQUIRED COMPONENTS Half REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenVDB REQUIRED)
if (WIN32)
  include_directories(C:/Users/archibate/packages/openvdb_x64-windows/include/)
endif()
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(AdaptSource AdaptiveGridGen.h AdaptiveSolver.cpp AdaptiveGridGen.cpp)
#file(GLOB NOSYS_SOURCE nosys/*.cpp nosys/*.h)

add_library(zeno_AdaptiveSolver SHARED ${AdaptSource} ${NOSYS_SOURCE})

#simd compiler flags
if (WIN32)
#target_compile_options(zeno_FastFLIP PRIVATE "/arch:AVX")
else()
target_compile_options(zeno_AdaptiveSolver PRIVATE "-march=native")
endif()


target_include_directories(zeno_AdaptiveSolver PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(zeno_AdaptiveSolver PRIVATE TBB::tbb TBB::tbbmalloc)
if (UNIX)
    target_link_libraries(zeno_AdaptiveSolver PRIVATE pthread)
endif()

#disable default malloc options
if (WIN32)
  target_compile_options(zeno_AdaptiveSolver PRIVATE -DNOMINMAX -D_USE_MATH_DEFINES)
endif()
if (UNIX)
  target_link_directories(zeno_AdaptiveSolver PUBLIC /usr/lib/x86_64-linux-gnu:/usr/local/lib)
endif()
target_link_libraries(zeno_AdaptiveSolver PRIVATE OpenVDB::openvdb IlmBase::Half)
target_link_libraries(zeno_AdaptiveSolver PRIVATE Eigen3::Eigen)


target_link_libraries(zeno_AdaptiveSolver PUBLIC zeno zeno_zenvdb)
target_include_directories(zeno_AdaptiveSolver PRIVATE ../oldzenbase/include)
target_include_directories(zeno_AdaptiveSolver PRIVATE ../zenvdb/include)

add_definitions(-w)