cmake_minimum_required(VERSION 3.19)
set(CMAKE_CXX_STANDARD 17)

project(AdaptiveSolver CXX CUDA)
set(ZS_ENABLE_CUDA ON)
add_definitions(-D__TBB_CPP20_COMPARISONS_PRESENT=0)  

# add cmake file
include (common)

find_package(TBB CONFIG REQUIRED COMPONENTS tbb tbbmalloc)
find_package(IlmBase REQUIRED COMPONENTS Half REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenVDB REQUIRED)
find_package(CUDAToolkit)

if (WIN32)
  include_directories(C:/Users/archibate/packages/openvdb_x64-windows/include/)
endif()
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(AdaptSource AdaptiveGridGen.h AdaptiveSolver.cpp AdaptiveGridGen.cpp )
set(gpuAdaptSource GridData.cuh GridData.cu)
#file(GLOB NOSYS_SOURCE nosys/*.cpp nosys/*.h)

add_library(zeno_AdaptiveSolver SHARED ${AdaptSource})
add_library(zeno_gpuAdaptiveSolver SHARED ${gpuAdaptSource})

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

# gpu adaptive solver
if(${ZS_ENABLE_CUDA})
    enable_language(CUDA)

    target_compile_features(zeno_gpuAdaptiveSolver PRIVATE cuda_std_17)
    target_compile_options(zeno_gpuAdaptiveSolver 
      PRIVATE        $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr --default-stream=per-thread --use_fast_math -lineinfo --ptxas-options=-allow-expensive-optimizations=true>
      PUBLIC        $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --diag_suppress=554>
    )
    target_link_options(zeno_gpuAdaptiveSolver 
      PUBLIC       $<$<LINK_LANGUAGE:CUDA>:-dlto>
    )
    set_target_properties(zeno_gpuAdaptiveSolver
      PROPERTIES  CUDA_EXTENSIONS ON
                  CUDA_SEPARABLE_COMPILATION ON
                  LINKER_LANGUAGE CUDA
                  POSITION_INDEPENDENT_CODE ON
                  CUDA_ARCHITECTURES OFF
    )
endif()

target_include_directories(zeno_gpuAdaptiveSolver PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(zeno_gpuAdaptiveSolver PRIVATE inc)

target_link_libraries(zeno_gpuAdaptiveSolver PRIVATE TBB::tbb TBB::tbbmalloc)
if (UNIX)
    target_link_libraries(zeno_gpuAdaptiveSolver PRIVATE pthread)
endif()

#disable default malloc options
if (WIN32)
  target_compile_options(zeno_gpuAdaptiveSolver PRIVATE -DNOMINMAX -D_USE_MATH_DEFINES)
endif()
if (UNIX)
  target_link_directories(zeno_gpuAdaptiveSolver PUBLIC /usr/lib/x86_64-linux-gnu:/usr/local/lib)
endif()
target_link_libraries(zeno_gpuAdaptiveSolver PRIVATE OpenVDB::openvdb IlmBase::Half)
target_link_libraries(zeno_gpuAdaptiveSolver PRIVATE Eigen3::Eigen)

target_link_libraries(zeno_gpuAdaptiveSolver PUBLIC zeno zeno_zenvdb)
target_include_directories(zeno_gpuAdaptiveSolver PRIVATE ../oldzenbase/include)
target_include_directories(zeno_gpuAdaptiveSolver PRIVATE ../zenvdb/include)

add_definitions(-w)