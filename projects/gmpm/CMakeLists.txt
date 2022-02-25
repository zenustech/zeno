cmake_minimum_required(VERSION 3.19) # TODO: wxl cihou object-library!

# assume cuda
project(MPM CXX CUDA)
set(ZS_ENABLE_CUDA ON)
set(ZS_ENABLE_OPENVDB ON)
set(ZENSIM_INSTALL OFF)
# set(ZS_BUILD_SHARED_LIBS ON)
add_subdirectory(zecomp)

find_package(CUDAToolkit)

add_library(zeno_zsfem SHARED)
target_sources(zeno_zsfem PRIVATE
  fem/test.cpp
  fem/MeshIO.cpp
  fem/Fiber.cpp
  fem/Integrator.cpp
)

add_library(zeno_zsfft SHARED)
target_sources(zeno_zsfft PRIVATE
  oceanfft/Ocean.cu
)
target_compile_options(zeno_zsfft # MSVC, GNU, Clang, Intel
  PUBLIC   $<$<COMPILE_LANGUAGE:CXX>: $<IF:$<CXX_COMPILER_ID:MSVC>, /openmp:llvm /openmp:experimental, -fopenmp $<IF:$<CXX_COMPILER_ID:Clang>, , > >>
)

add_library(zeno_mpm SHARED)
target_sources(zeno_mpm PRIVATE
  utils/IndexBuckets.cu
  mpm/Generation.cu
  mpm/Generation.cpp
  mpm/MPMPipeline.cu
  mpm/Maintenance.cu
)
target_link_libraries(zeno_mpm PRIVATE zensim)
target_link_libraries(zeno_mpm PRIVATE zeno_nodep)
target_link_libraries(zeno_mpm PUBLIC zeno ZFX)
target_include_directories(zeno_mpm PRIVATE ../oldzenbase/include)
target_include_directories(zeno_mpm PRIVATE ../zenvdb/include)
if (NOT MSVC)
target_sources(zeno_mpm PRIVATE 
  mpm/PW.cu
  mpm/PNW.cu
  mpm/PPW.cu
)
endif()
target_compile_options(zeno_mpm 
  PUBLIC        $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --diag_suppress=554>
)

# zeno_zsfluid
add_library(zeno_zsfluid SHARED)
target_sources(zeno_zsfluid PRIVATE
  volume/VolumeOperator.cu
)
target_link_libraries(zeno_zsfluid PRIVATE zensim)
target_link_libraries(zeno_zsfluid PRIVATE zeno_nodep)
target_link_libraries(zeno_zsfluid PUBLIC zeno ZFX)
target_include_directories(zeno_zsfluid PRIVATE ../oldzenbase/include)
target_include_directories(zeno_zsfluid PRIVATE ../zenvdb/include)
target_compile_options(zeno_zsfluid
  PUBLIC        $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --diag_suppress=554>
)

if (${ZS_ENABLE_CUDA})
    enable_language(CUDA)

    target_compile_features(zeno_zsfem PRIVATE cuda_std_17)
    target_compile_options(zeno_zsfem 
      PRIVATE        $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr --default-stream=per-thread --use_fast_math -lineinfo --ptxas-options=-allow-expensive-optimizations=true>
      PUBLIC        $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --diag_suppress=554>
    )
    target_link_options(zeno_zsfem 
      PUBLIC       $<$<LINK_LANGUAGE:CUDA>:-dlto>
    )
    set_target_properties(zeno_zsfem
      PROPERTIES  CUDA_EXTENSIONS ON
                  CUDA_SEPARABLE_COMPILATION ON
                  LINKER_LANGUAGE CUDA
                  POSITION_INDEPENDENT_CODE ON
                  CUDA_ARCHITECTURES OFF
    )

    target_compile_features(zeno_zsfft PRIVATE cuda_std_17)
    target_compile_options(zeno_zsfft 
      PRIVATE        $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr --default-stream=per-thread --use_fast_math -lineinfo --ptxas-options=-allow-expensive-optimizations=true>
      PUBLIC        $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --diag_suppress=554>
    )
    target_link_options(zeno_zsfft 
      PUBLIC       $<$<LINK_LANGUAGE:CUDA>:-dlto>
    )
    set_target_properties(zeno_zsfft
      PROPERTIES  CUDA_EXTENSIONS ON
                  CUDA_SEPARABLE_COMPILATION ON
                  LINKER_LANGUAGE CUDA
                  POSITION_INDEPENDENT_CODE ON
                  CUDA_ARCHITECTURES OFF
    )
endif()

target_compile_features(zeno_zsfem PRIVATE cxx_std_17)
target_compile_features(zeno_zsfft PRIVATE cxx_std_17)

#simd compiler flags
if (WIN32)
else()
target_compile_options(zeno_zsfem PRIVATE "-march=native")
target_compile_options(zeno_zsfft PRIVATE "-march=native")
endif()
target_link_libraries(zeno_zsfem PRIVATE zensim)
target_link_libraries(zeno_zsfem PUBLIC zeno)
target_link_libraries(zeno_zsfem PRIVATE zeno_nodep)
target_link_libraries(zeno_zsfem PUBLIC ZFX)
target_include_directories(zeno_zsfem PRIVATE ../oldzenbase/include)
target_include_directories(zeno_zsfem PRIVATE ../zenvdb/include)

find_package(OpenMP REQUIRED)
target_link_libraries(zeno_zsfft PRIVATE zensim zpcomp OpenMP::OpenMP_CXX CUDA::cufft)
target_link_libraries(zeno_zsfft PUBLIC zeno)
target_link_libraries(zeno_zsfft PRIVATE zeno_nodep)
target_link_libraries(zeno_zsfft PUBLIC ZFX)
target_include_directories(zeno_zsfft PRIVATE ../oldzenbase/include)
target_include_directories(zeno_zsfft PRIVATE ../zenvdb/include)
