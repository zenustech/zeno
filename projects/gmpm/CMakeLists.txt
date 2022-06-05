cmake_minimum_required(VERSION 3.19)

# assume cuda
project(MPM CXX CUDA)
set(ZS_ENABLE_CUDA ON)
set(ZS_ENABLE_OPENVDB ON)
set(ZENSIM_INSTALL OFF)
set(ZS_BUILD_SHARED_LIBS OFF)
add_subdirectory(zecomp)

find_package(CUDAToolkit)

add_library(zshelper INTERFACE)
target_link_libraries(zshelper INTERFACE zensim zeno zeno_nodep ZFX)
target_include_directories(zshelper INTERFACE 
  ../oldzenbase/include
  ../zenvdb/include
)
target_compile_features(zshelper INTERFACE cxx_std_17 cuda_std_17)
if (WIN32)
else()
target_compile_options(zshelper 
  INTERFACE        $<$<COMPILE_LANGUAGE:CXX>:-march=native>
)
endif()

#add_library(zeno_zstest SHARED test.cpp)
#target_link_libraries(zeno_zstest PRIVATE zensim)

#add_library(zeno_cutest SHARED test.cu)
#target_link_libraries(zeno_cutest PRIVATE zshelper)

#set_target_properties(zeno_cutest
#      PROPERTIES  #CUDA_EXTENSIONS ON
#                  CUDA_SEPARABLE_COMPILATION ON
#                  LINKER_LANGUAGE CUDA
#                  POSITION_INDEPENDENT_CODE ON
#                  CUDA_ARCHITECTURES OFF
#    )

# zeno_zsfluid
add_library(zeno_zsfluid SHARED)
target_sources(zeno_zsfluid PRIVATE
  volume/VolumeTopology.cu
  volume/VolumeOperator.cu
  volume/Transfer.cu
)
set_source_files_properties(volume/VolumeOperator.cu volume/Transfer.cu PROPERTIES  LANGUAGE CUDA)
target_link_libraries(zeno_zsfluid PRIVATE zshelper)
target_compile_options(zeno_zsfluid
  PRIVATE        $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --display_error_number,--diag_suppress=implicit_return_from_non_void_function,--diag_suppress=2642,--diag_suppress=554>
)

# zeno_gipc
add_library(zeno_gipc SHARED)
target_sources(zeno_gipc PRIVATE
  gipc/ACCD.cu
  gipc/device_fem_data.cu
  gipc/femEnergy.cu
  gipc/GIPC.cu
  gipc/gpu_eigen_libs.cu
  gipc/load_mesh.cpp
  gipc/mlbvh.cu
  gipc/PCG_SOLVER.cu
  gipc/svd.cu
)
target_link_libraries(zeno_gipc PRIVATE zshelper)
target_compile_options(zeno_gipc
  PRIVATE        $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --display_error_number,--diag_suppress=implicit_return_from_non_void_function,--diag_suppress=2642,--diag_suppress=554>
)
set_target_properties(zeno_gipc
    PROPERTIES  CUDA_EXTENSIONS ON
                CUDA_SEPARABLE_COMPILATION ON
                CUDA_RESOLVE_DEVICE_SYMBOLS ON
                CUDA_ARCHITECTURES "60"
                )

add_library(zeno_mpm SHARED)
target_sources(zeno_mpm PRIVATE
  utils/IndexBuckets.cu
  mpm/Boundary.cu
  mpm/Generation.cu
  mpm/Generation.cpp
  mpm/Transfer.cu
  mpm/MPMPipeline.cu
  mpm/Maintenance.cu
  mpm/Temp.cu
)
target_link_libraries(zeno_mpm PRIVATE zshelper)
if (NOT MSVC)
target_sources(zeno_mpm PRIVATE 
  mpm/PW.cu
  mpm/PNW.cu
  mpm/PPW.cu
)
endif()
target_compile_options(zeno_mpm 
  PRIVATE        $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --display_error_number,--diag_suppress=implicit_return_from_non_void_function,--diag_suppress=2642,--diag_suppress=554>
)

add_library(zeno_zsfft SHARED)
target_sources(zeno_zsfft PRIVATE
  oceanfft/Ocean.cu
)
target_compile_options(zeno_zsfft # MSVC, GNU, Clang, Intel
  PUBLIC   $<$<COMPILE_LANGUAGE:CXX>: $<IF:$<CXX_COMPILER_ID:MSVC>, /openmp:llvm /openmp:experimental, -fopenmp $<IF:$<CXX_COMPILER_ID:Clang>, , > >>
)
    target_compile_options(zeno_zsfft 
      PRIVATE        $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr --default-stream=per-thread -lineinfo --ptxas-options=-allow-expensive-optimizations=true -Xcudafe --display_error_number,--diag_suppress=implicit_return_from_non_void_function,--diag_suppress=2642,--diag_suppress=554>
    )
    target_link_options(zeno_zsfft 
      PUBLIC       $<$<LINK_LANGUAGE:CUDA>:-dlto>
    )
    set_target_properties(zeno_zsfft
      PROPERTIES  #CUDA_EXTENSIONS ON
                  #CUDA_SEPARABLE_COMPILATION ON
                  #LINKER_LANGUAGE CUDA
                  #POSITION_INDEPENDENT_CODE ON
                  CUDA_ARCHITECTURES OFF
    )

find_package(OpenMP REQUIRED)
target_link_libraries(zeno_zsfft PRIVATE zshelper zpc zpcomp OpenMP::OpenMP_CXX CUDA::cufft)


add_library(zeno_zsfem SHARED)
target_sources(zeno_zsfem PRIVATE
  fem/BoundaryCondition.cu
  fem/Collisions.cu
  fem/TimeStepping.cu
  fem/Check.cu
  fem/Generation.cpp
  fem/test.cpp
  fem/MeshIO.cpp
  fem/Fiber.cpp
  fem/Integrator.cpp
  fem/QuasiStaticStepping.cu
  fem/BaryCentricInterpolator.cu
)

if (${ZS_ENABLE_CUDA})
    enable_language(CUDA)

    target_compile_features(zeno_zsfem PRIVATE cuda_std_17)
    target_compile_options(zeno_zsfem 
      PRIVATE        $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr --default-stream=per-thread -lineinfo --ptxas-options=-allow-expensive-optimizations=true -Xcudafe --display_error_number,--diag_suppress=implicit_return_from_non_void_function,--diag_suppress=2642,--diag_suppress=554>
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
endif()

target_link_libraries(zeno_zsfem PRIVATE zshelper)
