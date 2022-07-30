cmake_minimum_required(VERSION 3.19)

if (NOT ZENO_WITH_zenvdb OR NOT ZENO_WITH_ZenoFX)
    message(FATAL_ERROR "zenvdb and ZenoFX must be ON when CUDA is ON! "
        "Please specify: -DZENO_WITH_zenvdb:BOOL=ON -DZENO_WITH_ZenoFX:BOOL=ON")
endif()

# assume cuda
SET(CMAKE_CUDA_ARCHITECTURES OFF)
project(ZENO_CUDA CXX CUDA)
set(ZS_ENABLE_CUDA ON)
set(ZS_ENABLE_OPENVDB ON)
set(ZENSIM_INSTALL OFF)
set(ZS_BUILD_SHARED_LIBS OFF)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)
find_package(CUDAToolkit)

add_subdirectory(zpc)

add_library(zshelper INTERFACE)
target_link_libraries(zshelper INTERFACE zensim ZFX)
target_compile_features(zshelper INTERFACE cxx_std_17 cuda_std_17)

if(WIN32)
else()
  # BATE: nono wxl stop using -march=native!
  # the GitHub CI machine has AVX512, but most people doesn't have it
  # this would cause one who doesn't have AVX512 cannot start zeno at all!
  # so this is anti-deployment, please stop.
  #target_compile_options(zshelper
    #INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-march=native>
  #)
endif()

target_compile_features(zeno PRIVATE cuda_std_17)
target_compile_options(zeno
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr --default-stream=per-thread -lineinfo --ptxas-options=-allow-expensive-optimizations=true -Xcudafe --display_error_number,--diag_suppress=implicit_return_from_non_void_function,--diag_suppress=2642,--diag_suppress=554>
)
target_link_options(zeno
  PUBLIC $<$<LINK_LANGUAGE:CUDA>:-dlto>
)
target_link_libraries(zeno PRIVATE zshelper CUDA::cufft)

#[=[
set_target_properties(zeno
  PROPERTIES  CUDA_EXTENSIONS ON
  CUDA_SEPARABLE_COMPILATION ON
# LINKER_LANGUAGE CUDA
  POSITION_INDEPENDENT_CODE ON
# CUDA_ARCHITECTURES OFF
)
#]=]


# fluid
target_sources(zeno PRIVATE
  volume/VolumeTopology.cu
  volume/VolumeOperator.cu
  volume/Transfer.cu
  volume/Conversion.cu
)
# mpm
target_sources(zeno PRIVATE
  utils/IndexBuckets.cu
  mpm/Boundary.cu
  mpm/Generation.cu
  mpm/Generation.cpp
  mpm/Transfer.cu
  mpm/MPMPipeline.cu
  mpm/Maintenance.cu
  mpm/Temp.cu
)
# fem
target_sources(zeno PRIVATE
  fem/BoundaryCondition.cu
  fem/Collisions.cu
  #fem/TimeStepping.cu
  fem/Check.cu
  fem/Generation.cpp
  fem/test.cpp
  fem/MeshIO.cpp
  fem/Fiber.cpp
  fem/Integrator.cpp
  fem/QuasiStaticStepping.cu
  # fem/BaryCentricInterpolator.cu
  fem/Codim.cu
  #fem/TestAugLag.cu
  fem/FastQuasiStaticStepping.cu

  fem/FleshQuasiStepping.cu
)
# geometry
target_sources(zeno PRIVATE
  geometry/file_parser/read_vtk_mesh.hpp
  geometry/file_parser/write_vtk_unstructured_mesh.hpp
  geometry/file_parser/vtk_types.hpp  
  geometry/linear_system/mfcg.hpp
  geometry/linear_system/active_set.hpp
  geometry/kernel/laplace_matrix.hpp
  geometry/kernel/gradient_field.hpp
  geometry/kernel/bary_centric_weights.hpp
  geometry/SolveLaplacian.cu
  geometry/VectorField.cu
  geometry/SetupSkinningWeight.cu
  geometry/Skinning.cu
  geometry/BaryCentricInterpolator.cu
  geometry/BiharmonicBoundedWeight.cu
  geometry/MeshIO.cu
)
# ocean
#target_sources(zeno PRIVATE
#  oceanfft/Ocean.cu
#)

# wrangler
if(NOT MSVC)
  target_sources(zeno PRIVATE
    mpm/PW.cu
    mpm/PNW.cu
    mpm/PPW.cu
  )
endif()
