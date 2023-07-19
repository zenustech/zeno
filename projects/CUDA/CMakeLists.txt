cmake_minimum_required(VERSION 3.19)

if(NOT ZENO_WITH_zenvdb OR NOT ZENO_WITH_ZenoFX)
  message(FATAL_ERROR "zenvdb and ZenoFX must be ON when CUDA is ON! "
    "Please specify: -DZENO_WITH_zenvdb:BOOL=ON -DZENO_WITH_ZenoFX:BOOL=ON")
endif()

if(ZENO_SYSTEM_OPENVDB)
  message(FATAL_ERROR "ZENO_SYSTEM_OPENVDB must be OFF when CUDA is ON! "
    "Please specify: -DZENO_SYSTEM_OPENVDB:BOOL=ON")
endif()

# assume cuda
project(ZENO_CUDA CXX CUDA)
set(ZS_ENABLE_CUDA ON)
set(ZS_ENABLE_OPENVDB ON)
set(ZENSIM_INSTALL OFF)
set(ZS_BUILD_SHARED_LIBS OFF)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)
find_package(CUDAToolkit REQUIRED COMPONENTS cufft REQUIRED)
find_package(Eigen3 REQUIRED)

add_subdirectory(zpc)

add_library(zshelper INTERFACE)
target_include_directories(zshelper INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
target_link_libraries(zshelper INTERFACE zensim ZFX)
target_compile_features(zshelper INTERFACE cuda_std_17)
target_compile_options(zpc_cuda_deps
  INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_${ZS_CUDA_ARCH}>
)

target_link_options(zeno
  PRIVATE $<DEVICE_LINK:-dlto>
)
target_link_libraries(zeno PRIVATE Eigen3::Eigen)
target_link_libraries(zeno PRIVATE zshelper)

#[=[
target_sources(zeno PRIVATE
  test1.cu
  test.cpp
)
#]=]
set_target_properties(zeno
  PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_ARCHITECTURES ${ZS_CUDA_ARCH}
)

# closest point query (from ingo wald)
target_sources(zeno PRIVATE
  iw_query/Query.cpp
  iw_query/bvh.cpp
  iw_query/distanceQueries.cpp
)

# wrangler
target_sources(zeno PRIVATE
  wrangle/PW.cu
  wrangle/P2W.cu
  wrangle/PNW.cu
  wrangle/PNBVHW.cu
  wrangle/PPW.cu
  wrangle/TVW.cu
  wrangle/VW.cu
)

# utils
target_sources(zeno PRIVATE
  Structures.cu
  utils/Conversion.cpp
  utils/Primitives.cpp
  utils/IndexBuckets.cu
  utils/Primitives.cu
  utils/TopoUtils.cu
  utils/Noise.cu
  utils/Groom.cpp
)

add_subdirectory(remesh)