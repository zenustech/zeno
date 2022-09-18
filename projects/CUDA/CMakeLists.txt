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
SET(CMAKE_CUDA_ARCHITECTURES OFF)
project(ZENO_CUDA CXX CUDA)
set(ZS_ENABLE_CUDA ON)
set(ZS_ENABLE_OPENVDB ON)
set(ZENSIM_INSTALL OFF)
set(ZS_BUILD_SHARED_LIBS OFF)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)
find_package(CUDAToolkit REQUIRED COMPONENTS cufft REQUIRED)

add_subdirectory(zpc)

add_library(zshelper INTERFACE)
target_include_directories(zshelper INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
target_link_libraries(zshelper INTERFACE zensim ZFX)
target_compile_features(zshelper INTERFACE cuda_std_17)
target_link_options(zshelper
  INTERFACE $<$<LINK_LANGUAGE:CUDA>:-dlto>
)
target_link_libraries(zeno PRIVATE zshelper)

#[=[
target_sources(zeno PRIVATE
  test1.cu
)
#]=]
set_target_properties(zeno
  PROPERTIES # CUDA_EXTENSIONS ON
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_ARCHITECTURES OFF
)

# wrangler
target_sources(zeno PRIVATE
  utils/IndexBuckets.cu
  wrangle/PW.cu
  wrangle/PNW.cu
  wrangle/PNBVHW.cu
  wrangle/PPW.cu
)

target_sources(zeno PRIVATE
  test1.cu
)
