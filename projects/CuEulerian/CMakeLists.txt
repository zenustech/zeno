cmake_minimum_required(VERSION 3.19)

if(NOT ZENO_WITH_CUDA)
  message(FATAL_ERROR "CUDA must be ON when CuOcean is ON! "
    "Please specify: -DZENO_WITH_CUDA:BOOL=ON -DZENO_WITH_zenvdb:BOOL=ON -DZENO_WITH_ZenoFX:BOOL=ON")
endif()

target_link_libraries(zeno PRIVATE zshelper CUDA::cufft)

# pbd
target_sources(zeno PRIVATE
  oceanfft/Ocean.cu
)

# fluid
target_sources(zeno PRIVATE
  volume/VolumeTopology.cu
  volume/VolumeOperator.cu
  volume/Transfer.cu
  volume/Conversion.cu
)

# shallow water
target_sources(zeno PRIVATE
  swe/FDGather.cu
)

# level set function
target_sources(zeno PRIVATE
  levelset/Grid_creator.cu
  levelset/Reinitialization.cu
)