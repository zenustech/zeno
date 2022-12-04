cmake_minimum_required(VERSION 3.18)

set_target_properties(zeno
  PROPERTIES # CUDA_EXTENSIONS ON
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_ARCHITECTURES OFF
)

# rxmesh
target_sources(zeno PRIVATE
  attribute.h
  context.h
  handle.h
  iterator.cuh
  patch_info.h
  rxmesh_static.h
  rxmesh.h
  rxmesh.cu
  RXMeshNode.cu
  types.h
)

set(CMAKE_CUDA_FLAGS "--expt-extended-lambda --expt-relaxed-constexpr")

add_subdirectory(kernels)
add_subdirectory(patcher)
add_subdirectory(utils)
