#ifndef __SORT_KERNELS_CUH_
#define __SORT_KERNELS_CUH_
#include <stdint.h>

namespace zs {

  template <typename ElemType, unsigned int Size, typename IndexType>
  __global__ void gather_entry(int num, const AttribPort<ElemType, Size> _from,
                               AttribPort<ElemType, Size> _to, const IndexType* _prev) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num) return;
    for (int i = 0; i < Size; ++i) _to[i][idx] = _from[i][_prev[idx]];
  }

  template <typename ElemType, unsigned int Size, typename IndexType>
  __global__ void scatter_entry(int num, const AttribPort<ElemType, Size> _from,
                                AttribPort<ElemType, Size> _to, const IndexType* _next) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num) return;
    for (int i = 0; i < Size; ++i) _to[i][_next[idx]] = _from[i][idx];
  }

}  // namespace zs

#endif