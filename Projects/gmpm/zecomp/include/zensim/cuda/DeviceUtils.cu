#include "DeviceUtils.cuh"

namespace zs {

  template <int NumPageBits> __device__ int Retrieve_Block_Local_Offset(
      int level, uint64_t blockOffset) {  ///< the level number starts from 0
    return (blockOffset >> (NumPageBits + level * 3)) & 7;
  }

}  // namespace zs
