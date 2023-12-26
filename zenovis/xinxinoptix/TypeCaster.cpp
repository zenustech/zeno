#include "TypeCaster.h"

#include <cuda_fp16.h>
// #include <cuda_fp16.hpp>

ushort3 toHalf(float4 in)
{
      half hx = __float2half(in.x);
      half hy = __float2half(in.y);
      half hz = __float2half(in.z);

      return make_ushort3(*(unsigned short*)&(hx),
                          *(unsigned short*)&(hy),
                          *(unsigned short*)&(hz));
}