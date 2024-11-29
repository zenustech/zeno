#include "TypeCaster.h"

#include <cuda_fp16.h>
#include <vector_functions.hpp>
// #include <cuda_fp16.hpp>

ushort2 toHalf(const float2& in)
{
    half hx = __float2half(in.x);
    half hy = __float2half(in.y);

    return {*(unsigned short*)&(hx),
            *(unsigned short*)&(hy)};
}

ushort3 toHalf(const float3& in)
{
    half hx = __float2half(in.x);
    half hy = __float2half(in.y);
    half hz = __float2half(in.z);

    return {*(unsigned short*)&(hx),
            *(unsigned short*)&(hy),
            *(unsigned short*)&(hz)};
}

ushort3 toHalf(const float4& in)
{
    return toHalf( float3 {in.x, in.y, in.z} );
}

float3 toFloat(ushort3 in) {
    half x = reinterpret_cast<half&>(in.x);
    half y = reinterpret_cast<half&>(in.y);
    half z = reinterpret_cast<half&>(in.z);
    return {
        __half2float(x),
        __half2float(y),
        __half2float(z),
    };
}
float toFloat(ushort1 in) {
    half x = reinterpret_cast<half&>(in);
    return __half2float(x);
}