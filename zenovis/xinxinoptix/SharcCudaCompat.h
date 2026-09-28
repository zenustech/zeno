#pragma once

#include <cuda_fp16.h>
#include <cuda/cstdint.h>
#include <sutil/vec_math.h>

using uint = unsigned int;
using float16_t4 = ushort4;

static __forceinline__ __device__ float2 sharcMakeFloat2(float x, float y)
{
    return make_float2(x, y);
}

static __forceinline__ __device__ float2 sharcMakeFloat2(const uint2& value)
{
    return make_float2(float(value.x), float(value.y));
}

static __forceinline__ __device__ float2 sharcMakeFloat2(const int2& value)
{
    return make_float2(float(value.x), float(value.y));
}

static __forceinline__ __device__ float3 sharcMakeFloat3(float x, float y, float z)
{
    return make_float3(x, y, z);
}

static __forceinline__ __device__ float3 sharcMakeFloat3(float value)
{
    return make_float3(value);
}

static __forceinline__ __device__ float3 sharcMakeFloat3(const float3& value)
{
    return value;
}

static __forceinline__ __device__ float3 sharcMakeFloat3(const float4& value)
{
    return make_float3(value.x, value.y, value.z);
}

static __forceinline__ __device__ float3 sharcMakeFloat3(const uint3& value)
{
    return make_float3(value);
}

static __forceinline__ __device__ float3 sharcMakeFloat3(const uint4& value)
{
    return make_float3(float(value.x), float(value.y), float(value.z));
}

static __forceinline__ __device__ float4 sharcMakeFloat4(float x, float y, float z, float w)
{
    return make_float4(x, y, z, w);
}

static __forceinline__ __device__ float4 sharcMakeFloat4(const float3& value, float w)
{
    return make_float4(value, w);
}

static __forceinline__ __device__ float4 sharcMakeFloat4(const float4& value)
{
    return value;
}

static __forceinline__ __device__ float4 sharcMakeFloat4(const uint4& value)
{
    return make_float4(value);
}

static __forceinline__ __device__ float4 sharcMakeFloat4(const int4& value)
{
    return make_float4(
        float(value.x), float(value.y), float(value.z), float(value.w));
}

static __forceinline__ __device__ int2 sharcMakeInt2(float2 value)
{
    return make_int2(value);
}

static __forceinline__ __device__ int2 sharcMakeInt2(int x, int y)
{
    return make_int2(x, y);
}

static __forceinline__ __device__ int3 sharcMakeInt3(float3 value)
{
    return make_int3(value);
}

static __forceinline__ __device__ int3 sharcMakeInt3(int x, int y, int z)
{
    return make_int3(x, y, z);
}

static __forceinline__ __device__ int4 sharcMakeInt4(int x, int y, int z, int w)
{
    return make_int4(x, y, z, w);
}

static __forceinline__ __device__ int4 sharcMakeInt4(const int3& value, int w)
{
    return make_int4(value, w);
}

static __forceinline__ __device__ int4 sharcMakeInt4(const float4& value)
{
    return make_int4(
        int(value.x), int(value.y), int(value.z), int(value.w));
}

static __forceinline__ __device__ uint3 sharcMakeUint3(float3 value)
{
    return make_uint3(value);
}

static __forceinline__ __device__ uint4 sharcMakeUint4(const int4& value)
{
    return make_uint4(
        uint(value.x), uint(value.y), uint(value.z), uint(value.w));
}

static __forceinline__ __device__ uint4 sharcMakeUint4(
    uint x, uint y, uint z, uint w)
{
    return make_uint4(x, y, z, w);
}

static __forceinline__ __device__ ushort4 sharcMakeFloat16x4(
    float x, float y, float z, float w)
{
    return make_ushort4(
        __half_as_ushort(__float2half_rn(x)),
        __half_as_ushort(__float2half_rn(y)),
        __half_as_ushort(__float2half_rn(z)),
        __half_as_ushort(__float2half_rn(w)));
}

static __forceinline__ __device__ ushort4 sharcMakeFloat16x4(const float4& value)
{
    return sharcMakeFloat16x4(value.x, value.y, value.z, value.w);
}

static __forceinline__ __device__ float4 sharcMakeFloat4(const ushort4& value)
{
    return make_float4(
        __half2float(__ushort_as_half(value.x)),
        __half2float(__ushort_as_half(value.y)),
        __half2float(__ushort_as_half(value.z)),
        __half2float(__ushort_as_half(value.w)));
}

static __forceinline__ __device__ float rcp(float value)
{
    return 1.0f / value;
}

static __forceinline__ __device__ float3 rcp(const float3& value)
{
    return make_float3(1.0f / value.x, 1.0f / value.y, 1.0f / value.z);
}

static __forceinline__ __device__ float sharcSaturate(float value)
{
    return clamp(value, 0.0f, 1.0f);
}

static __forceinline__ __device__ float lerp(int lhs, int rhs, float amount)
{
    return float(lhs) + amount * float(rhs - lhs);
}

static __forceinline__ __device__ uint clamp(uint value, int lower, int upper)
{
    return min(max(value, uint(lower)), uint(upper));
}

static __forceinline__ __device__ float3 min(const float3& lhs, const float3& rhs)
{
    return fminf(lhs, rhs);
}

static __forceinline__ __device__ float3 max(const float3& lhs, const float3& rhs)
{
    return fmaxf(lhs, rhs);
}

static __forceinline__ __device__ float4 min(const float4& lhs, const float4& rhs)
{
    return fminf(lhs, rhs);
}

static __forceinline__ __device__ int3 operator<<(const int3& value, int shift)
{
    return make_int3(value.x << shift, value.y << shift, value.z << shift);
}

static __forceinline__ __device__ int3 operator>>(const int3& value, int shift)
{
    return make_int3(value.x >> shift, value.y >> shift, value.z >> shift);
}

static __forceinline__ __device__ float3 operator/(const int3& value, float divisor)
{
    return make_float3(value) / divisor;
}

static __forceinline__ __device__ float3 operator+(const int3& value, float offset)
{
    return make_float3(value) + offset;
}

static __forceinline__ __device__ float3 operator*(const int3& value, float scale)
{
    return make_float3(value) * scale;
}

static __forceinline__ __device__ uint f32tof16(float value)
{
    return uint(__half_as_ushort(__float2half_rn(value)));
}

static __forceinline__ __device__ float f16tof32(uint value)
{
    return __half2float(__ushort_as_half(static_cast<unsigned short>(value)));
}

static __forceinline__ __device__ uint countbits(uint value)
{
    return __popc(value);
}

template <typename T>
static __forceinline__ __device__ void sharcInterlockedAdd(T& destination, T value)
{
    atomicAdd(&destination, value);
}

template <typename T>
static __forceinline__ __device__ void sharcInterlockedExchange(
    T& destination, T value, T& original)
{
    original = atomicExch(&destination, value);
}

static __forceinline__ __device__ void sharcInterlockedCompareExchange(
    uint64_t& destination, uint64_t compare, uint64_t value, uint64_t& original)
{
    original = uint64_t(atomicCAS(
        reinterpret_cast<unsigned long long*>(&destination),
        static_cast<unsigned long long>(compare),
        static_cast<unsigned long long>(value)));
}

static __forceinline__ __device__ void sharcInterlockedCompareExchange(
    uint& destination, uint compare, uint value, uint& original)
{
    original = atomicCAS(&destination, compare, value);
}

#define float2(...) sharcMakeFloat2(__VA_ARGS__)
#define float3(...) sharcMakeFloat3(__VA_ARGS__)
#define float4(...) sharcMakeFloat4(__VA_ARGS__)
#define int2(...) sharcMakeInt2(__VA_ARGS__)
#define int3(...) sharcMakeInt3(__VA_ARGS__)
#define int4(...) sharcMakeInt4(__VA_ARGS__)
#define uint3(...) sharcMakeUint3(__VA_ARGS__)
#define uint4(...) sharcMakeUint4(__VA_ARGS__)
#define float16_t4(...) sharcMakeFloat16x4(__VA_ARGS__)
#define saturate sharcSaturate

#define RW_STRUCTURED_BUFFER(name, type) type* name
#define BUFFER_AT_OFFSET(name, offset) name[offset]
#define HASH_GRID_CONST static constexpr
#define HASH_GRID_LOOP_ATTR
#define InterlockedAdd(destination, value) \
    sharcInterlockedAdd(destination, value)
#define InterlockedExchange(destination, value, original) \
    sharcInterlockedExchange(destination, value, original)
#define InterlockedCompareExchange(destination, compare, value, original) \
    sharcInterlockedCompareExchange(destination, compare, value, original)
