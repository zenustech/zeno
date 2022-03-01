
#ifndef DOUBLE_MATH_H
#define DOUBLE_MATH_H


#include "cuda_runtime.h"


inline  __device__ __host__ double3 operator-(double3 &a)
{
    return make_double3(-a.x, -a.y, -a.z);
}

inline  __device__ __host__ double3 operator+(double3 &a)
{
    return make_double3(-a.x, -a.y, -a.z);
}

inline  __device__ __host__ double3 operator+(double3 a, double3 b)
{
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline  __device__ __host__ double3 operator-(double3 a, double3 b)
{
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline  __device__ __host__ double3 operator/(double3 a, double b)
{
    return make_double3(a.x/b, a.y/b, a.z/b);
}

inline  __device__ __host__ double3 operator*(double3 a, double b)
{
    return make_double3(a.x*b, a.y*b, a.z*b);
}

inline  __device__ __host__ double3 operator*(double3 a, double3 b)
{
    return make_double3(a.x*b.x, a.y*b.y, a.z*b.z);
}

inline  __device__ __host__ double3 operator*(double b, double3 a)
{
    return make_double3(a.x*b, a.y*b, a.z*b);
}

inline  __device__ __host__ double4 operator+(double4 a, double4 b)
{
    return make_double4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + a.w);
}




inline  __device__ __host__ int toInt(char a){
    return a;
}

inline  __device__ __host__ int2 toInt(char2 a){
    return make_int2(a.x, a.y);
}

inline  __device__ __host__ int3 toInt(char3 a){
    return make_int3(a.x, a.y, a.z);
}

inline  __device__ __host__ int4 toInt(char4 a){
    return make_int4(a.x, a.y, a.z, a.w);
}


#endif