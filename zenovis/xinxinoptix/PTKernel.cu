//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
#include <optix.h>

enum RayType
{
    RAY_TYPE_RADIANCE  = 0,
    RAY_TYPE_OCCLUSION = 1,
    RAY_TYPE_COUNT
};


struct ParallelogramLight
{
    float3 corner;
    float3 v1, v2;
    float3 normal;
    float3 emission;
};


struct Params
{
    unsigned int subframe_index;
    float4*      accum_buffer;
    uchar4*      frame_buffer;
    unsigned int width;
    unsigned int height;
    unsigned int samples_per_launch;

    float3       eye;
    float3       U;
    float3       V;
    float3       W;

    ParallelogramLight     light; // TODO: make light list
    OptixTraversableHandle handle;
};


struct RayGenData
{
};


struct MissData
{
    float4 bg_color;
};


struct HitGroupData
{
    float3  emission_color;
    float3  diffuse_color;
    float4* vertices;
};
#include "random.h"

#include <sutil/vec_math.h>
#include <cuda/helpers.h>

extern "C" {
__constant__ Params params;
}
struct vec2{
    float x, y;
    vec2(const vec2 &_v)
    {
        x = _v.x;
        y = _v.y;
    }
    vec2(float _x, float _y)
    {
        x = _x; y = _y;
    }
    vec2(float _x)
    {
        x = _x; y = _x;
    }
};
struct vec3{
    float x, y, z;
    vec3(const vec3 &_v)
    {
        x = _v.x;
        y = _v.y;
        z = _v.z;
    }
    vec3(float _x, float _y, float _z)
    {
        x = _x; y = _y; z = _z;
    }
    vec3(float _x)
    {
        x = _x; y = _x; z = _x;
    }
};

struct vec4{
    float x, y, z, w;
    vec4(const vec4 &_v)
    {
        x = _v.x; z = _v.z;
        y = _v.y; w = _v.w;
    }
    vec4(float _x, float _y, float _z, float _w)
    {
        x = _x; y = _y; z = _z, w = _w;
    }
    vec4(float _x)
    {
        x = _x; y = _x; z = _x; w = _x;
    }
};
//////////////// + - * /////////////////////////////////////////////
__forceinline__ __device__ vec2 operator+(vec2 a, float b)
{
    return vec2(a.x+b, a.y+b);
}
__forceinline__ __device__ vec3 operator+(vec3 a, float b)
{
    return vec3(a.x+b, a.y+b, a.z+b);
}
__forceinline__ __device__ vec4 operator+(vec4 a, float b)
{
    return vec4(a.x+b, a.y+b, a.z+b, a.w+b);
}
__forceinline__ __device__ vec2 operator+(float b, vec2 a)
{
    return vec2(a.x+b, a.y+b);
}
__forceinline__ __device__ vec3 operator+(float b, vec3 a)
{
    return vec3(a.x+b, a.y+b, a.z+b);
}
__forceinline__ __device__ vec4 operator+(float b, vec4 a)
{
    return vec4(a.x+b, a.y+b, a.z+b, a.w+b);
}
__forceinline__ __device__ vec2 operator+(vec2 b, vec2 a)
{
    return vec2(a.x+b.x, a.y+b.y);
}
__forceinline__ __device__ vec3 operator+(vec3 b, vec3 a)
{
    return vec3(a.x+b.x, a.y+b.y, a.z+b.z);
}
__forceinline__ __device__ vec4 operator+(vec4 b, vec4 a)
{
    return vec4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}

__forceinline__ __device__ vec2 operator-(vec2 a, vec2 b)
{
    return vec2(a.x-b.x, a.y-b.y);
}
__forceinline__ __device__ vec3 operator-(vec3 a, vec3 b)
{
    return vec3(a.x-b.x, a.y-b.y, a.z-b.z);
}
__forceinline__ __device__ vec4 operator-(vec4 a, vec4 b)
{
    return vec4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w);
}

__forceinline__ __device__ vec2 operator-(vec2 a, float b)
{
    return vec2(a.x-b, a.y-b);
}
__forceinline__ __device__ vec3 operator-(vec3 a, float b)
{
    return vec3(a.x-b, a.y-b, a.z-b);
}
__forceinline__ __device__ vec4 operator-(vec4 a, float b)
{
    return vec4(a.x-b, a.y-b, a.z-b, a.w-b);
}
__forceinline__ __device__ vec2 operator-(float b, vec2 a)
{
    return vec2(b-a.x, b-a.y);
}
__forceinline__ __device__ vec3 operator-(float b, vec3 a)
{
    return vec3(b-a.x, b-a.y, b-a.z);
}
__forceinline__ __device__ vec4 operator-(float b, vec4 a)
{
    return vec4(b-a.x, b-a.y, b-a.z, b-a.w);
}


__forceinline__ __device__ vec2 operator*(vec2 a, vec2 b)
{
    return vec2(a.x*b.x, a.y*b.y);
}
__forceinline__ __device__ vec3 operator*(vec3 a, vec3 b)
{
    return vec3(a.x*b.x, a.y*b.y, a.z*b.z);
}
__forceinline__ __device__ vec4 operator*(vec4 a, vec4 b)
{
    return vec4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w);
}

__forceinline__ __device__ vec2 operator*(vec2 a, float b)
{
    return vec2(a.x*b, a.y*b);
}
__forceinline__ __device__ vec3 operator*(vec3 a, float b)
{
    return vec3(a.x*b, a.y*b, a.z*b);
}
__forceinline__ __device__ vec4 operator*(vec4 a, float b)
{
    return vec4(a.x*b, a.y*b, a.z*b, a.w*b);
}
__forceinline__ __device__ vec2 operator*(float b, vec2 a)
{
    return vec2(a.x*b, a.y*b);
}
__forceinline__ __device__ vec3 operator*(float b, vec3 a)
{
    return vec3(a.x*b, a.y*b, a.z*b);
}
__forceinline__ __device__ vec4 operator*(float b, vec4 a)
{
    return vec4(a.x*b, a.y*b, a.z*b, a.w*b);
}

__forceinline__ __device__ vec2 operator/(vec2 a, float b)
{
    return vec2(a.x/b, a.y/b);
}
__forceinline__ __device__ vec3 operator/(vec3 a, float b)
{
    return vec3(a.x/b, a.y/b, a.z/b);
}
__forceinline__ __device__ vec4 operator/(vec4 a, float b)
{
    return vec4(a.x/b, a.y/b, a.z/b, a.w/b);
}
__forceinline__ __device__ vec2 operator/(vec2 a, vec2 b)
{
    return vec2(a.x/b.x, a.y/b.y);
}
__forceinline__ __device__ vec3 operator/(vec3 a, vec3 b)
{
    return vec3(a.x/b.x, a.y/b.y, a.z/b.z);
}
__forceinline__ __device__ vec4 operator/(vec4 a, vec4 b)
{
    return vec4(a.x/b.x, a.y/b.y, a.z/b.z, a.w/b.w);
}
////////////////end of + - * /////////////////////////////////////////////


/////////////////trig func//////////////////////////////////////////////////
__forceinline__ __device__ vec2 sin(vec2 a)
{
    return vec2(sinf(a.x), sinf(a.y));
}
__forceinline__ __device__ vec3 sin(vec3 a)
{
    return vec3(sinf(a.x), sinf(a.y), sinf(a.z));
} 
__forceinline__ __device__ vec4 sin(vec4 a)
{
    return vec4(sinf(a.x), sinf(a.y), sinf(a.z), sinf(a.w));
}


__forceinline__ __device__ vec2 cos(vec2 a)
{
    return vec2(cosf(a.x), cosf(a.y));
}
__forceinline__ __device__ vec3 cos(vec3 a)
{
    return vec3(cosf(a.x), cosf(a.y), cosf(a.z));
} 
__forceinline__ __device__ vec4 cos(vec4 a)
{
    return vec4(cosf(a.x), cosf(a.y), cosf(a.z), cosf(a.w));
}


__forceinline__ __device__ vec2 tan(vec2 a)
{
    return vec2(tanf(a.x), tanf(a.y));
}
__forceinline__ __device__ vec3 tan(vec3 a)
{
    return vec3(tanf(a.x), tanf(a.y), tanf(a.z));
} 
__forceinline__ __device__ vec4 tan(vec4 a)
{
    return vec4(tanf(a.x), tanf(a.y), tanf(a.z), tanf(a.w));
}

__forceinline__ __device__ vec2 asin(vec2 a)
{
    return vec2(asinf(a.x), asinf(a.y));
}
__forceinline__ __device__ vec3 asin(vec3 a)
{
    return vec3(asinf(a.x), asinf(a.y), asinf(a.z));
} 
__forceinline__ __device__ vec4 asin(vec4 a)
{
    return vec4(asinf(a.x), asinf(a.y), asinf(a.z), asinf(a.w));
}

__forceinline__ __device__ vec2 acos(vec2 a)
{
    return vec2(acosf(a.x), acosf(a.y));
}
__forceinline__ __device__ vec3 acos(vec3 a)
{
    return vec3(acosf(a.x), acosf(a.y), acosf(a.z));
} 
__forceinline__ __device__ vec4 acos(vec4 a)
{
    return vec4(acosf(a.x), acosf(a.y), acosf(a.z), acosf(a.w));
}

__forceinline__ __device__ vec2 atan(vec2 a)
{
    return vec2(atanf(a.x), atanf(a.y));
}
__forceinline__ __device__ vec3 atan(vec3 a)
{
    return vec3(atanf(a.x), atanf(a.y), atanf(a.z));
} 
__forceinline__ __device__ vec4 atan(vec4 a)
{
    return vec4(atanf(a.x), atanf(a.y), atanf(a.z), atanf(a.w));
}

__forceinline__ __device__ vec2 atan(vec2 &a, vec2 &b)
{
    return atan(a/b);
}
__forceinline__ __device__ vec2 atan(vec2 &a, float &b)
{
    return atan(a/b);
}
__forceinline__ __device__ vec3 atan(vec3 &a, vec3 &b)
{
    return atan(a/b);
}
__forceinline__ __device__ vec3 atan(vec3 &a, float &b)
{
    return atan(a/b);
}
__forceinline__ __device__ vec4 atan(vec4 &a, vec4 &b)
{
    return atan(a/b);
}
__forceinline__ __device__ vec4 atan(vec4 &a, float &b)
{
    return atan(a/b);
}
/////////////////end of trig func//////////////////////////////////////////////////


////////////////exponential////////////////////////////////////////////////////////
__forceinline__ __device__ vec2 pow(vec2 a, vec2 b)
{
    return vec2(powf(a.x, b.x), powf(a.y, b.y));
}
__forceinline__ __device__ vec2 pow(vec2 a, float b)
{
    return vec2(powf(a.x, b), powf(a.y, b));
}

__forceinline__ __device__ vec3 pow(vec3 a, vec3 b)
{
    return vec3(powf(a.x, b.x), powf(a.y, b.y), powf(a.z, b.z));
}
__forceinline__ __device__ vec3 pow(vec3 a, float b)
{
    return vec3(powf(a.x, b), powf(a.y, b), powf(a.z, b));
}

__forceinline__ __device__ vec4 pow(vec4 a, vec4 b)
{
    return vec4(powf(a.x, b.x), powf(a.y, b.y), powf(a.z, b.z), powf(a.w, b.w));
}
__forceinline__ __device__ vec4 pow(vec4 a, float b)
{
    return vec4(powf(a.x, b), powf(a.y, b), powf(a.z, b), powf(a.w, b));
}

__forceinline__ __device__ vec2 exp(vec2 a)
{
    return vec2(expf(a.x), expf(a.y));
}

__forceinline__ __device__ vec3 exp(vec3 a)
{
    return vec3(expf(a.x), expf(a.y), expf(a.z));
}

__forceinline__ __device__ vec4 exp(vec4 a)
{
    return vec4(expf(a.x), expf(a.y), expf(a.z), expf(a.w));
}


__forceinline__ __device__ vec2 log(vec2 a)
{
    return vec2(logf(a.x), logf(a.y));
}

__forceinline__ __device__ vec3 log(vec3 a)
{
    return vec3(logf(a.x), logf(a.y), logf(a.z));
}

__forceinline__ __device__ vec4 log(vec4 a)
{
    return vec4(logf(a.x), logf(a.y), logf(a.z), logf(a.w));
}

__forceinline__ __device__ vec2 sqrt(vec2 a)
{
    return vec2(sqrtf(a.x), sqrtf(a.y));
}
__forceinline__ __device__ vec3 sqrt(vec3 a)
{
    return vec3(sqrtf(a.x), sqrtf(a.y), sqrtf(a.z));
}
__forceinline__ __device__ vec4 sqrt(vec4 a)
{
    return vec4(sqrtf(a.x), sqrtf(a.y), sqrtf(a.z), sqrtf(a.w));
}

__forceinline__ __device__ float inversesqrt(float a)
{
    return rsqrt(a);
}
__forceinline__ __device__ vec2 inversesqrt(vec2 a)
{
    return vec2(rsqrtf(a.x), rsqrtf(a.y));
}
__forceinline__ __device__ vec3 inversesqrt(vec3 a)
{
    return vec3(rsqrtf(a.x), rsqrtf(a.y), rsqrtf(a.z));
}
__forceinline__ __device__ vec4 inversesqrt(vec4 a)
{
    return vec4(rsqrtf(a.x), rsqrtf(a.y), rsqrtf(a.z), rsqrtf(a.w));
}
///////////////end of exponential//////////////////////////////////////////////////


//////////////begin of common math/////////////////////////////////////////////////
__forceinline__ __device__ vec2 abs(vec2 a)
{
    return vec2(abs(a.x), abs(a.y));
}
__forceinline__ __device__ vec3 abs(vec3 a)
{
    return vec3(abs(a.x), abs(a.y), abs(a.z));
}
__forceinline__ __device__ vec4 abs(vec4 a)
{
    return vec4(abs(a.x), abs(a.y), abs(a.z), abs(a.w));
}

__forceinline__ __device__ float m_sign(float a)
{
    if(a<0) return -1;
    if(a==0) return 0;
    if(a>0)  return 1;
}
__forceinline__ __device__ vec2 sign(vec2 a)
{

    return vec2(m_sign(a.x), m_sign(a.y));
}
__forceinline__ __device__ vec3 sign(vec3 a)
{

    return vec3(m_sign(a.x), m_sign(a.y), m_sign(a.z));
}
__forceinline__ __device__ vec4 sign(vec4 a)
{

    return vec4(m_sign(a.x), m_sign(a.y), m_sign(a.z), m_sign(a.w));
}

__forceinline__ __device__ vec2 floor(vec2 a)
{
    return vec2(floorf(a.x), floorf(a.y));
}
__forceinline__ __device__ vec3 floor(vec3 a)
{
    return vec3(floorf(a.x), floorf(a.y), floorf(a.z));
}
__forceinline__ __device__ vec4 floor(vec4 a)
{
    return vec4(floorf(a.x), floorf(a.y), floorf(a.z), floorf(a.w));
}

__forceinline__ __device__ vec2 ceil(vec2 a)
{
    return vec2(ceilf(a.x), ceilf(a.y));
}
__forceinline__ __device__ vec3 ceil(vec3 a)
{
    return vec3(ceilf(a.x), ceilf(a.y), ceilf(a.z));
}
__forceinline__ __device__ vec4 ceil(vec4 a)
{
    return vec4(ceilf(a.x), ceilf(a.y), ceilf(a.z), ceilf(a.w));
}

__forceinline__ __device__ vec2 mod(vec2 a, float b)
{
    return vec2(fmodf(a.x, b), fmodf(a.y,b));
}
__forceinline__ __device__ vec2 mod(vec2 a, vec2 b)
{
    return vec2(fmodf(a.x, b.x), fmodf(a.y,b.y));
}

__forceinline__ __device__ vec3 mod(vec3 a, float b)
{
    return vec3(fmodf(a.x, b), fmodf(a.y,b), fmodf(a.z, b));
}
__forceinline__ __device__ vec3 mod(vec3 a, vec3 b)
{
    return vec3(fmodf(a.x, b.x), fmodf(a.y,b.y), fmodf(a.z, b.z));
}

__forceinline__ __device__ vec4 mod(vec4 a, float b)
{
    return vec4(fmodf(a.x, b), fmodf(a.y,b), fmodf(a.z, b), fmodf(a.w, b));
}
__forceinline__ __device__ vec4 mod(vec4 a, vec4 b)
{
    return vec4(fmodf(a.x, b.x), fmodf(a.y,b.y), fmodf(a.z, b.z), fmodf(a.w, b.w));
}

__forceinline__ __device__ vec2 min(vec2 a, float b)
{
    return vec2(fminf(a.x, b), fminf(a.y,b));
}
__forceinline__ __device__ vec2 min(vec2 a, vec2 b)
{
    return vec2(fminf(a.x, b.x), fminf(a.y,b.y));
}
__forceinline__ __device__ vec3 min(vec3 a, float b)
{
    return vec3(fminf(a.x, b), fminf(a.y,b), fminf(a.z, b));
}
__forceinline__ __device__ vec3 min(vec3 a, vec3 b)
{
    return vec3(fminf(a.x, b.x), fminf(a.y,b.y), fminf(a.z, b.z));
}
__forceinline__ __device__ vec4 min(vec4 a, float b)
{
    return vec4(fminf(a.x, b), fminf(a.y,b), fminf(a.z, b), fminf(a.w, b));
}
__forceinline__ __device__ vec4 min(vec4 a, vec4 b)
{
    return vec4(fminf(a.x, b.x), fminf(a.y,b.y), fminf(a.z, b.z), fminf(a.w, b.w));
}

__forceinline__ __device__ vec2 max(vec2 a, float b)
{
    return vec2(fmaxf(a.x, b), fmaxf(a.y,b));
}
__forceinline__ __device__ vec2 max(vec2 a, vec2 b)
{
    return vec2(fmaxf(a.x, b.x), fmaxf(a.y,b.y));
}
__forceinline__ __device__ vec3 max(vec3 a, float b)
{
    return vec3(fmaxf(a.x, b), fmaxf(a.y,b), fmaxf(a.z, b));
}
__forceinline__ __device__ vec3 max(vec3 a, vec3 b)
{
    return vec3(fmaxf(a.x, b.x), fmaxf(a.y,b.y), fmaxf(a.z, b.z));
}
__forceinline__ __device__ vec4 max(vec4 a, float b)
{
    return vec4(fmaxf(a.x, b), fmaxf(a.y,b), fmaxf(a.z, b), fmaxf(a.w, b));
}
__forceinline__ __device__ vec4 max(vec4 a, vec4 b)
{
    return vec4(fmaxf(a.x, b.x), fmaxf(a.y,b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w));
}

__forceinline__ __device__ vec2 clamp(vec2 a, float b, float c)
{
    return min(max(a, b), c);
}
__forceinline__ __device__ vec2 clamp(vec2 a, vec2 b, vec2 c)
{
    return min(max(a, b), c);
}
__forceinline__ __device__ vec3 clamp(vec3 a, float b, float c)
{
    return min(max(a, b), c);
}
__forceinline__ __device__ vec3 clamp(vec3 a, vec3 b, vec3 c)
{
    return min(max(a, b), c);
}
__forceinline__ __device__ vec4 clamp(vec4 a, float b, float c)
{
    return min(max(a, b), c);
}
__forceinline__ __device__ vec4 clamp(vec4 a, vec4 b, vec4 c)
{
    return min(max(a, b), c);
}

__forceinline__ __device__ float mix(float a, float b, float c)
{
    return (1-c)*a + c * b;
}
__forceinline__ __device__ vec2 mix(vec2 a, float b, float c)
{
    return (1-c)*a + c * b;
}
__forceinline__ __device__ vec2 mix(vec2 a, vec2 b, float c)
{
    return (1-c)*a + c * b;
}
__forceinline__ __device__ vec2 mix(vec2 a, vec2 b, vec2 c)
{
    return (1-c)*a + c * b;
}
__forceinline__ __device__ vec2 mix(vec2 a, float b, vec2 c)
{
    return (1-c)*a + c * b;
}
__forceinline__ __device__ vec2 mix(float a, float b, vec2 c)
{
    return (1-c)*a + c * b;
}
__forceinline__ __device__ vec2 mix(float a, vec2 b, float c)
{
    return (1-c)*a + c * b;
}
__forceinline__ __device__ vec2 mix(float a, vec2 b, vec2 c)
{
    return (1-c)*a + c * b;
}




__forceinline__ __device__ vec3 mix(vec3 a, float b, float c)
{
    return (1-c)*a + c * b;
}
__forceinline__ __device__ vec3 mix(vec3 a, vec3 b, float c)
{
    return (1-c)*a + c * b;
}
__forceinline__ __device__ vec3 mix(vec3 a, vec3 b, vec3 c)
{
    return (1-c)*a + c * b;
}
__forceinline__ __device__ vec3 mix(vec3 a, float b, vec3 c)
{
    return (1-c)*a + c * b;
}
__forceinline__ __device__ vec3 mix(float a, float b, vec3 c)
{
    return (1-c)*a + c * b;
}
__forceinline__ __device__ vec3 mix(float a, vec3 b, float c)
{
    return (1-c)*a + c * b;
}
__forceinline__ __device__ vec3 mix(float a, vec3 b, vec3 c)
{
    return (1-c)*a + c * b;
}



__forceinline__ __device__ vec4 mix(vec4 a, float b, float c)
{
    return (1-c)*a + c * b;
}
__forceinline__ __device__ vec4 mix(vec4 a, vec4 b, float c)
{
    return (1-c)*a + c * b;
}
__forceinline__ __device__ vec4 mix(vec4 a, vec4 b, vec4 c)
{
    return (1-c)*a + c * b;
}
__forceinline__ __device__ vec4 mix(vec4 a, float b, vec4 c)
{
    return (1-c)*a + c * b;
}
__forceinline__ __device__ vec4 mix(float a, float b, vec4 c)
{
    return (1-c)*a + c * b;
}
__forceinline__ __device__ vec4 mix(float a, vec4 b, float c)
{
    return (1-c)*a + c * b;
}
__forceinline__ __device__ vec4 mix(float a, vec4 b, vec4 c)
{
    return (1-c)*a + c * b;
}

__forceinline__ __device__ float step(float limit, float a)
{
    return a>limit?1:0;
}

__forceinline__ __device__ vec2 step(float limit, vec2 a)
{
    return vec2(step(limit, a.x), step(limit, a.y));
}
__forceinline__ __device__ vec3 step(float limit, vec3 a)
{
    return vec3(step(limit, a.x), step(limit, a.y), step(limit, a.z));
}
__forceinline__ __device__ vec4 step(float limit, vec4 a)
{
    return vec4(step(limit, a.x), step(limit, a.y), step(limit, a.z), step(limit, a.w));
}
__forceinline__ __device__ vec2 step(vec2 limit, vec2 a)
{
    return vec2(step(limit.x, a.x), step(limit.y, a.y));
}

__forceinline__ __device__ vec3 step(vec3 limit, vec3 a)
{
    return vec3(step(limit.x, a.x), step(limit.y, a.y), step(limit.z, a.z));
}

__forceinline__ __device__ vec4 step(vec4 limit, vec4 a)
{
    return vec4(step(limit.x, a.x), step(limit.y, a.y), step(limit.z, a.z), step(limit.w, a.w));
}

__forceinline__ __device__ float smoothstep(float a, float b, float c)
{
    if(c>b)
    {
        return 1;
    }
    if(c>a)
    {
        return mix(a, b, (c-a)/(b-a));
    }
    return 0;
}
__forceinline__ __device__ vec2 smoothstep(vec2 a, vec2 b, vec2 c)
{
    return vec2(smoothstep(a.x, b.x, c.x), smoothstep(a.y, b.y, c.y));
}
__forceinline__ __device__ vec3 smoothstep(vec3 a, vec3 b, vec3 c)
{
    return vec3(smoothstep(a.x, b.x, c.x), smoothstep(a.y, b.y, c.y), smoothstep(a.z, b.z, c.z));
}
__forceinline__ __device__ vec4 smoothstep(vec4 a, vec4 b, vec4 c)
{
    return vec4(smoothstep(a.x, b.x, c.x), smoothstep(a.y, b.y, c.y), smoothstep(a.z, b.z, c.z), smoothstep(a.w, b.w, c.w));
}
__forceinline__ __device__ vec2 smoothstep(float a, float b, vec2 c)
{
    return smoothstep(vec2(a), vec2(b), c);
}
__forceinline__ __device__ vec3 smoothstep(float a, float b, vec3 c)
{
    return smoothstep(vec3(a), vec3(b), c);
}
__forceinline__ __device__ vec4 smoothstep(float a, float b, vec4 c)
{
    return smoothstep(vec4(a), vec4(b), c);
}
__forceinline__ __device__ float fract(float a)
{
    return a-floor(a);
}
__forceinline__ __device__ vec2 fract(vec2 a)
{
    return a-floor(a);
}
__forceinline__ __device__ vec3 fract(vec3 a)
{
    return a-floor(a);
}
__forceinline__ __device__ vec4 fract(vec4 a)
{
    return a-floor(a);
}
//////////////end of common math///////////////////////////////////////////////////

/////////////begin of geometry math///////////////////////////////////////////////
__forceinline__ __device__ float dot(vec2 a, vec2 b)
{
    return a.x*b.x + a.y*b.y ;
}
__forceinline__ __device__ float dot(vec3 a, vec3 b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
__forceinline__ __device__ float dot(vec4 a, vec4 b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}
__forceinline__ __device__ float length(vec2 a)
{
    return sqrtf(dot(a,a));
}
__forceinline__ __device__ float length(vec3 a)
{
    return sqrtf(dot(a,a));
}
__forceinline__ __device__ float length(vec4 a)
{
    return sqrtf(dot(a,a));
}
__forceinline__ __device__ vec2 normalize(vec2 a)
{
    return a/(length(a)+0.0000001);
}

__forceinline__ __device__ vec3 normalize(vec3 a)
{
    return a/(length(a)+0.0000001);
}
__forceinline__ __device__ vec4 normalize(vec4 a)
{
    return a/(length(a)+0.0000001);
}

__forceinline__ __device__ float distance(vec2 a, vec2 b)
{
    return length(b-a);
}
__forceinline__ __device__ float distance(vec3 a, vec3 b)
{
    return length(b-a);
}
__forceinline__ __device__ float distance(vec4 a, vec4 b)
{
    return length(b-a);
}
__forceinline__ __device__ vec3 cross(vec3 a, vec3 b)
{
    float3 res = cross(make_float3(a.x, a.y, a.z), make_float3(b.x, b.y, b.z));
    return vec3(res.x, res.y, res.z);
}
/////////////end of geometry math/////////////////////////////////////////////////

//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

struct RadiancePRD
{
    // TODO: move some state directly into payload registers?
    float3       emitted;
    float3       radiance;
    float3       attenuation;
    float3       origin;
    float3       direction;
    float        opacity;
    float        prob;
    float        prob2;
    unsigned int seed;
    unsigned int flags = 0;
    int          countEmitted;
    int          done;
    int          pad;
};


struct Onb
{
  __forceinline__ __device__ Onb(const float3& normal)
  {
    m_normal = normal;

    if( fabs(m_normal.x) > fabs(m_normal.z) )
    {
      m_binormal.x = -m_normal.y;
      m_binormal.y =  m_normal.x;
      m_binormal.z =  0;
    }
    else
    {
      m_binormal.x =  0;
      m_binormal.y = -m_normal.z;
      m_binormal.z =  m_normal.y;
    }

    m_binormal = normalize(m_binormal);
    m_tangent = cross( m_binormal, m_normal );
  }

  __forceinline__ __device__ void inverse_transform(float3& p) const
  {
    p = p.x*m_tangent + p.y*m_binormal + p.z*m_normal;
  }

  float3 m_tangent;
  float3 m_binormal;
  float3 m_normal;
};

namespace zeno{
__forceinline__ __device__ float lerp(float a, float b, float c)
{
    return (1-c)*a + c*b;
}
__forceinline__ __device__ float3 lerp(float3 a, float3 b, float c)
{
    float3 coef = make_float3(c,c,c);
    return (make_float3(1,1,1)-coef)*a + coef*b;
}
__forceinline__ __device__ float length(float3 vec){
    return sqrtf(dot(vec,vec));
}
__forceinline__ __device__ float3 normalize(float3 vec){
    return vec/(zeno::length(vec)+0.00001);
}
__forceinline__ __device__ float clamp(float c, float a, float b){
    return max(min(c,b),a);
}
__forceinline__ __device__ float3 sin(float3 a){
    return make_float3(sinf(a.x), sinf(a.y), sinf(a.z));
}
__forceinline__ __device__ float3 fract(float3 a){
    float3 temp;
    return make_float3(modff(a.x, &temp.x), modff(a.y, &temp.y), modff(a.z, &temp.z));
}
};
namespace BRDFBasics{
__forceinline__ __device__  float fresnel(float cosT){
    float v = zeno::clamp(1-cosT,0.0f,1.0f);
    float v2 = v *v;
    return v2 * v2 * v;
}
__forceinline__ __device__  float GTR1(float cosT,float a){
    if(a >= 1.0) return 1/M_PIf;
    float t = (1+(a*a-1)*cosT*cosT);
    return (a*a-1.0f) / (M_PIf*logf(a*a)*t);
}
__forceinline__ __device__  float GTR2(float cosT,float a){
    float t = (1+(a*a-1)*cosT*cosT);
    return (a*a) / (M_PIf*t*t);
}
__forceinline__ __device__  float GGX(float cosT, float a){
    float a2 = a*a;
    float b = cosT*cosT;
    return 1.0/ (cosT + sqrtf(a2 + b - a2*b));
}
__forceinline__ __device__  float3 sampleOnHemisphere(unsigned int &seed, float roughness)
{
    float x = rnd(seed);
    float y = rnd(seed);

    float a = roughness*roughness;
	
	float phi = 2.0 * M_PIf * x;
	float cosTheta = sqrtf((1.0 - y) / (1.0 + (a*a - 1.0) * y));
	float sinTheta = sqrtf(1.0 - cosTheta*cosTheta);
	

    return make_float3(cos(phi) * sinTheta,  sin(phi) * sinTheta, cosTheta);
}
};
namespace DisneyBRDF
{   
__forceinline__ __device__ float pdf(
        float3 baseColor,
        float metallic,
        float subsurface,
        float specular,
        float roughness,
        float specularTint,
        float anisotropic,
        float sheen,
        float sheenTint,
        float clearcoat,
        float clearcoatGloss,
        float3 N,
        float3 T,
        float3 B,
        float3 wi,
        float3 wo)
    {
        float3 n = N;
        float spAlpha = max(0.001, roughness);
        float ccAlpha = zeno::lerp(0.1, 0.001, clearcoatGloss);
        float diffRatio = 0.5*(1.0 - metallic);
        float spRatio = 1.0 - diffRatio;

        float3 half = zeno::normalize(wi + wo);

        float cosTheta = abs(dot(n, half));
        float pdfGTR2 = BRDFBasics::GTR2(cosTheta, spAlpha) * cosTheta;
        float pdfGTR1 = BRDFBasics::GTR1(cosTheta, ccAlpha) * cosTheta;

        float ratio = 1.0/(1.0 + clearcoat);
        float pdfSpec = zeno::lerp(pdfGTR1, pdfGTR2, ratio)/(4.0 * abs(dot(wi, half)));
        float pdfDiff = abs(dot(wi, n)) * (1.0/M_PIf);

        return diffRatio * pdfDiff + spRatio * pdfSpec;
    }

__forceinline__ __device__ float3 sample_f(
        unsigned int &seed, 
        float3 baseColor,
        float metallic,
        float subsurface,
        float specular,
        float roughness,
        float specularTint,
        float anisotropic,
        float sheen,
        float sheenTint,
        float clearcoat,
        float clearcoatGloss,
        float3 N,
        float3 T,
        float3 B,
        float3 wo,
        float &is_refl)
    {
        
        float ratiodiffuse = (1.0 - metallic)/2.0;
        float p = rnd(seed);
        
        Onb tbn = Onb(N);
        
        float3 wi;
        
        if( p < ratiodiffuse){
            //sample diffuse lobe
            
            float3 P = BRDFBasics::sampleOnHemisphere(seed, 1.0);
            wi = P;
            tbn.inverse_transform(wi);
            wi = normalize(wi);
            is_refl = 0;
        }else{
            //sample specular lobe.
            float a = max(0.001, roughness);
            
            float3 P = BRDFBasics::sampleOnHemisphere(seed, a*a);
            float3 half = normalize(P);
            tbn.inverse_transform(half);            
            wi = half* 2.0* dot(normalize(wo), half) - normalize(wo); //reflection vector
            wi = normalize(wi);
            is_refl = 1;
        }
        
        return wi;
    }
__forceinline__ __device__ float3 eval(
        float3 baseColor,
        float metallic,
        float subsurface,
        float specular,
        float roughness,
        float specularTint,
        float anisotropic,
        float sheen,
        float sheenTint,
        float clearcoat,
        float clearcoatGloss,
        float3 N,
        float3 T,
        float3 B,
        float3 wi,
        float3 wo)
    {
        float3 wh = normalize(wi+ wo);
        float ndoth = dot(N, wh);
        float ndotwi = dot(N, wi);
        float ndotwo = dot(N, wo);
        float widoth = dot(wi, wh);

        if(ndotwi <=0 || ndotwo <=0 )
            return make_float3(0,0,0);

        float3 Cdlin = baseColor;
        float Cdlum = 0.3*Cdlin.x + 0.6*Cdlin.y + 0.1*Cdlin.z;

        float3 Ctint = Cdlum > 0.0 ? Cdlin / Cdlum : make_float3(1.0,1.0,1.0);
        float3 Cspec0 = zeno::lerp(specular*0.08*zeno::lerp(make_float3(1,1,1), Ctint, specularTint), Cdlin, metallic);
        float3 Csheen = zeno::lerp(make_float3(1.0,1.0,1.0), Ctint, sheenTint);

        //diffuse
        float Fd90 = 0.5 + 2.0 * ndoth * ndoth * roughness;
        float Fi = BRDFBasics::fresnel(ndotwi);
        float Fo = BRDFBasics::fresnel(ndotwo);
        
        float Fd = (1 +(Fd90-1)*Fi)*(1+(Fd90-1)*Fo);

        float Fss90 = widoth*widoth*roughness;
        float Fss = zeno::lerp(1.0, Fss90, Fi) * zeno::lerp(1.0,Fss90, Fo);
        float ss = 1.25 * (Fss *(1.0 / (ndotwi + ndotwo) - 0.5) + 0.5);

        float a = max(0.001, roughness);
        float Ds = BRDFBasics::GTR2(ndoth, a);
        float Dc = BRDFBasics::GTR1(ndoth, zeno::lerp(0.1, 0.001, clearcoatGloss));

        float roughg = sqrtf(roughness*0.5 + 0.5);
        float Gs = BRDFBasics::GGX(ndotwo, roughg) * BRDFBasics::GGX(ndotwi, roughg);

        float Gc = BRDFBasics::GGX(ndotwo, 0.25) * BRDFBasics::GGX(ndotwi, 0.25);

        float Fh = BRDFBasics::fresnel(widoth);
        float3 Fs = zeno::lerp(Cspec0, make_float3(1.0,1.0,1.0), Fh);
        float Fc = zeno::lerp(0.04, 1.0, Fh);

        float3 Fsheen = Fh * sheen * Csheen;

        return ((1/M_PIf) * zeno::lerp(Fd, ss, subsurface) * Cdlin + Fsheen) * (1.0 - metallic)
        + Gs*Fs*Ds + 0.25*clearcoat*Gc*Fc*Dc;
    }
};

//////////////////////////////////////////
///here inject common code in glsl style
__forceinline__ __device__ vec3 perlin_hash22(vec3 p)
{
    p = vec3( dot(p,vec3(127.1,311.7,284.4)),
              dot(p,vec3(269.5,183.3,162.2)),
	      	  dot(p,vec3(228.3,164.9,126.0)));
    float a;
    return -1.0 + 2.0 * fract(sin(p)*43758.5453123);
}

__forceinline__ __device__ float perlin_lev1(vec3 p)
{
    vec3 pi = vec3(floor(p));
    vec3 pf = p - pi;
    vec3 w = pf * pf * (3.0 - 2.0 * pf);
    return .08 + .8 * (mix(
			            mix(
                            mix(
                            dot(perlin_hash22(pi + 0), pf - 0),
                            dot(perlin_hash22(pi + 0), pf - 0),
                            w.x),
                            mix(
                            dot(perlin_hash22(pi + vec3(0, 1, 0)), pf - vec3(0, 1, 0)),
                            dot(perlin_hash22(pi + vec3(1, 1, 0)), pf - vec3(1, 1, 0)),
                            w.x),
				        w.y),
			            mix(
				            mix(
                            dot(perlin_hash22(pi + vec3(0, 0, 1)), pf - vec3(0, 0, 1)),
                            dot(perlin_hash22(pi + vec3(1, 0, 1)), pf - vec3(1, 0, 1)),
                            w.x),
				            mix(
                            dot(perlin_hash22(pi + vec3(0, 1, 1)), pf - vec3(0, 1, 1)),
                            dot(perlin_hash22(pi + vec3(1, 1, 1)), pf - vec3(1, 1, 1)),
                            w.x),
				        w.y),
			          w.z));
}

__forceinline__ __device__ float perlin(float p,int n,vec3 a)
{
    float total = 0;
    for(int i=0; i<n; i++)
    {
        float frequency = pow(2.0f,i*1.0f);
        float amplitude = pow(p,i*1.0f);
        total = total + perlin_lev1(a * frequency) * amplitude;
    }

    return total;
}

///end example of common code injection in glsl style








//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

static __forceinline__ __device__ void* unpackPointer( unsigned int i0, unsigned int i1 )
{
    const unsigned long long uptr = static_cast<unsigned long long>( i0 ) << 32 | i1;
    void*           ptr = reinterpret_cast<void*>( uptr );
    return ptr;
}


static __forceinline__ __device__ void  packPointer( void* ptr, unsigned int& i0, unsigned int& i1 )
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}


static __forceinline__ __device__ RadiancePRD* getPRD()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<RadiancePRD*>( unpackPointer( u0, u1 ) );
}


static __forceinline__ __device__ void setPayloadOcclusion( bool occluded )
{
    optixSetPayload_0( static_cast<unsigned int>( occluded ) );
}


static __forceinline__ __device__ void cosine_sample_hemisphere(const float u1, const float u2, float3& p)
{
  // Uniformly sample disk.
  const float r   = sqrtf( u1 );
  const float phi = 2.0f*M_PIf * u2;
  p.x = r * cosf( phi );
  p.y = r * sinf( phi );

  // Project up to hemisphere.
  p.z = sqrtf( fmaxf( 0.0f, 1.0f - p.x*p.x - p.y*p.y ) );
}


static __forceinline__ __device__ void traceRadiance(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        RadiancePRD*           prd
        )
{
    // TODO: deduce stride from num ray-types passed in params

    unsigned int u0, u1;
    packPointer( prd, u0, u1 );
    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_NONE,
            RAY_TYPE_RADIANCE,        // SBT offset
            RAY_TYPE_COUNT,           // SBT stride
            RAY_TYPE_RADIANCE,        // missSBTIndex
            u0, u1 );
}


static __forceinline__ __device__ bool traceOcclusion(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax
        )
{
    unsigned int occluded = 0u;
    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                    // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_NONE,
            RAY_TYPE_OCCLUSION,      // SBT offset
            RAY_TYPE_COUNT,          // SBT stride
            RAY_TYPE_OCCLUSION       // missSBTIndex
            );

}


//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

extern "C" __global__ void __raygen__rg()
{
    const int    w   = params.width;
    const int    h   = params.height;
    const float3 eye = params.eye;
    const float3 U   = params.U;
    const float3 V   = params.V;
    const float3 W   = params.W;
    const uint3  idx = optixGetLaunchIndex();
    const int    subframe_index = params.subframe_index;

    unsigned int seed = tea<4>( idx.y*w + idx.x, subframe_index );

    float3 result = make_float3( 0.0f );
    int i = params.samples_per_launch;
    do
    {
        // The center of each pixel is at fraction (0.5,0.5)
        const float2 subpixel_jitter = make_float2( rnd( seed ), rnd( seed ) );

        const float2 d = 2.0f * make_float2(
                ( static_cast<float>( idx.x ) + subpixel_jitter.x ) / static_cast<float>( w ),
                ( static_cast<float>( idx.y ) + subpixel_jitter.y ) / static_cast<float>( h )
                ) - 1.0f;
        float3 ray_direction = normalize(d.x*U + d.y*V + W);
        float3 ray_origin    = eye;

        RadiancePRD prd;
        prd.emitted      = make_float3(0.f);
        prd.radiance     = make_float3(0.f);
        prd.attenuation  = make_float3(1.f);
        prd.prob         = 1.0f;
        prd.prob2        = 1.0f;
        prd.countEmitted = true;
        prd.done         = false;
        prd.seed         = seed;
        prd.opacity      = 0;
        int depth = 0;
        for( ;; )
        {
            traceRadiance(
                    params.handle,
                    ray_origin,
                    ray_direction,
                    0.01f,  // tmin       // TODO: smarter offset
                    1e16f,  // tmax
                    &prd );

            result += prd.emitted;
            result += prd.radiance * prd.attenuation/prd.prob;

            if( prd.done  || depth >= 5 ) // TODO RR, variable for depth
                break;

            ray_origin    = prd.origin;
            ray_direction = prd.direction;
            if(prd.opacity<0.99)
                ++depth;
        }
    }
    while( --i );

    const uint3    launch_index = optixGetLaunchIndex();
    const unsigned int image_index  = launch_index.y * params.width + launch_index.x;
    float3         accum_color  = result / static_cast<float>( params.samples_per_launch );

    if( subframe_index > 0 )
    {
        const float                 a = 1.0f / static_cast<float>( subframe_index+1 );
        const float3 accum_color_prev = make_float3( params.accum_buffer[ image_index ]);
        accum_color = lerp( accum_color_prev, accum_color, a );
    }
    params.accum_buffer[ image_index ] = make_float4( accum_color, 1.0f);
    params.frame_buffer[ image_index ] = make_color ( accum_color );
}


extern "C" __global__ void __miss__radiance()
{
    MissData* rt_data  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    RadiancePRD* prd = getPRD();

    prd->radiance = make_float3( rt_data->bg_color );
    prd->done      = true;
}


extern "C" __global__ void __closesthit__occlusion()
{
    setPayloadOcclusion( true );
}
extern "C" __global__ void __anyhit__shadow_cutout()
{
    HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();

    const int    prim_idx        = optixGetPrimitiveIndex();
    const float3 ray_dir         = optixGetWorldRayDirection();
    const int    vert_idx_offset = prim_idx*3;

    const float3 v0   = make_float3( rt_data->vertices[ vert_idx_offset+0 ] );
    const float3 v1   = make_float3( rt_data->vertices[ vert_idx_offset+1 ] );
    const float3 v2   = make_float3( rt_data->vertices[ vert_idx_offset+2 ] );
    const float3 N_0  = normalize( cross( v1-v0, v2-v0 ) );

    const float3 N    = faceforward( N_0, -ray_dir, N_0 );
    const float3 P    = optixGetWorldRayOrigin() + optixGetRayTmax()*ray_dir;

    RadiancePRD* prd = getPRD();
    float3 baseColor = make_float3(1.0,0.766,0.336);
    float  metallic = 1;
    float  roughness = 0.1;
    float  subsurface = 0.0;
    float  specular = 0;
    float  specularTint = 0.0;
    float  anisotropic = 0.0;
    float  sheen = 0.0;
    float  sheenTint = 0.0;
    float  clearCoat = 0.0;
    float  clearCoatGloss = 0.0;
    float  opacity = 0.0;

    vec3  mat_baseColor = vec3(1.0,0.766,0.336);
    float mat_metallic = 1;
    float mat_roughness = 0.1;
    float mat_subsurface = 0.0;
    float mat_specular = 0;
    float mat_specularTint = 0.0;
    float mat_anisotropic = 0.0;
    float mat_sheen = 0.0;
    float mat_sheenTint = 0.0;
    float mat_clearCoat = 0.0;
    float mat_clearCoatGloss = 0.0;
    float mat_opacity = 0.0;
    vec3 attr_pos = vec3(P.x, P.y, P.z);
    vec3 attr_norm = vec3(0,0,1);
    vec3 attr_uv = vec3(0,0,0);//todo later
    vec3 attr_clr = vec3(rt_data->diffuse_color.x, rt_data->diffuse_color.y, rt_data->diffuse_color.z);
    vec3 attr_tang = vec3(0,0,0);
///////here injecting of material code in GLSL style///////////////////////////////
    

    float pnoise = perlin(1, 3, attr_pos*0.02);
    pnoise = clamp(pnoise, 0.0f, 1.0f);

    float pnoise2 = perlin(1, 4, attr_pos*0.02);
    mat_metallic = pnoise;

    mat_roughness = pnoise2;
    mat_roughness = clamp(mat_roughness, 0.01f,0.99f)*0.5f;

    float pnoise3 = perlin(10.0, 5, attr_pos*0.005);
    mat_opacity = clamp(pnoise3, 0.0f,1.0f);

////////////end of GLSL material code injection///////////////////////////////////////////////
    baseColor = make_float3(mat_baseColor.x, mat_baseColor.y, mat_baseColor.z);
    metallic = mat_metallic;;
    roughness = mat_roughness;
    subsurface = mat_subsurface;
    specular = mat_specular;
    specularTint = mat_specularTint;
    anisotropic = mat_anisotropic;
    sheen = mat_sheen;
    sheenTint = mat_sheenTint;
    clearCoat = mat_clearCoat;
    clearCoatGloss = mat_clearCoatGloss;
    opacity = mat_opacity;
    // Stochastic alpha test to get an alpha blend effect.
    if (opacity >0.99 ) // No need to calculate an expensive random number if the test is going to fail anyway.
    {
        optixIgnoreIntersection();
    }
    else
    {
        prd->flags |= 1;
        optixTerminateRay();
    }
}


extern "C" __global__ void __closesthit__radiance()
{
    HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();

    const int    prim_idx        = optixGetPrimitiveIndex();
    const float3 ray_dir         = optixGetWorldRayDirection();
    const int    vert_idx_offset = prim_idx*3;

    const float3 v0   = make_float3( rt_data->vertices[ vert_idx_offset+0 ] );
    const float3 v1   = make_float3( rt_data->vertices[ vert_idx_offset+1 ] );
    const float3 v2   = make_float3( rt_data->vertices[ vert_idx_offset+2 ] );
    const float3 N_0  = normalize( cross( v1-v0, v2-v0 ) );

    const float3 N    = faceforward( N_0, -ray_dir, N_0 );
    const float3 P    = optixGetWorldRayOrigin() + optixGetRayTmax()*ray_dir;

    RadiancePRD* prd = getPRD();

    if( prd->countEmitted )
        prd->emitted = rt_data->emission_color;
    else
        prd->emitted = make_float3( 0.0f );


    float3 baseColor = make_float3(1.0,0.766,0.336);
    float  metallic = 1;
    float  roughness = 0.1;
    float  subsurface = 0.0;
    float  specular = 0;
    float  specularTint = 0.0;
    float  anisotropic = 0.0;
    float  sheen = 0.0;
    float  sheenTint = 0.0;
    float  clearCoat = 0.0;
    float  clearCoatGloss = 0.0;
    float  opacity = 0.0;

    vec3  mat_baseColor = vec3(1.0,0.766,0.336);
    float mat_metallic = 1;
    float mat_roughness = 0.1;
    float mat_subsurface = 0.0;
    float mat_specular = 0;
    float mat_specularTint = 0.0;
    float mat_anisotropic = 0.0;
    float mat_sheen = 0.0;
    float mat_sheenTint = 0.0;
    float mat_clearCoat = 0.0;
    float mat_clearCoatGloss = 0.0;
    float mat_opacity = 0.0;
    vec3 attr_pos = vec3(P.x, P.y, P.z);
    vec3 attr_norm = vec3(0,0,1);
    vec3 attr_uv = vec3(0,0,0);//todo later
    vec3 attr_clr = vec3(rt_data->diffuse_color.x, rt_data->diffuse_color.y, rt_data->diffuse_color.z);
    vec3 attr_tang = vec3(0,0,0);
///////here injecting of material code in GLSL style///////////////////////////////
    

    float pnoise = perlin(1, 3, attr_pos*0.02);
    pnoise = clamp(pnoise, 0.0f, 1.0f);

    float pnoise2 = perlin(1, 4, attr_pos*0.02);
    mat_metallic = pnoise;

    mat_roughness = pnoise2;
    mat_roughness = clamp(mat_roughness, 0.01f,0.99f)*0.5f;

    float pnoise3 = perlin(10.0, 5, attr_pos*0.005);
    mat_opacity = clamp(pnoise3, 0.0f,1.0f);

////////////end of GLSL code injection///////////////////////////////////////////////
    baseColor = make_float3(mat_baseColor.x, mat_baseColor.y, mat_baseColor.z);
    metallic = mat_metallic;;
    roughness = mat_roughness;
    subsurface = mat_subsurface;
    specular = mat_specular;
    specularTint = mat_specularTint;
    anisotropic = mat_anisotropic;
    sheen = mat_sheen;
    sheenTint = mat_sheenTint;
    clearCoat = mat_clearCoat;
    clearCoatGloss = mat_clearCoatGloss;
    opacity = mat_opacity;
    //todo normal mapping TBN*N;


    
    //end of material computation
    metallic = zeno::clamp(metallic,0.01, 0.99);
    roughness = zeno::clamp(roughness, 0.01,0.99);
    //discard fully opacity pixels
    prd->opacity = opacity;
    if(opacity>0.99)
    {
        prd->radiance += make_float3(0.0f);
        prd->origin = P;
        prd->direction = ray_dir;
        return;
    }

    //{
    unsigned int seed = prd->seed;
    float is_refl;
    float3 wi = DisneyBRDF::sample_f(
                                seed, 
                                baseColor,
                                metallic,
                                subsurface,
                                specular,
                                roughness,
                                specularTint,
                                anisotropic,
                                sheen,
                                sheenTint,
                                clearCoat,
                                clearCoatGloss,
                                N,
                                make_float3(0,0,0),
                                make_float3(0,0,0),
                                -normalize(ray_dir),
                                is_refl);

    float pdf = DisneyBRDF::pdf(baseColor,
                                metallic,
                                subsurface,
                                specular,
                                roughness,
                                specularTint,
                                anisotropic,
                                sheen,
                                sheenTint,
                                clearCoat,
                                clearCoatGloss,
                                N,
                                make_float3(0,0,0),
                                make_float3(0,0,0),
                                wi,
                                -normalize(ray_dir)
                                );
    float3 f = DisneyBRDF::eval(baseColor,
                                metallic,
                                subsurface,
                                specular,
                                roughness,
                                specularTint,
                                anisotropic,
                                sheen,
                                sheenTint,
                                clearCoat,
                                clearCoatGloss,
                                N,
                                make_float3(0,0,0),
                                make_float3(0,0,0),
                                wi,
                                -normalize(ray_dir)
                                );
    prd->prob2 = prd->prob;
    prd->prob *= pdf; 
    prd->origin = P;
    prd->direction = wi;
    prd->countEmitted = false;
    if(is_refl)
        prd->attenuation *= f * clamp(dot(wi, N),0.0f,1.0f);
    else
        prd->attenuation *= f * clamp(dot(wi, N),0.0f,1.0f);
    //}

    // {
    //     const float z1 = rnd(seed);
    //     const float z2 = rnd(seed);

    //     float3 w_in;
    //     cosine_sample_hemisphere( z1, z2, w_in );
    //     Onb onb( N );
    //     onb.inverse_transform( w_in );
    //     prd->direction = w_in;
    //     prd->origin    = P;

    //     prd->attenuation *= rt_data->diffuse_color;
    //     prd->countEmitted = false;
    // }

    const float z1 = rnd(seed);
    const float z2 = rnd(seed);
    prd->seed = seed;

    ParallelogramLight light = params.light;
    const float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

    // Calculate properties of light sample (for area based pdf)
    const float  Ldist = length(light_pos - P );
    const float3 L     = normalize(light_pos - P );
    const float  nDl   = dot( N, L );
    const float  LnDl  = -dot( light.normal, L );

    float weight = 0.0f;
    if( nDl > 0.0f && LnDl > 0.0f )
    {
        prd->flags = 0;
        traceOcclusion(
            params.handle,
            P,
            L,
            0.01f,         // tmin
            Ldist - 0.01f  // tmax
            );
        unsigned int occluded = prd->flags;
        if( !occluded )
        {
            const float A = length(cross(light.v1, light.v2));
            weight = nDl * LnDl * A / (M_PIf * Ldist * Ldist);
        }
    }

    prd->radiance += light.emission * weight;
}
