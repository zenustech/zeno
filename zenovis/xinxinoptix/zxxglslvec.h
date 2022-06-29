#pragma once

#include <cuda/helpers.h>

struct vec4{
    float x, y, z, w;
    __forceinline__ __device__ vec4(const float4 &_v)
    {
        x = _v.x; z = _v.z;
        y = _v.y; w = _v.w;
    }
    __forceinline__ __device__ vec4(float _x, float _y, float _z, float _w)
    {
        x = _x; y = _y; z = _z, w = _w;
    }
    __forceinline__ __device__ vec4(float _x)
    {
        x = _x; y = _x; z = _x; w = _x;
    }
    vec4() = default;
    __forceinline__ __device__ operator float4() const {
        return make_float4(x, y, z, w);
    }
};
struct vec3{
    float x, y, z;
    __forceinline__ __device__ vec3(const float3 &_v)
    {
        x = _v.x;
        y = _v.y;
        z = _v.z;
    }
    __forceinline__ __device__ vec3(float _x, float _y, float _z)
    {
        x = _x; y = _y; z = _z;
    }
    explicit __forceinline__ __device__ vec3(float _x)
    {
        x = _x; y = _x; z = _x;
    }
    explicit __forceinline__ __device__ vec3(const vec4 &_v) : vec3(_v.x, _v.y, _v.z) {
    }
    explicit __forceinline__ __device__ operator float() const {
        return x;
    }
    vec3() = default;
    __forceinline__ __device__ operator float3() const {
        return make_float3(x, y, z);
    }
};

struct vec2{
    float x, y;
    __forceinline__ __device__ vec2(const float2 &_v)
    {
        x = _v.x;
        y = _v.y;
    }
    __forceinline__ __device__ vec2(float _x, float _y)
    {
        x = _x; y = _y;
    }
    __forceinline__ __device__ vec2(float _x)
    {
        x = _x; y = _x;
    }
    explicit __forceinline__ __device__ vec2(const vec4 &_v) : vec2(_v.x, _v.y) {
    }
    explicit __forceinline__ __device__ vec2(const vec3 &_v) : vec2(_v.x, _v.y) {
    }
    explicit __forceinline__ __device__ operator float() const {
        return x;
    }
    vec2() = default;
    __forceinline__ __device__ operator float2() const {
        return make_float2(x, y);
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
__forceinline__ __device__ vec2 operator+(vec2 a)
{
    return vec2(+a.x, +a.y);
}
__forceinline__ __device__ vec3 operator+(vec3 a)
{
    return vec3(+a.x, +a.y, +a.z);
}
__forceinline__ __device__ vec4 operator+(vec4 a)
{
    return vec4(+a.x, +a.y, +a.z, +a.w);
}
__forceinline__ __device__ vec2 operator-(vec2 a)
{
    return vec2(-a.x, -a.y);
}
__forceinline__ __device__ vec3 operator-(vec3 a)
{
    return vec3(-a.x, -a.y, -a.z);
}
__forceinline__ __device__ vec4 operator-(vec4 a)
{
    return vec4(-a.x, -a.y, -a.z, -a.w);
}
////////////////end of + - * /////////////////////////////////////////////


/////////////////trig func//////////////////////////////////////////////////
__forceinline__ __device__ float sin(float a)
{
    return sinf(a);
}
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


__forceinline__ __device__ float cos(float a)
{
    return cosf(a);
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


__forceinline__ __device__ float tan(float a)
{
    return tanf(a);
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

__forceinline__ __device__ float asin(float a)
{
    return asinf(a);
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

__forceinline__ __device__ float acos(float a)
{
    return acosf(a);
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

__forceinline__ __device__ float atan(float a)
{
    return atanf(a);
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

// TODO: call the real atan2f
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
__forceinline__ __device__ float pow(float a, float b)
{
    return powf(a, b);
}
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

__forceinline__ __device__ float exp(float a)
{
    return expf(a);
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


__forceinline__ __device__ float log(float a)
{
    return logf(a);
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

__forceinline__ __device__ float sqrt(float a)
{
    return sqrtf(a);
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
    return rsqrtf(a);
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
__forceinline__ __device__ float abs(float a)
{
    return float(fabsf(a));
}
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
    else if(a==0) return 0;
    else  return 1;
}
__forceinline__ __device__ float sign(float a)
{
    return m_sign(a);
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

__forceinline__ __device__ float floor(float a)
{
    return floorf(a);
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

__forceinline__ __device__ float ceil(float a)
{
    return ceilf(a);
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

__forceinline__ __device__ float mod(float a, float b)
{
    return fmodf(a, b);
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

__forceinline__ __device__ float min(float a, float b)
{
    return fminf(a, b);
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

__forceinline__ __device__ float max(float a, float b)
{
    return fmaxf(a, b);
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

__forceinline__ __device__ float clamp(float a, float b, float c)
{
    return min(max(a, b), c);
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
__forceinline__ __device__ vec2 faceforward(vec2 n, vec2 i, vec2 nref)
{
    return dot(nref, i) >= 0 ? n : -n;
}
__forceinline__ __device__ vec3 faceforward(vec3 n, vec3 i, vec3 nref)
{
    return dot(nref, i) >= 0 ? n : -n;
}
__forceinline__ __device__ vec4 faceforward(vec4 n, vec4 i, vec4 nref)
{
    return dot(nref, i) >= 0 ? n : -n;
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
    return a/(length(a));
}

__forceinline__ __device__ vec3 normalize(vec3 a)
{
    return a/(length(a));
}
__forceinline__ __device__ vec4 normalize(vec4 a)
{
    return a/(length(a));
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

__forceinline__ __device__ vec4 texture2D(cudaTextureObject_t texObj, vec2 uv)
{
    float4 res = tex2D<float4>(texObj, uv.x, uv.y);
    return vec4(res.x, res.y, res.z, res.w);
}
/////////////end of geometry math/////////////////////////////////////////////////

//__forceinline__ __device__ float cudatoglsl(float a) {
    //return a;
//}

//__forceinline__ __device__ vec2 cudatoglsl(float2 a) {
    //return vec2(a.x, a.y);
//}

//__forceinline__ __device__ vec3 cudatoglsl(float3 a) {
    //return vec3(a.x, a.y, a.z);
//}

//__forceinline__ __device__ vec4 cudatoglsl(float4 a) {
    //return vec4(a.x, a.y, a.z, a.w);
//}

//__forceinline__ __device__ float glsltocuda(float a) {
    //return a;
//}

//__forceinline__ __device__ float2 glsltocuda(vec2 a) {
    //return make_float2(a.x, a.y);
//}

//__forceinline__ __device__ float3 glsltocuda(vec3 a) {
    //return make_float3(a.x, a.y, a.z);
//}

//__forceinline__ __device__ float4 glsltocuda(vec4 a) {
    //return make_float4(a.x, a.y, a.z, a.w);
//}
