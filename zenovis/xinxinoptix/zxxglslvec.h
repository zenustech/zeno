#pragma once

#include <cuda/helpers.h>

__forceinline__ __device__ float to_radians(float degrees) {
    return degrees * M_PIf / 180.0f;
}
__forceinline__ __device__ float to_degrees(float radians) {
    return radians * M_1_PIf * 180.0f;
}

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
    explicit __forceinline__ __device__ operator float() const {
        return x;
    }
    vec4() = default;
    __forceinline__ __device__ operator float4() const {
        return make_float4(x, y, z, w);
    }
};

struct vec3{
    float x, y, z;

    __forceinline__ __device__ float operator[](unsigned int index) const {
        auto ptr= &this->x;
        ptr += index;
        return *ptr;
    }

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
    __forceinline__ __device__ vec3 rotX(float a) {
        return vec3(x, cos(a) * y - sin(a) * z, sin(a) * y + cos(a) * z);
    }
    __forceinline__ __device__ vec3 rotY(float a) {
        return vec3(cos(a) * x - sin(a) * z, y, cos(a) * z + sin(a) * x);
    }
    __forceinline__ __device__ vec3 rotZ(float a) {
        return vec3(cos(a) * x - sin(a) * y, cos(a) * y + sin(a) * x, z);
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

//__forceinline__ __device__ float clamp(float a, float b, float c)
//{
    //return min(max(a, b), c);
//}
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
        return (c-a)/(b-a);
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
    return a/(length(a)+1e-6f);
}

__forceinline__ __device__ vec3 normalize(vec3 a)
{
    return a/(length(a)+1e-6f);
}
__forceinline__ __device__ vec4 normalize(vec4 a)
{
    return a/(length(a)+1e-6f);
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

// __forceinline__ __device__ float textureSparse3D(nanovdb *aaa, vec3 pos)
// {

// }

/////////////end of geometry math/////////////////////////////////////////////////

////////////matrix operator...////////////////////////////////////////////////////
struct mat4{
    vec4 m0, m1, m2, m3;
    __forceinline__ __device__ mat4(
        float m00, float m01, float m02, float m03,
        float m10, float m11, float m12, float m13,
        float m20, float m21, float m22, float m23,
        float m30, float m31, float m32, float m33)
    {
        m0 = vec4(m00, m01, m02, m03);
        m1 = vec4(m10, m11, m12, m13);
        m2 = vec4(m20, m21, m22, m23);
        m3 = vec4(m30, m31, m32, m33);
    }
};

struct mat3{
    vec3 m0, m1, m2;
    __forceinline__ __device__ mat3(float m00, float m01, float m02, float m10, float m11, float m12, float m20, float m21, float m22)
    {
        m0 = vec3(m00, m01, m02);
        m1 = vec3(m10, m11, m12);
        m2 = vec3(m20, m21, m22);
    }

    explicit __forceinline__ __device__ mat3(const mat4& _v)
    {
        m0 = vec3(_v.m0);
        m1 = vec3(_v.m1);
        m2 = vec3(_v.m2);
    }
};

__forceinline__ __device__ vec3 operator*(mat3 a, vec3 b)
{
    return vec3(dot(a.m0, b), dot(a.m1, b), dot(a.m2, b));
}

__forceinline__ __device__ vec4 operator*(mat4 a, vec4 b)
{
    return vec4(dot(a.m0, b), dot(a.m1, b), dot(a.m2, b), dot(a.m3, b));
}

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

__forceinline__ __device__ vec3 normalmap(vec3 norm, float scale) {
    norm = norm * 2 - 1;
    float x = norm.x * scale;
    float y = norm.y * scale;
    float z = 1 - sqrt(x * x + y * y);
    norm.x = x;
    norm.y = y;
    norm.z = z;
    return norm;
}

__forceinline__ __device__ vec3 hsvToRgb(vec3 hsv) {
    // Reference for this technique: Foley & van Dam
    float h = hsv.x; float s = hsv.y; float v = hsv.z;
    if (s < 0.0001f) {
        return vec3 (v, v, v);
    } else {
        h = 6.0f * (h - floor(h));  // expand to [0..6)
        int hi = int(trunc(h));
        float f = h - float(hi);
        float p = v * (1.0f-s);
        float q = v * (1.0f-s*f);
        float t = v * (1.0f-s*(1.0f-f));
        if (hi == 0)
            return vec3 (v, t, p);
        else if (hi == 1)
            return vec3 (q, v, p);
        else if (hi == 2)
            return vec3 (p, v, t);
        else if (hi == 3)
            return vec3 (p, q, v);
        else if (hi == 4)
            return vec3 (t, p, v);
        return vec3 (v, p, q);
    }
}

__forceinline__ __device__ vec3 rgbToHsv(vec3 c) {
    // See Foley & van Dam
    float r = c.x; float g = c.y; float b = c.z;
    float mincomp = min (r, min(g, b));
    float maxcomp = max (r, max(g, b));
    float delta = maxcomp - mincomp;  // chroma
    float h, s, v;
    v = maxcomp;
    if (maxcomp > 0.0f)
        s = delta / maxcomp;
    else s = 0.0f;
    if (s <= 0.0f)
        h = 0.0f;
    else {
        if      (r >= maxcomp) h = (g-b) / delta;
        else if (g >= maxcomp) h = 2.0f + (b-r) / delta;
        else                   h = 4.0f + (r-g) / delta;
        h *= (1.0f/6.0f);
        if (h < 0.0f)
            h += 1.0f;
    }
    return vec3(h, s, v);
}

__forceinline__ __device__ vec2 convertTo2(float v) {
    return vec2(v, v);
}
__forceinline__ __device__ vec2 convertTo2(vec2 v) {
    return v;
}
__forceinline__ __device__ vec2 convertTo2(vec3 v) {
    return vec2(v.x, v.y);
}
__forceinline__ __device__ vec2 convertTo2(vec4 v) {
    return vec2(v.x, v.y);
}

__forceinline__ __device__ vec3 convertTo3(float v) {
    return vec3(v, v, v);
}
__forceinline__ __device__ vec3 convertTo3(vec2 v) {
    return vec3(v.x, v.y, 1);
}
__forceinline__ __device__ vec3 convertTo3(vec3 v) {
    return v;
}
__forceinline__ __device__ vec3 convertTo3(vec4 v) {
    return vec3(v.x, v.y, v.z);
}

__forceinline__ __device__ vec4 convertTo4(float v) {
    return vec4(v, v, v, v);
}
__forceinline__ __device__ vec4 convertTo4(vec2 v) {
    return vec4(v.x, v.y, 1, 1);
}
__forceinline__ __device__ vec4 convertTo4(vec3 v) {
    return vec4(v.x, v.y, v.z, 1);
}
__forceinline__ __device__ vec4 convertTo4(vec4 v) {
    return v;
}

__forceinline__ __device__ float luminance(vec3 c) {
    return dot(c, vec3(0.2722287, 0.6740818, 0.0536895));
}

__forceinline__ __device__ float safepower(float in1, float in2) {
    return sign(in1) * pow(abs(in1), in2);
}

__forceinline__ __device__ vec2 safepower(vec2 in1, vec2 in2) {
    return sign(in1) * pow(abs(in1), in2);
}
__forceinline__ __device__ vec2 safepower(vec2 in1, float in2) {
    return sign(in1) * pow(abs(in1), in2);
}

__forceinline__ __device__ vec3 safepower(vec3 in1, vec3 in2) {
    return sign(in1) * pow(abs(in1), in2);
}
__forceinline__ __device__ vec3 safepower(vec3 in1, float in2) {
    return sign(in1) * pow(abs(in1), in2);
}

__forceinline__ __device__ vec3 hsvAdjust(vec3 c, vec3 amount) {
    vec3 hsv = rgbToHsv(c);
    hsv.x = fract(hsv.x + amount.x);
    hsv.y = hsv.y * amount.y;
    hsv.z = hsv.z * amount.z;
    return hsvToRgb(hsv);
}
__forceinline__ __device__ float cosTheta(vec3 w) {
    return w.z;
}

__forceinline__ __device__ float cosTheta2(vec3 w) {
    return w.z * w.z;
}

__forceinline__ __device__ float sinTheta2(vec3 w) {
    return 1.0f - cosTheta2(w);
}

__forceinline__ __device__ float sinTheta(vec3 w) {
    return sqrt(sinTheta2(w));
}

__forceinline__ __device__ float tanTheta(vec3 w) {
    return sinTheta(w) / cosTheta(w);
}

__forceinline__ __device__ float tanTheta2(vec3 w) {
    return sinTheta2(w) / cosTheta2(w);
}

__forceinline__ __device__ float cosPhi(vec3 w) {
    float s = sinTheta(w);
    return (s == 0.0f) ? 1.0f : clamp(w.x / s, -1.0f, 1.0f);
}

__forceinline__ __device__ float sinPhi(vec3 w) {
    float s = sinTheta(w);
    return (s == 0.0f) ? 0.0f : clamp(w.y / s, -1.0f, 1.0f);
}

__forceinline__ __device__ float cosPhi2(vec3 w) {
    float c = cosPhi(w);
    return c * c;
}

__forceinline__ __device__ float sinPhi2(vec3 w) {
    float s = sinPhi(w);
    return s * s;
}

__forceinline__ __device__ float Luminance(vec3 c)
{
  return dot(c, vec3(1.0f))/3.0f;
  //return 0.2126729f * c.x + 0.7151522f * c.y + 0.0721750f * c.z;
}

__forceinline__ __device__ vec3 refract(vec3 I, vec3 N, float eta)
{
  float k = 1.0f + eta * eta * (dot(N, I) * dot(N, I) - 1);
  if (k < 0.0) {
    return vec3(0,0,0);
  }
  else
    return -eta * I + (eta * dot(N, I) - sqrt(k)) * N;
}

__forceinline__ __device__ unsigned int laine_karras_permutation(unsigned int x, unsigned int seed) {
    x += seed;
    x ^= x*0x6c50b47cu;
    x ^= x*0xb82f1e52u;
    x ^= x*0xc7afe638u;
    x ^= x*0x8d22f6e6u;
    return x;
}

__forceinline__ __device__ unsigned int reverse_bits(unsigned int x) {
    x = ((x >>  1u) & 0x55555555u) | ((x & 0x55555555u) <<  1u);
    x = ((x >>  2u) & 0x33333333u) | ((x & 0x33333333u) <<  2u);
    x = ((x >>  4u) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) <<  4u);
    x = ((x >>  8u) & 0x00ff00ffu) | ((x & 0x00ff00ffu) <<  8u);
    x = ((x >> 16u)              ) | ((x              ) << 16u);
    return x;
}
__forceinline__ __device__ unsigned int nested_uniform_scramble(unsigned int x, unsigned int seed) {
    x = reverse_bits(x);
    x = laine_karras_permutation(x, seed);
    x = reverse_bits(x);
    return x;
}
__forceinline__ __device__ unsigned int hash_combine(unsigned int seed, unsigned int v) {
    return seed ^ (v + 0x9e3779b9u + (seed << 6u) + (seed >> 2u));
}

__forceinline__ __device__ uint2 sobol_2d(unsigned int index) {
    uint2 p = make_uint2(0u);
    uint2 d = make_uint2(0x80000000u);

    for(; index != 0u; index >>= 1u) {
        if((index & 1u) != 0u) {
            p.x ^= d.x;
            p.y ^= d.y;
        }

        d.x >>= 1u;  // 1st dimension Sobol matrix, is same as base 2 Van der Corput
        d.y ^= d.y >> 1u; // 2nd dimension Sobol matrix
    }

    return p;
}

__forceinline__ __device__ vec2 shuffled_scrambled_sobol_2d(unsigned int index, unsigned int seed) {
    index = nested_uniform_scramble(index, seed);
    uint2 p = sobol_2d(index);
    p.x = nested_uniform_scramble(p.x, hash_combine(seed, 0u));
    p.y = nested_uniform_scramble(p.y, hash_combine(seed, 1u));
    return vec2(p.x, p.y)*exp2(-32.);
}