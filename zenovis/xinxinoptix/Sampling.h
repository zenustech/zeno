#pragma once 
#include <climits.h>
#include <math_constants.h>
#include <sutil/vec_math.h>

namespace pbrt {
    
inline void CoordinateSystem(const float3& a, float3& b, float3& c) {
    
   if (fabsf(a.x) > fabsf(a.y))
       b = float3{-a.z, 0, a.x} /
             sqrtf(fmaxf(__FLT_DENORM_MIN__, a.x * a.x + a.z * a.z));
   else
       b = float3{0, a.z, -a.y} /
             sqrtf(fmaxf(__FLT_DENORM_MIN__, a.y * a.y + a.z * a.z));
    
    // if (fabs(a.x) > fabs(a.y))
    //     b = float3{-a.z, 0, a.x};
    // else
    //     b = float3{0, a.z, -a.y};

    b = normalize(b);
    c = cross(a, b);
}

inline float3 SphericalDirection(float sinTheta, float cosTheta, float phi) {
    return make_float3(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);
}

inline float3 SphericalDirection(float sinTheta, float cosTheta, float phi, 
                                const float3 &x, const float3 &y, const float3 &z) {
    return sinTheta * cosf(phi) * x + sinTheta * sinf(phi) * y + cosTheta * z;
}

 inline float3 UniformSampleSphere(const float2 &uu) {
    float z = 1 - 2 * uu.x;
    float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    float phi = 2 * CUDART_PI_F * uu.y;
    return make_float3(r * cosf(phi), r * sinf(phi), z);
}

} // namespace pbrt