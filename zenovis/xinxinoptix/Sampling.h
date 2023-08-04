#pragma once 
#include <cuda/climits.h>
#include <math_constants.h>
#include <sutil/vec_math.h>

#include <nanovdb/NanoVDB.h>

namespace pbrt {

template <typename T>
inline float Sqr(T v) { return v * v; }

inline void CoordinateSystem(const float3& v1, float3 *v2, float3 *v3) {

    float _sign_ = copysignf(float(1), v1.z);
    float a = -1 / (_sign_ + v1.z);
    float b = v1.x * v1.y * a;
    *v2 = make_float3(1 + _sign_ * Sqr(v1.x) * a, _sign_ * b, -_sign_ * v1.x);
    *v3 = make_float3(b, _sign_ + Sqr(v1.y) * a, -v1.y);

    normalize(*v2);
    normalize(*v3);
}
    
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

// *Really* minimal PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)

// typedef struct { uint64_t state;  uint64_t inc; } pcg32_random_t;

// inline uint32_t pcg32_random_r(pcg32_random_t* rng)
// {
//     uint64_t oldstate = rng->state;
//     // Advance internal state
//     rng->state = oldstate * 6364136223846793005ULL + (rng->inc|1);
//     // Calculate output function (XSH RR), uses old state for max ILP
//     uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
//     uint32_t rot = oldstate >> 59u;
//     return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
// }

//https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
static __host__ __device__ __inline__ uint32_t pcg_hash(uint32_t &seed )
{
    auto state = seed * 747796405u + 2891336453u;
    auto word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

static __host__ __device__ __inline__ uint32_t pcg_rng(uint32_t &seed) 
{
	auto state = seed;
	seed = seed * 747796405u + 2891336453u;
	auto word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

static inline uint32_t hash_iqnt2d(const uint32_t x, const uint32_t y)
{
    const uint32_t qx = 1103515245U * ((x >> 1U) ^ (y));
    const uint32_t qy = 1103515245U * ((y >> 1U) ^ (x));
    const uint32_t n = 1103515245U * ((qx) ^ (qy >> 3U));

    return n;
}