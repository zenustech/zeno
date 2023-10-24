#pragma once 
#include <math_constants.h>
#include <sutil/vec_math.h>

#include <cuda/climits.h>
#include <cuda/cstdint.h>

#ifdef __CUDACC_RTC__
    #include "zxxglslvec.h"
    using Vector3f = vec3;
#else 
    #include "Host.h"
    #include <zeno/utils/vec.h>
    using Vector3f = zeno::vec<3, float>;
#endif

#ifdef __CUDACC_DEBUG__
    #define DCHECK assert
#else

#define DCHECK(x) \
    do {          \
    } while (false) /* swallow semicolon */

#endif

namespace pbrt {
    
template <typename T>
inline float Sqr(T v) { return v * v; }

inline float SafeASin(float x) {
    DCHECK(x >= -1.0001 && x <= 1.0001);
    return asinf(clamp(x, -1.0f, 1.0f));
}
 inline float SafeACos(float x) {
    DCHECK(x >= -1.0001 && x <= 1.0001);
    return acosf(clamp(x, -1.0f, 1.0f));
}

inline float SafeSqrt(float x) {
    DCHECK(x > -1e-3f);  // not too negative
    return sqrtf(fmaxf(0.f, x));
}

inline float AbsDot(Vector3f v, Vector3f n) {
    return abs(dot(v, n));
}

inline float AngleBetween(Vector3f v1, Vector3f v2) {
    if (dot(v1, v2) < 0)
        return M_PIf - 2 * SafeASin(length(v1 + v2) / 2);
    else
        return 2 * SafeASin(length(v2 - v1) / 2);
}

inline float Radians(float deg) {
    return (M_PIf / 180) * deg;
}

inline float Degrees(float rad) {
    return (180 / M_PIf) * rad;
}

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
    
//    if (fabsf(a.x) > fabsf(a.y))
//        b = float3{-a.z, 0, a.x} /
//              sqrtf(fmaxf(__FLT_DENORM_MIN__, a.x * a.x + a.z * a.z));
//    else
//        b = float3{0, a.z, -a.y} /
//              sqrtf(fmaxf(__FLT_DENORM_MIN__, a.y * a.y + a.z * a.z));
    
    if (fabs(a.x) > fabs(a.y))
        b = float3{-a.z, 0, a.x};
    else
        b = float3{0, a.z, -a.y};

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
    float phi = 2 * M_PIf * uu.y;
    return make_float3(r * cosf(phi), r * sinf(phi), z);
}

} // namespace pbrt


namespace rtgems {

    constexpr float origin()      { return 1.0f / 16.0f; }
    constexpr float int_scale()   { return 3.0f * 256.0f; }
    constexpr float float_scale() { return 3.0f / 65536.0f; }
    
    // Normal points outward for rays exiting the surface, else is flipped.
    static __inline__ __device__ float3 offset_ray(const float3 p, const float3 n)
    {
        int3 of_i {
            (int)(int_scale() * n.x),
            (int)(int_scale() * n.y), 
            (int)(int_scale() * n.z) };

        float3 p_i {
            __int_as_float(__float_as_int(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
            __int_as_float(__float_as_int(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
            __int_as_float(__float_as_int(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)) };

        return float3{
                fabsf(p.x) < origin() ? p.x+float_scale()*n.x : p_i.x,
                fabsf(p.y) < origin() ? p.y+float_scale()*n.y : p_i.y,
                fabsf(p.z) < origin() ? p.z+float_scale()*n.z : p_i.z };
    }
}

static __host__ __device__ __inline__ float3 sphereUV(const float3 &dir, bool internal) {
//https://en.wikipedia.org/wiki/UV_mapping

    auto x = internal? dir.x:-dir.x;

    auto u = 0.5f + atan2f(dir.z, x) * 0.5f / M_PIf;
    auto v = 0.5f + asinf(dir.y) / M_PIf;

    return float3 {u, v, 0.0f};
} 

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