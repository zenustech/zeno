#pragma once

#include <Sampling.h>
#include <nanovdb/NanoVDB.h>

#ifndef __CUDACC_RTC__
#include <Host.h>
#endif

namespace nanovdb {    
    using Float = float;
    using Float3 = Vec3f;
    using Float4 = Vec4f;

    using Double = double;
    
    using Short = int16_t;

    using Int = int32_t;
    using Int3 = Vec3i;
    using Int4 = Vec4i;

    using Long = int64_t;
};

struct VolumeIn {
    float3 pos_view;

    uint32_t* seed;
    void* sbt_ptr;
    uint8_t forceNearestVDBSampling;
    
    float4 objectToWorld[3];
    float4 worldToObject[3];
};

struct VolumeOut {
    __half density;
    __half anisotropy;
    float albedoAmp;

    float3 albedo;
    float3 extinction;
    float3 emission;
};

namespace pbrt {

struct HenyeyGreenstein {
    float g, gg;
    __device__ HenyeyGreenstein(float g)
        : g(fminf(fmaxf(g, -0.999f), 0.999f)), gg(this->g * this->g) {}
    
    float p(const float3 &wo, const float3 &wi) const;
    float sample(const float3 &wo, float3 &wi, const float2 &uu) const;
};

// Media Inline Functions
inline float PhaseHG(float cosTheta, float g, float gg) {

    cosTheta = fminf(fmaxf(cosTheta, -1.0f), 1.0f);
    float denom = fmaxf(1 + gg + 2 * g * cosTheta, 1e-20f);

    auto P = (0.25f / M_PIf) * (1 - gg) / (denom * sqrtf(denom));
    return fmaxf(P, 0.0f);
}

// HenyeyGreenstein Method Definitions
inline float HenyeyGreenstein::p(const float3 &wo, const float3 &wi) const {
    return PhaseHG(dot(wo, wi), g, gg);
}

inline float HenyeyGreenstein::sample(const float3 &wo, float3 &wi, const float2 &uu) const {
    // Compute $\cos \theta$ for Henyey--Greenstein sample

    if (fabsf(g) >= 1.0f) { 
        wi = copysignf(1.0f, -g) * wo;
        return 1.0f;
    }

    float cosTheta;
    if (fabs(g) < 1e-3f)
        cosTheta = 1 - 2 * uu.x;
    else {
        float sqrTerm = (1 - gg) / (1 + g - 2 * g * uu.x);
        cosTheta = -(1 + gg - sqrTerm * sqrTerm) / (2 * g);
    }

    // Compute direction _wi_ for Henyey--Greenstein sample
    float sinTheta = sqrtf(fmax(0.0f, 1.0f - cosTheta * cosTheta));
    float phi = 2 * M_PIf * uu.y;
    
    float3 v1, v2;
    CoordinateSystem(wo, v1, v2);
    //CoordinateSystem(wo, &v1, &v2);
    wi = SphericalDirection(sinTheta, cosTheta, phi, v1, v2, wo);

    return PhaseHG(cosTheta, g, gg);
}

} // namespace pbrt

inline int3 interp_trilinear_stochastic(const float3& P, float randu)
{
    const float ix = floorf(P.x);
    const float iy = floorf(P.y);
    const float iz = floorf(P.z);
    int idx[3] = {(int)ix, (int)iy, (int)iz};

    const float tx = P.x - ix;
    const float ty = P.y - iy;
    const float tz = P.z - iz;

    if (randu < tx) {
        idx[0]++;
        randu /= tx;
    }
    else {
        randu = (randu - tx) / (1 - tx);
    }

    if (randu < ty) {
        idx[1]++;
        randu /= ty;
    }
    else {
        randu = (randu - ty) / (1 - ty);
    }

    if (randu < tz) {
        idx[2]++;
    }

    return make_int3(idx[0], idx[1], idx[2]);
}

inline float3 interp_triquadratic_to_trilinear_stochastic(const float3& P, float randu)
{
    const float3 p = floor(P);
    const float3 t = P - p;

    // Corrected quadratic B-spline weights
    const float3 w_minus1 = 0.5f * (1.0f - t) * (1.0f - t);
    const float3 w_0 = 0.5f + t - t * t;
    const float3 w_plus1 = 0.5f * t * t;

    const float3 g0 = w_minus1 + w_0;

    const float3 P0 = p + (w_0 / g0) - 1.0f;
    const float3 P1 = p + 1.0f;

    float3 Pnew = P0;

    if (randu < g0.x) {
        randu /= g0.x;
    }
    else {
        Pnew.x = P1.x;
        randu = (randu - g0.x) / (1.0f - g0.x);
    }

    if (randu < g0.y) {
        randu /= g0.y;
    }
    else {
        Pnew.y = P1.y;
        randu = (randu - g0.y) / (1.0f - g0.y);
    }

    if (randu < g0.z) {
        // stay with P0.z
    }
    else {
        Pnew.z = P1.z;
    }

    return Pnew;
}

inline float3 interp_tricubic_to_trilinear_stochastic(const float3& P, float randu)
{
    const float3 p = floor(P);
    const float3 t = P - p;

    /* Cubic weights. */
    const float3 w0 = (1.0f / 6.0f) * (t * (t * (-t + 3.0f) - 3.0f) + 1.0f);
    const float3 w1 = (1.0f / 6.0f) * (t * t * (3.0f * t - 6.0f) + 4.0f);
    //    float3 w2 = (1.0f / 6.0f) * (t * (t * (-3.0f * t + 3.0f) + 3.0f) + 1.0f);
    const float3 w3 = (1.0f / 6.0f) * (t * t * t);

    const float3 g0 = w0 + w1;
    const float3 P0 = p + (w1 / g0) - 1.0f;
    const float3 P1 = p + (w3 / (make_float3(1.0f) - g0)) + 1.0f;

    float3 Pnew = P0;

    if (randu < g0.x) {
        randu /= g0.x;
    }
    else {
        Pnew.x = P1.x;
        randu = (randu - g0.x) / (1 - g0.x);
    }

    if (randu < g0.y) {
        randu /= g0.y;
    }
    else {
        Pnew.y = P1.y;
        randu = (randu - g0.y) / (1 - g0.y);
    }

    if (randu < g0.z) {
    }
    else {
        Pnew.z = P1.z;
    }

    return Pnew;
}