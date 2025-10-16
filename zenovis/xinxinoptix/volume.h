#pragma once

#include <Sampling.h>

#ifndef __CUDACC_RTC__
#include <Host.h>
#endif

namespace nanovdb {
    using Fp32 = float;
};

struct VolumeIn {
    float3 pos_world;
    float3 pos_view;

    bool isShadowRay;

    float sigma_t;
    uint32_t* seed;

    void* sbt_ptr;
    float* world2object;

    float3 _local_pos_ = make_float3(CUDART_NAN_F);
    float3 _uniform_pos_ = make_float3(CUDART_NAN_F);
};

struct VolumeOut {
    float step_scale=__FLT_MAX__;

    float max_density;
    float density;

    float3 albedo;
    float3 extinction;
    float albedoAmp;
    
    float anisotropy;
    float3 emission;
};

namespace pbrt {

struct HenyeyGreenstein {
    float g, gg;
    __device__ HenyeyGreenstein(float g) : g(g), gg(g*g) {}
    
    float p(const float3 &wo, const float3 &wi) const;
    float sample(const float3 &wo, float3 &wi, const float2 &uu) const;
};

// Media Inline Functions
inline float PhaseHG(float cosTheta, float g, float gg) {

    float denom = 1 + gg + 2 * g * cosTheta;

    if (denom < __FLT_EPSILON__) {
        return 1.0f;
    }

    auto P = (0.25f / M_PIf) * (1 - gg) / (denom * sqrtf(denom));
    return clamp(P, 0.0f, 1.0f);
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