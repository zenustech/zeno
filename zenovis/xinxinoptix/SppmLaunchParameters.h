#pragma once

#include <cuda/cstdint.h>


// Plain storage shared by C++ and NVRTC. All positions are camera relative,
// like the OptiX acceleration structures and radiance payloads.
struct SppmMaterial {
    float3 basecolor, transColor, diffractColor;
    float roughness, anisotropic, anisoRotation, ior, metallic;
    float specular, specularTint, sheen, sheenTint;
    float clearcoat, clearcoatRoughness, clearcoatIOR;
    float specTrans, F0, diffraction, thin;
};

struct SppmPixel {
    float3 direct, tau;
    // Independent pilot moments select camera effort, never radiance weights.
    double cameraMean, cameraM2;
    uint32_t cameraSamples;
    float radius, n;
    float3 phi;
    uint32_t m, visible;
    float3 position, normal, shadingNormal, tangent, bitangent, wo, beta;
    SppmMaterial material;
    int3 cell;
    uint32_t next;
};

struct SppmStatistics {
    unsigned long long emitted, traced, gathered, visible, unsupported, caustic, absorbed;
    // Positive float bit patterns support atomicMin during the first camera pass.
    uint32_t cameraDepthBits, casterDepthBits;
    float initialRadius, radiusReferenceDepth;
    unsigned long long usefulPhotonPaths;
    uint32_t guideUpdates;
};

constexpr uint32_t SppmGuideSide = 8u;
constexpr uint32_t SppmGuideBins = SppmGuideSide * SppmGuideSide;

// AABB followed by its per-primitive SBT index; 32-byte stride meets OptiX alignment.
struct SppmSearchBounds {
    float minX, minY, minZ, maxX, maxY, maxZ;
    uint32_t record, padding;
};

struct SppmLaunchParameters {
    SppmPixel* pixels;
    uint32_t* heads;
    SppmStatistics* statistics;
    uint32_t enabled, iteration, photonsPerIteration, hashSize;
    float initialRadius, alpha, radiusPixels;
    float3 sceneCenter;
    float sceneRadius;
    uint32_t emitterCount, distantCount, environment;
    // Joint distribution over emitter and its uniform 2D emission coordinates.
    double* guideCdf;
    double* guideScores;
    uint32_t guiding, guideCells, guideActive;
    SppmSearchBounds* searchBounds;
    unsigned long long searchHandle;
    uint32_t searchBackend, searchRecord;
    uint32_t cameraSobol[8], cacheCameraSobol;
};

// Cell keys are retained in each node: bucket collisions must never count a
// visible point twice when querying the 27 neighboring cells.
#ifdef __CUDACC_RTC__
static __forceinline__ __device__ uint32_t sppmGuideSelect(const double* cdf, uint32_t count, float u) {
    uint32_t lo=0u, hi=count;
    while (lo<hi) {
        const uint32_t mid=lo+(hi-lo)/2u;
        if (cdf[mid] <= double(u)) lo=mid+1u; else hi=mid;
    }
    return min(lo,count-1u);
}
static __forceinline__ __device__ float sppmGuideWeight(const double* cdf, uint32_t bin) {
    const double lower = bin ? cdf[bin-1u] : 0.0;
    return float(1.0/(double(SppmGuideBins)*(cdf[bin]-lower)));
}
static __forceinline__ __device__ uint32_t sppmHash(int3 c, uint32_t size) {
    return ((uint32_t(c.x) * 73856093u) ^ (uint32_t(c.y) * 19349663u) ^
            (uint32_t(c.z) * 83492791u)) & (size - 1u);
}
static __forceinline__ __device__ int3 sppmCell(float3 p, float radius) {
    return make_int3(int(floorf(p.x / radius)), int(floorf(p.y / radius)),
                     int(floorf(p.z / radius)));
}
#endif
