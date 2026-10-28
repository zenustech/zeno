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
    uint32_t photonRecords;
    uint32_t causticPhotonRecords;
    unsigned long long fgInitial, fgTemporal, fgSpatial;
};

// Opaque FG evaluation never reads transmission colors, diffraction or thin
// surface state. Retain full precision for every parameter it does read.
struct FgMaterial {
    float3 basecolor;
    float roughness, anisotropic, anisoRotation, ior, metallic;
    float specular, specularTint, sheen, sheenTint;
    float clearcoat, clearcoatRoughness, clearcoatIOR, specTrans;
};
struct FgSurface {
    float3 position, normal, shadingNormal, tangent, bitangent, wo;
    FgMaterial material;
    uint32_t valid, depth;
    float3 extinction;
};
struct FgPhoton {
    float3 position, normal, wi, power;
    uint32_t caustic;
};
struct FgCandidate {
    FgSurface point;
    float pdf;
};
struct FgReservoir {
    FgSurface point;
    // A flux-weighted photon sample estimates the directional incident field.
    // Keeping its direction lets reconnection reevaluate the full Disney BSDF.
    float3 incident, photonDirection;
    // Exact outgoing radiance for the renderer's purely Lambertian closure.
    // Direction-dependent closures retain the directional photon estimator.
    float3 diffuseRadiance;
    uint32_t diffuse;
    float weight, target, jacobian;
    uint32_t confidence, valid;
};
struct FgCausticReservoir {
    FgPhoton photon;
    float weight, target;
    uint32_t confidence, valid;
};
struct FgReservoirPair {
    FgSurface source;
    FgReservoir fg;
    FgCausticReservoir caustic;
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
    uint32_t statisticsEnabled;
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
    uint32_t fgEnabled, fgReuse, fgCull;
    uint32_t fgConfidence, fgSpatialSamples, fgPairwiseOutput;
    uint32_t fgPhotonCapacity, fgBuildBounds;
    FgPhoton* fgPhotons;
    FgCandidate* fgCandidates;
    FgReservoirPair *fgInitial, *fgPrevious, *fgCurrent;
    uint32_t cameraSobol[8], cacheCameraSobol;
};

// Cell keys are retained in each node: bucket collisions must never count a
// visible point twice when querying the 27 neighboring cells.
#ifdef __CUDACC_RTC__
template<class Material>
static __forceinline__ __device__ bool fgLambertian(const Material& m) {
    // EvalDisneyDiffuse is Lambertian with no sheen. Directional closures
    // must keep the full Disney evaluation and directional photon estimator.
    return m.specular==0.0f && m.metallic==0.0f && m.specTrans==0.0f &&
        m.sheen==0.0f && m.clearcoat==0.0f;
}
// Front-side Lambertian closure, including EvaluateDisney3's grazing floor.
static __forceinline__ __device__ float3 fgDiffuseBsdf(float3 base, float cosine) {
    const float scale=cosine>0.0f ? fmaxf(.001f,cosine)*0.3183098861837907f : 0.0f;
    return make_float3(base.x*scale,base.y*scale,base.z*scale);
}
struct FgPairWeights { double current, other; };
static __forceinline__ __device__ FgPairWeights fgPairWeights(
    double target0, double target1, double target0Elsewhere, double target1Here,
    double jacobian0, double jacobian1, double weight0, double weight1,
    uint32_t confidence0, uint32_t confidence1) {
    const double p0=target0*confidence0, q0=target0Elsewhere*jacobian0*confidence1;
    const double p1=target1*confidence1, q1=target1Here*jacobian1*confidence0;
    return {p0+q0>0 ? p0/(p0+q0)*target0*weight0 : 0,
            p1+q1>0 ? p1/(p1+q1)*target1Here*jacobian1*weight1 : 0};
}
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
