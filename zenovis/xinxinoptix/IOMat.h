#pragma once

#include "Sampling.h"
#include "zxxglslvec.h"
#include <cuda_fp16.h>
#include <optix_device.h>

struct MatOutput {
    vec3 basecolor;
    float roughness;
    float opacity;
    float thin;
    float flatness;
    float doubleSide;
    float anisotropic;
    float anisoRotation;
    float ior;
    vec3 reflectance;

    float metallic;
    vec3 metalColor;

    float specular;
    float specularTint;
    float sheen;
    float sheenTint;

    float clearcoat;
    vec3 clearcoatColor;
    float clearcoatRoughness;
    float clearcoatIOR;

    float specTrans;
    vec3 transColor;
    vec3 transTint;
    float transTintDepth;
    float transDistance;
    vec3 transScatterColor;

    float diffraction;
    vec3  diffractColor;

    float subsurface;
    vec3  sssColor;
    vec3  sssParam;
    bool sssFxiedRadius;
    float scatterDistance;
    float scatterStep;
    float smoothness;
    float displacement;
    float shadowReceiver;
    float shadowTerminatorOffset;
    float isHair;
    vec3  mask_value;

    vec3 nrm;
    vec3 emission;
};

__forceinline__ float3 transformPoint(float3 p, const float4* matrix) {
    const auto& m0 = *(matrix+0);
    const auto& m1 = *(matrix+1);
    const auto& m2 = *(matrix+2);
    return optix_impl::optixTransformPoint(m0, m1, m2, p);
}

__forceinline__ float3 transformVector(float3 v, const float4* matrix) {
    const auto& m0 = *(matrix+0);
    const auto& m1 = *(matrix+1);
    const auto& m2 = *(matrix+2);
    return optix_impl::optixTransformVector(m0, m1, m2, v);
}

__forceinline__ float3 transformNormal(float3 n, const float4* matrix) {
    const auto& m0 = *(matrix+0);
    const auto& m1 = *(matrix+1);
    const auto& m2 = *(matrix+2);
    return optix_impl::optixTransformNormal(m0, m1, m2, n);
}

struct MatInput {
    OptixPrimitiveType ptype;
    uint64_t gas;
    uint32_t sbtIdx;
    uint32_t priIdx;
    uint32_t instIdx;
    vec3 ray;
    vec3 pOffset = vec3(0);
    int depth;

    uint32_t seed;

    float3 objPos;
    float3 objNorm;
    float3 wldPos;
    float3 wldNorm;

    float2 barys2;
    float3* vertices;
    uint3 vertex_idx;

    float4 objectToWorld[3];
    float4 worldToObject[3];

    __forceinline__ auto getGasPointer() const {
        return (void**)optixGetGASPointerFromHandle(gas);
    }

    float rayLength;
    bool isBackFace  : 1;
    bool isShadowRay : 1;
    
    vec3 reflectance;

    vec3 N;
    vec3 T;
    vec3 B;

    vec3 V;
    vec3 fresnel;
};

struct TriangleInput : MatInput {

    __forceinline__ float3 barys() const {
        return { 1.0f-barys2.x-barys2.y, barys2.x, barys2.y };
    }

    inline vec3 interpNorm(float smooth=0.0f) const {
        let gas_ptr = getGasPointer();
        let nrm_ptr = reinterpret_cast<const ushort3*>(*(gas_ptr-4) );
        if (nrm_ptr == nullptr) { return wldNorm; }

        float3 n0 = decodeHalf( nrm_ptr[ vertex_idx.x ] );
        float3 n1 = decodeHalf( nrm_ptr[ vertex_idx.y ] );
        float3 n2 = decodeHalf( nrm_ptr[ vertex_idx.z ] );

        if (smooth > 0.0f) {
            n0 = dot(n0, objNorm)>(1-smooth)?n0:objNorm;
            n1 = dot(n1, objNorm)>(1-smooth)?n1:objNorm;
            n2 = dot(n2, objNorm)>(1-smooth)?n2:objNorm;
        }

        auto tmp = interp(barys2, n0, n1, n2);
        if (tmp.x==0 && tmp.y==0 && tmp.z==0) return wldNorm;
        tmp = transformNormal(tmp, worldToObject);
        return normalize(tmp);
    }

    inline vec3 interpTang() const {
        let gas_ptr = getGasPointer();
        let tan_ptr = reinterpret_cast<const ushort3*>(*(gas_ptr-5) );
        if (tan_ptr == nullptr) { return {}; }

        auto t0 = decodeHalf( tan_ptr[ vertex_idx.x ] );
        auto t1 = decodeHalf( tan_ptr[ vertex_idx.y ] );
        auto t2 = decodeHalf( tan_ptr[ vertex_idx.z ] );

        auto tmp = interp(barys2, t0, t1, t2);
        if (tmp.x==0 && tmp.y==0 && tmp.z==0) return {};
        tmp = transformNormal(tmp, worldToObject);
        return normalize(tmp);
    }
    
    inline vec3 uv() const {
        let gas_ptr = getGasPointer();
        let uv_ptr  = reinterpret_cast<const float2*>( *(gas_ptr-2) );
        if (uv_ptr == nullptr) { return {}; }

        const auto& uv0 = uv_ptr[ vertex_idx.x ];
        const auto& uv1 = uv_ptr[ vertex_idx.y ];
        const auto& uv2 = uv_ptr[ vertex_idx.z ];

        auto tmp = interp(barys2, uv0, uv1, uv2);
        return {tmp.x, tmp.y, 0.0f};
    }
    inline vec3 clr() const {
        let gas_ptr = getGasPointer();
        let clr_ptr = reinterpret_cast<const ushort3*>(*(gas_ptr-3) );
        if (clr_ptr == nullptr) { return {}; }

        auto clr0 = decodeHalf( clr_ptr[ vertex_idx.x ] );
        auto clr1 = decodeHalf( clr_ptr[ vertex_idx.y ] );
        auto clr2 = decodeHalf( clr_ptr[ vertex_idx.z ] );

        return interp(barys2, clr0, clr1, clr2);
    }
    
    inline float area(bool local=false) const {
        auto e0 = vertices[1] - vertices[0];
        auto e1 = vertices[2] - vertices[0];

        if (!local) {
            e0 = transformVector(e0, objectToWorld);
            e1 = transformVector(e1, objectToWorld);
        }
        return 0.5f*length(cross(e0,e1));
    }
    inline float3 els(bool local=false) const {
        auto e0 = vertices[1] - vertices[0];
        auto e1 = vertices[2] - vertices[1];
        auto e2 = vertices[0] - vertices[2];

        if (!local) {
            e0 = transformVector(e0, objectToWorld);
            e1 = transformVector(e1, objectToWorld);
            e2 = transformVector(e2, objectToWorld);
        }
        auto res = vec3(length(e0), length(e1), length(e2));
        res = area(local) / (res+0.000001);
        return res;
    }
};

struct SphereInput : MatInput {

    inline vec3 interpNorm(float smooth=0.0f) const {
        return wldNorm;
    }
    inline vec3 interpTang() const {
        return {};
    }
    inline vec3 uv() const {
        return sphereUV(objNorm, false);
    }
    inline vec3 clr() const {
        let gas_ptr = getGasPointer();
        let clr_ptr = (float3*)( *(gas_ptr-1) );
        if (clr_ptr == nullptr) return {0,0,0};
        return clr_ptr[priIdx]; 
    }
    inline float area(bool local=false) const {
        return 0.0f;
    }
    inline float3 els(bool local=false) const {
        return {};
    }
};

template<typename Func>
inline auto dispatch(const MatInput* input, int16_t ptype, Func&& func) {
    switch (ptype) {
        case OPTIX_PRIMITIVE_TYPE_TRIANGLE:
            return func(*reinterpret_cast<const TriangleInput*>(input));
        case OPTIX_PRIMITIVE_TYPE_SPHERE:
            return func(*reinterpret_cast<const SphereInput*>(input));
        default: {
            using ReturnT = decltype(func(*reinterpret_cast<const SphereInput*>(input)));
            return ReturnT{};
        }
    }
}

struct WrapperInput : MatInput {
    inline vec3 interpNorm(float smooth=0.0f) const {
        return dispatch(this, ptype, [&](const auto& in) { return in.interpNorm(smooth); });
    }
    inline vec3 interpTang() const {
        return dispatch(this, ptype, [&](const auto& in) { return in.interpTang(); });
    }
    inline vec3 uv() const {
        return dispatch(this, ptype, [&](const auto& in) { return in.uv(); });
    }
    inline vec3 clr() const {
        return dispatch(this, ptype, [&](const auto& in) { return in.clr(); });
    }
    inline float area(bool local=false) const {
        return dispatch(this, ptype, [&](const auto& in) { return in.area(local); });
    }
    inline float3 els(bool local=false) const {
        return dispatch(this, ptype, [&](const auto& in) { return in.els(local); });
    }
};
