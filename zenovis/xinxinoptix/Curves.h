#pragma once

#include <optix.h>
#include <optix_device.h>

#include <cuda/curve.h>
#include "GeometryAux.h"

struct CurveAttr {
    float3 normal, tangent;
    float radius; float3 center;
    float2 uv;
};

// Compute surface normal of quadratic pimitive in world space.
static __forceinline__ __device__ CurveAttr attrLinear( float3& objPos, const int primitiveIndex )
{
    const OptixTraversableHandle gas = optixGetGASTraversableHandle();
    const unsigned int           gasSbtIndex = optixGetSbtGASIndex();
    float4                       controlPoints[2];

    optixGetLinearCurveVertexData( gas, primitiveIndex, gasSbtIndex, 0.0f, controlPoints );

    LinearInterpolator interpolator;
    interpolator.initialize(controlPoints);

    const auto u = optixGetCurveParameter();
    const float3 normal = surfaceNormal( interpolator, u, objPos );
    const float3 tangent = curveTangent( interpolator, u );
    
    return { normal, tangent, interpolator.radius(u), interpolator.position3(u) };
}

// Compute surface normal of quadratic pimitive in world space.
static __forceinline__ __device__ CurveAttr attrQuadratic( float3& objPos, const int primitiveIndex )
{
    const OptixTraversableHandle gas         = optixGetGASTraversableHandle();
    const unsigned int           gasSbtIndex = optixGetSbtGASIndex();
    float4                       controlPoints[3];

    optixGetQuadraticBSplineVertexData( gas, primitiveIndex, gasSbtIndex, 0.0f, controlPoints );

    QuadraticInterpolator interpolator;
    interpolator.initializeFromBSpline(controlPoints);

    const auto u = optixGetCurveParameter();
    const float3 normal = surfaceNormal( interpolator, u, objPos );
    const float3 tangent = curveTangent( interpolator, u );
    
    return { normal, tangent, interpolator.radius(u), interpolator.position3(u) };
}

// Compute surface normal of cubic b-spline pimitive in world space.
static __forceinline__ __device__ CurveAttr attrCubic( float3& objPos, const int primitiveIndex )
{
    const OptixTraversableHandle gas         = optixGetGASTraversableHandle();
    const unsigned int           gasSbtIndex = optixGetSbtGASIndex();
    float4                       controlPoints[4];

    optixGetCubicBSplineVertexData( gas, primitiveIndex, gasSbtIndex, 0.0f, controlPoints );

    CubicInterpolator interpolator;
    interpolator.initializeFromBSpline(controlPoints);

    const auto u = optixGetCurveParameter();
    const float3 normal = surfaceNormal( interpolator, u, objPos );
    const float3 tangent = curveTangent( interpolator, u );
    
    return { normal, tangent, interpolator.radius(u), interpolator.position3(u) };
}

// Compute surface normal of Catmull-Rom pimitive in world space.
static __forceinline__ __device__ CurveAttr attrCatrom( float3& objPos, const int primitiveIndex )
{
    const OptixTraversableHandle gas         = optixGetGASTraversableHandle();
    const unsigned int           gasSbtIndex = optixGetSbtGASIndex();
    float4                       controlPoints[4];

    optixGetCatmullRomVertexData( gas, primitiveIndex, gasSbtIndex, 0.0f, controlPoints );

    CubicInterpolator interpolator;
    interpolator.initializeFromCatrom(controlPoints);

    const auto u = optixGetCurveParameter();
    const float3 normal = surfaceNormal( interpolator, u, objPos );
    const float3 tangent = curveTangent( interpolator, u );
    
    return { normal, tangent, interpolator.radius(u), interpolator.position3(u) };
}

// Compute surface normal of Catmull-Rom pimitive in world space.
static __forceinline__ __device__ CurveAttr attrBezier( float3& objPos, const int primitiveIndex )
{
    const OptixTraversableHandle gas         = optixGetGASTraversableHandle();
    const unsigned int           gasSbtIndex = optixGetSbtGASIndex();
    float4                       controlPoints[4];

    optixGetCubicBezierVertexData( gas, primitiveIndex, gasSbtIndex, 0.0f, controlPoints );

    CubicInterpolator interpolator;
    interpolator.initializeFromBezier(controlPoints);
    
    const auto u = optixGetCurveParameter();
    const float3 normal = surfaceNormal( interpolator, u, objPos );
    const float3 tangent = curveTangent( interpolator, u );
    
    return { normal, tangent, interpolator.radius(u), interpolator.position3(u) };
}

// Compute normal
//
static __forceinline__ __device__ CurveAttr CurveAttributes( float3& objPos, OptixPrimitiveType type, const int primitiveIndex )
{
    switch( type ) {
    case OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR:
        return attrLinear( objPos, primitiveIndex );
    case OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE:
        return attrQuadratic( objPos, primitiveIndex );
    case OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE:
        return attrCubic( objPos, primitiveIndex );
    case OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM:
        return attrCatrom( objPos, primitiveIndex );
    case OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER:
        return attrBezier( objPos, primitiveIndex );
        
    case OPTIX_PRIMITIVE_TYPE_FLAT_QUADRATIC_BSPLINE:
        {
            const unsigned int           prim_idx    = optixGetPrimitiveIndex();
            const OptixTraversableHandle gas         = optixGetGASTraversableHandle();
            const unsigned int           sbtGASIndex = optixGetSbtGASIndex();
            const float2                 uv          = optixGetRibbonParameters();
            auto normal = optixGetRibbonNormal( gas, prim_idx, sbtGASIndex, 0.f /*time*/, uv );
            normal = normalize(normal);

            float4                       controlPoints[3];
            optixGetRibbonVertexData( gas, primitiveIndex, sbtGASIndex, 0.0f, controlPoints );
            QuadraticInterpolator interpolator;
            interpolator.initializeFromBSpline(controlPoints);

            const float3 tangent = curveTangent( interpolator, uv.x );
            const float radius = interpolator.radius(uv.x);
            return { normal, tangent, radius, interpolator.position3(uv.x), uv };
        }
    }
    return {};
}
