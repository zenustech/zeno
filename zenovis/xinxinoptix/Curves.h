#pragma once

#include <optix.h>
#include <optix_device.h>

#include <cuda/curve.h>
#include "GeometryAux.h"

// Get curve hit-point in world coordinates.
__forceinline__ __device__ float3 getHitPoint()
{
    const float  t            = optixGetRayTmax();
    const float3 rayOrigin    = optixGetWorldRayOrigin();
    const float3 rayDirection = optixGetWorldRayDirection();

    return rayOrigin + t * rayDirection;
}

struct CurveAttr {
    float3 normal, tangent;
    float radius; float3 center;
};

// Compute surface normal of quadratic pimitive in world space.
static __forceinline__ __device__ CurveAttr attrLinear( const int primitiveIndex )
{
    const OptixTraversableHandle gas = optixGetGASTraversableHandle();
    const unsigned int           gasSbtIndex = optixGetSbtGASIndex();
    float4                       controlPoints[2];

    optixGetLinearCurveVertexData( gas, primitiveIndex, gasSbtIndex, 0.0f, controlPoints );

    LinearInterpolator interpolator;
    interpolator.initialize(controlPoints);

    float3 hitPoint = optixTransformPointFromWorldToObjectSpace( getHitPoint() );

    const auto u = optixGetCurveParameter();
    const float3 normal = surfaceNormal( interpolator, u, hitPoint );
    const float3 tangent = curveTangent( interpolator, u );
    
    return { normal, tangent, interpolator.radius(u), interpolator.position3(u) };
}

// Compute surface normal of quadratic pimitive in world space.
static __forceinline__ __device__ CurveAttr attrQuadratic( const int primitiveIndex )
{
    const OptixTraversableHandle gas         = optixGetGASTraversableHandle();
    const unsigned int           gasSbtIndex = optixGetSbtGASIndex();
    float4                       controlPoints[3];

    optixGetQuadraticBSplineVertexData( gas, primitiveIndex, gasSbtIndex, 0.0f, controlPoints );

    QuadraticInterpolator interpolator;
    interpolator.initializeFromBSpline(controlPoints);

    float3 hitPoint = optixTransformPointFromWorldToObjectSpace( getHitPoint() );

    const auto u = optixGetCurveParameter();
    const float3 normal = surfaceNormal( interpolator, u, hitPoint );
    const float3 tangent = curveTangent( interpolator, u );
    
    return { normal, tangent, interpolator.radius(u), interpolator.position3(u) };
}

// Compute surface normal of cubic b-spline pimitive in world space.
static __forceinline__ __device__ CurveAttr attrCubic( const int primitiveIndex )
{
    const OptixTraversableHandle gas         = optixGetGASTraversableHandle();
    const unsigned int           gasSbtIndex = optixGetSbtGASIndex();
    float4                       controlPoints[4];

    optixGetCubicBSplineVertexData( gas, primitiveIndex, gasSbtIndex, 0.0f, controlPoints );

    CubicInterpolator interpolator;
    interpolator.initializeFromBSpline(controlPoints);

    float3 hitPoint = optixTransformPointFromWorldToObjectSpace( getHitPoint() );

    const auto u = optixGetCurveParameter();
    const float3 normal = surfaceNormal( interpolator, u, hitPoint );
    const float3 tangent = curveTangent( interpolator, u );
    
    return { normal, tangent, interpolator.radius(u), interpolator.position3(u) };
}

// Compute surface normal of Catmull-Rom pimitive in world space.
static __forceinline__ __device__ CurveAttr attrCatrom( const int primitiveIndex )
{
    const OptixTraversableHandle gas         = optixGetGASTraversableHandle();
    const unsigned int           gasSbtIndex = optixGetSbtGASIndex();
    float4                       controlPoints[4];

    optixGetCatmullRomVertexData( gas, primitiveIndex, gasSbtIndex, 0.0f, controlPoints );

    CubicInterpolator interpolator;
    interpolator.initializeFromCatrom(controlPoints);

    float3 hitPoint = optixTransformPointFromWorldToObjectSpace( getHitPoint() );

    const auto u = optixGetCurveParameter();
    const float3 normal = surfaceNormal( interpolator, u, hitPoint );
    const float3 tangent = curveTangent( interpolator, u );
    
    return { normal, tangent, interpolator.radius(u), interpolator.position3(u) };
}

// Compute surface normal of Catmull-Rom pimitive in world space.
static __forceinline__ __device__ CurveAttr attrBezier( const int primitiveIndex )
{
    const OptixTraversableHandle gas         = optixGetGASTraversableHandle();
    const unsigned int           gasSbtIndex = optixGetSbtGASIndex();
    float4                       controlPoints[4];

    optixGetCubicBezierVertexData( gas, primitiveIndex, gasSbtIndex, 0.0f, controlPoints );

    CubicInterpolator interpolator;
    interpolator.initializeFromBezier(controlPoints);

    float3 hitPoint = optixTransformPointFromWorldToObjectSpace( getHitPoint() );
    
    const auto u = optixGetCurveParameter();
    const float3 normal = surfaceNormal( interpolator, u, hitPoint );
    const float3 tangent = curveTangent( interpolator, u );
    
    return { normal, tangent, interpolator.radius(u), interpolator.position3(u) };
}

// Compute normal
//
static __forceinline__ __device__ CurveAttr CurveAttributes( OptixPrimitiveType type, const int primitiveIndex )
{
    switch( type ) {
    case OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR:
        return attrLinear( primitiveIndex );
    case OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE:
        return attrQuadratic( primitiveIndex );
    case OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE:
        return attrCubic( primitiveIndex );
    case OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM:
        return attrCatrom( primitiveIndex );
    case OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER:
        return attrBezier( primitiveIndex );
        
    case OPTIX_PRIMITIVE_TYPE_FLAT_QUADRATIC_BSPLINE:
        {
            const unsigned int           prim_idx    = optixGetPrimitiveIndex();
            const OptixTraversableHandle gas         = optixGetGASTraversableHandle();
            const unsigned int           sbtGASIndex = optixGetSbtGASIndex();
            const float2                 uv          = optixGetRibbonParameters();
            auto normal = optixGetRibbonNormal( gas, prim_idx, sbtGASIndex, 0.f /*time*/, uv );

            float4                       controlPoints[3];
            optixGetRibbonVertexData( gas, primitiveIndex, sbtGASIndex, 0.0f, controlPoints );
            QuadraticInterpolator interpolator;
            interpolator.initializeFromBSpline(controlPoints);

            const float3 tangent = curveTangent( interpolator, uv.x );
            const float radius = interpolator.radius(uv.x);
            return { normal, tangent, radius, interpolator.position3(uv.x) };
        }
    }
    return {};
}
