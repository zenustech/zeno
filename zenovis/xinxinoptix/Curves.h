#pragma once

#include <optix.h>
#include <optix_device.h>

#include <cuda/curve.h>
#include "GeometryAux.h"

// Get curve hit-point in world coordinates.
static __forceinline__ __device__ float3 getHitPoint()
{
    const float  t            = optixGetRayTmax();
    const float3 rayOrigin    = optixGetWorldRayOrigin();
    const float3 rayDirection = optixGetWorldRayDirection();

    return rayOrigin + t * rayDirection;
}

// Compute surface normal of quadratic pimitive in world space.
static __forceinline__ __device__ float3 normalLinear( const int primitiveIndex )
{
    const OptixTraversableHandle gas = optixGetGASTraversableHandle();
    const unsigned int           gasSbtIndex = optixGetSbtGASIndex();
    float4                       controlPoints[2];

    optixGetLinearCurveVertexData( gas, primitiveIndex, gasSbtIndex, 0.0f, controlPoints );

    LinearInterpolator interpolator;
    interpolator.initialize(controlPoints);

    float3               hitPoint = getHitPoint();
    // interpolators work in object space
    hitPoint            = optixTransformPointFromWorldToObjectSpace( hitPoint );
    const float3 normal = surfaceNormal( interpolator, optixGetCurveParameter(), hitPoint );
    return optixTransformNormalFromObjectToWorldSpace( normal );
}

// Compute surface normal of quadratic pimitive in world space.
static __forceinline__ __device__ float3 normalQuadratic( const int primitiveIndex )
{
    const OptixTraversableHandle gas         = optixGetGASTraversableHandle();
    const unsigned int           gasSbtIndex = optixGetSbtGASIndex();
    float4                       controlPoints[3];

    optixGetQuadraticBSplineVertexData( gas, primitiveIndex, gasSbtIndex, 0.0f, controlPoints );

    QuadraticInterpolator interpolator;
    interpolator.initializeFromBSpline(controlPoints);

    float3                  hitPoint = getHitPoint();
    // interpolators work in object space
    hitPoint            = optixTransformPointFromWorldToObjectSpace( hitPoint );
    const float3 normal = surfaceNormal( interpolator, optixGetCurveParameter(), hitPoint );
    return optixTransformNormalFromObjectToWorldSpace( normal );
}

// Compute surface normal of cubic b-spline pimitive in world space.
static __forceinline__ __device__ float3 normalCubic( const int primitiveIndex )
{
    const OptixTraversableHandle gas         = optixGetGASTraversableHandle();
    const unsigned int           gasSbtIndex = optixGetSbtGASIndex();
    float4                       controlPoints[4];

    optixGetCubicBSplineVertexData( gas, primitiveIndex, gasSbtIndex, 0.0f, controlPoints );

    CubicInterpolator interpolator;
    interpolator.initializeFromBSpline(controlPoints);

    float3              hitPoint = getHitPoint();
    // interpolators work in object space
    hitPoint            = optixTransformPointFromWorldToObjectSpace( hitPoint );
    const float3 normal = surfaceNormal( interpolator, optixGetCurveParameter(), hitPoint );
    return optixTransformNormalFromObjectToWorldSpace( normal );
}

// Compute surface normal of Catmull-Rom pimitive in world space.
static __forceinline__ __device__ float3 normalCatrom( const int primitiveIndex )
{
    const OptixTraversableHandle gas         = optixGetGASTraversableHandle();
    const unsigned int           gasSbtIndex = optixGetSbtGASIndex();
    float4                       controlPoints[4];

    optixGetCatmullRomVertexData( gas, primitiveIndex, gasSbtIndex, 0.0f, controlPoints );

    CubicInterpolator interpolator;
    interpolator.initializeFromCatrom(controlPoints);

    float3              hitPoint = getHitPoint();
    // interpolators work in object space
    hitPoint            = optixTransformPointFromWorldToObjectSpace( hitPoint );
    const float3 normal = surfaceNormal( interpolator, optixGetCurveParameter(), hitPoint );
    return optixTransformNormalFromObjectToWorldSpace( normal );
}

// Compute surface normal of Catmull-Rom pimitive in world space.
static __forceinline__ __device__ float3 normalBezier( const int primitiveIndex )
{
    const OptixTraversableHandle gas         = optixGetGASTraversableHandle();
    const unsigned int           gasSbtIndex = optixGetSbtGASIndex();
    float4                       controlPoints[4];

    optixGetCubicBezierVertexData( gas, primitiveIndex, gasSbtIndex, 0.0f, controlPoints );

    CubicInterpolator interpolator;
    interpolator.initializeFromBezier(controlPoints);

    float3              hitPoint = getHitPoint();
    // interpolators work in object space
    hitPoint            = optixTransformPointFromWorldToObjectSpace( hitPoint );
    const float3 normal = surfaceNormal( interpolator, optixGetCurveParameter(), hitPoint );
    return optixTransformNormalFromObjectToWorldSpace( normal );
}

// Compute normal
//
static __forceinline__ __device__ float3 computeCurveNormal( OptixPrimitiveType type, const int primitiveIndex )
{
    switch( type ) {
    case OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR:
        return normalLinear( primitiveIndex );
    case OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE:
        return normalQuadratic( primitiveIndex );
    case OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE:
        return normalCubic( primitiveIndex );
    case OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM:
        return normalCatrom( primitiveIndex );
    case OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER:
        return normalBezier( primitiveIndex );
        
    case OPTIX_PRIMITIVE_TYPE_FLAT_QUADRATIC_BSPLINE:
        {
            const unsigned int           prim_idx    = optixGetPrimitiveIndex();
            const OptixTraversableHandle gas         = optixGetGASTraversableHandle();
            const unsigned int           sbtGASIndex = optixGetSbtGASIndex();
            const float2                 uv          = optixGetRibbonParameters();
            auto normal = optixGetRibbonNormal( gas, prim_idx, sbtGASIndex, 0.f /*time*/, uv );
            return normalize(normal);
        }
    }
    return make_float3(0.0f);
}
