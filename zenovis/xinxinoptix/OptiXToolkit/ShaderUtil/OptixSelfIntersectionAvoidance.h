//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#pragma once

/// \file OptixSelfIntersectionAvoidance.h
/// Primary Optix interface of Self Intersection Avoidance library.
///
/// Example use:
///
/// extern "C" __global__ void __closesthit__ch()
/// {
///     ...
///
///     float3 objPos, objNorm;
///     float objOffset;
///
///     if( optixIsTriangleHit() )
///     {
///         // generate object space spawn point and offset
///         SelfIntersectionAvoidance::getSafeTriangleSpawnOffset( objPos, objNorm, objOffset );
///     }
///     else
///     {
///         // custom primitives
///         ...
///     }
///
///     float3 wldPos, wldNorm;
///     float wldOffset;
///
///     // convert object space spawn point and offset to world space
///     SelfIntersectionAvoidance::transformSafeSpawnOffset( wldPos, wldNorm, wldOffset, objPos, objNorm, objOffset );
///
///     float3 front, back;
///     // offset world space spawn point to generate self intersection safe front and back spawn points
///     SelfIntersectionAvoidance::offsetSpawnPoint( front, back, wldPos, wldNorm, wldOffset );
///
///     // flip normal to point towards incoming direction
///     if( dot( wldNorm, optixGetWorldRayDirection() ) > 0.f )
///     {
///         wldNorm = -wldNorm;
///         swap( front, back );
///     }
///     ...
///
///     // pick safe spawn point for secondary scatter ray
///     float3 scatterPos = ( dot( scatterDir, wldNorm ) > 0.f ) ? front : back;
///
///     // trace secondary ray
///     optixTrace( handle, scatterPos, scatterDir, 0.f, FLT_MAX, ... );
///
///     ...
///
/// }

#include <OptiXToolkit/ShaderUtil/Preprocessor.h>

#include "SelfIntersectionAvoidanceTypes.h"

namespace SelfIntersectionAvoidance {

// Generate spawn point and safe offset on triangle.
//
// Generates a spawn point on a triangle, a triangle normal and an offset along the normal.
// Offsetting the spawn point away from the triangle along the normal by the safe offset in either direction
// gives safe front and back spawn points. Rays starting at these safe spawn points and directed away from the triangle
// are guaranteed not to self-intersect with the source triangle.
//
// \param[out] outPosition      base spawn position on the triangle, without the safe offset applied.
// \param[out] outNormal        unit length triangle normal
// \param[out] outOffset        safe offset along the triangle normal to avoid self intersection
// \param[in]  v0               triangle vertex 0
// \param[in]  v1               triangle vertex 1
// \param[in]  v2               triangle vertex 2
// \param[in]  bary             barycentric coordinates of spawn point
OTK_INLINE __device__ void getSafeTriangleSpawnOffset( float3&       outPosition,
                                                       float3&       outNormal,
                                                       float&        outOffset,
                                                       const float3& v0,
                                                       const float3& v1,
                                                       const float3& v2,
                                                       const float2& bary );

// Generate object space spawn point and safe offset on triangle.
//
// Generates a spawn point on a triangle, a triangle normal and an offset along the normal.
// This function reconstructs the spawn point from the current optix hit context and is only available in CH, AH and IS programs.
// The current hit must be a triangle hit (\see optixIsTriangleHit) and the corresponding triangle GAS must support
// random vertex access (\see OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS).
//
// Offsetting the spawn point away from the triangle along the normal by the safe offset in either direction
// gives safe front and back spawn points. Rays starting at these safe spawn points and directed away from the triangle
// are guaranteed not to self-intersect with the source triangle.
//
// \param[out] outPosition      base spawn position on the triangle, without the safe offset applied.
// \param[out] outNormal        unit length triangle normal
// \param[out] outOffset        safe offset along the triangle normal to avoid self intersection
OTK_INLINE __device__ void getSafeTriangleSpawnOffset( float3& outPosition, float3& outNormal, float& outOffset );

// Transform object space spawn point and safe offset into world space by chain of transform traversables.
//
// Generates a world space spawn point, normal and offset along the normal from an input object space spawn point, normal and offset.
// The transformation is specified as a sequence of generic object to world transformations.
//
// The input offset is assumed to be sufficient for object space self intersection avoidance (\see getSafeTriangleSpawnOffset).
// The resulting world space offset accounts for both the object space offset and any error introduced by the object to world
// and traversal world to object transformations. Offsetting the world space spawn point away along the world space normal by
// the safe offset in either direction gives safe front and back spawn points (\see offsetSpawnPoint). Rays starting at these
// safe spawn points and directed away from the spawn point are guaranteed not to self-intersect with the source primitive.
//
// \paramp[out] outPosition     surface position in world space
// \paramp[out] outNormal       unit length spawn point normal in world space
// \paramp[out] outOffset       safe offset along normal in world space to avoid self intersection
// \paramp[in]  inPosition      object space spawn point
// \paramp[in]  inNormal        object space normal
// \paramp[in]  inOffset        object space offset
// \paramp[in]  time            motion time
// \paramp[in]  numTransforms   length of the chain of transforms
// \paramp[in]  transforms      transform traversables
OTK_INLINE __device__ void transformSafeSpawnOffset( float3&                       outPosition,
                                                     float3&                       outNormal,
                                                     float&                        outOffset,
                                                     const float3&                 inPosition,
                                                     const float3&                 inNormal,
                                                     const float                   inOffset,
                                                     const float                   time,
                                                     const unsigned int            numTransforms,
                                                     const OptixTraversableHandle* const __restrict transformHandles );

// Transform object space spawn point and safe offset into world space using the transform list of the current optix hit context.
//
// Generates a world space spawn point, normal and offset along the normal from an input object space spawn point, normal and offset.
// This function applies the transform associated with the current optix hit and is only available in CH, AH and IS programs.
// (\see optixGetTransformListSize and optixGetTransformListHandle).
//
// The input offset is assumed to be sufficient for object space self intersection avoidance (\see getSafeTriangleSpawnOffset).
// The resulting world space offset accounts for both the object space offset and any error introduced by the object to world
// and traversal world to object transformations. Offsetting the world space spawn point away along the world space normal by
// the safe offset in either direction gives safe front and back spawn points (\see offsetSpawnPoint). Rays starting at these
// safe spawn points and directed away from the spawn point are guaranteed not to self-intersect with the source primitive.
//
// \paramp[out] outPosition     surface position in world space
// \paramp[out] outNormal       unit length spawn point normal in world space
// \paramp[out] outOffset       safe offset along normal in world space to avoid self intersection
// \paramp[in]  inPosition      object space spawn point
// \paramp[in]  inNormal        object space normal
// \paramp[in]  inOffset        object space offset
OTK_INLINE __device__ void transformSafeSpawnOffset( float3&       outPosition,
                                                     float3&       outNormal,
                                                     float&        outOffset,
                                                     const float3& inPosition,
                                                     const float3& inNormal,
                                                     const float   inOffset );

// Offset spawn point to safe spawn points on either side of the surface.
//
// Takes a spawn point in object or world space (\see getSafeTriangleSpawnOffset or transformSafeSpawnOffset)
// and offset the spawn point along the normal to obtain a safe front and back spawn points.
// Rays starting at these safe spawn points and directed away from the spawn point
// are guaranteed not to self-intersect with the source primitive.
//
// \param[out] outFront     offset spawn point on the front of the surface, safe from self intersection
// \param[out] outBack      offset spawn point on the back of the surface, safe from self intersection
// \param[in]  inPosition   spawn point on the surface
// \param[in]  inNormal     surface normal
// \param[in]  inOffset     safe offset to avoid self intersection
OTK_INLINE __device__ void offsetSpawnPoint( float3& outFront, float3& outBack, const float3& inPosition, const float3& inNormal, const float inOffset );

// Offset spawn point to safe spawn point above the surface.
//
// Takes a spawn point in object or world space (\see getSafeTriangleSpawnOffset or transformSafeSpawnOffset)
// and offset the spawn point along the normal to obtain a safe spawn point.
// Rays starting at the safe spawn point and directed away from the surface along the normal
// are guaranteed not to self-intersect with the source primitive.
//
// \param[out] outPosition  offset spawn point on the front of the surface, safe from self intersection
// \param[in]  inPosition   spawn point on the surface
// \param[in]  inNormal     surface normal
// \param[in]  inOffset     safe offset to avoid self intersection
OTK_INLINE __device__ void offsetSpawnPoint( float3& outPosition, const float3& inPosition, const float3& inNormal, const float inOffset );

}  // namespace SelfIntersectionAvoidance

#ifdef __CUDACC__
#include "SelfIntersectionAvoidance/OptixSelfIntersectionAvoidanceImpl.h"
#endif
