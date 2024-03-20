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

/// \file CudaSelfIntersectionAvoidance.h
/// Primary cuda interface of Self Intersection Avoidance library.
///
/// Example use:
///
///     float3 v0 = ..., v1 = ..., v2 = ...;
///     float2 bary = ...;
///
///     Matrix3x4 o2w = ...;
///     Matrix3x4 w2o = ...;
///
///     float3 objPos, objNorm;
///     float objOffset;
///
///     // generate object space spawn point and offset
///     getSafeTriangleSpawnOffset( objPos, objNorm, objOffset, v0, v1, v2, bary );
///
///     float3 wldPos, wldNorm;
///     float wldOffset;
///
///     // convert object space spawn point and offset to world space
///     transformSafeSpawnOffset( wldPos, wldNorm, wldOffset, objPos, objNorm, objOffset, &o2w, &w2o  );
///
///     float3 front, back;
///     // offset world space spawn point to generate self intersection safe front and back spawn points
///     offsetSpawnPoint( front, back, wldPos, wldNorm, wldOffset );
///

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
// \param[out] outOffset        offset along the triangle normal to avoid self intersection
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

// Transform object space spawn point and safe offset into world space by chain of transforms.
//
// Generates a world space spawn point, normal and offset along the normal from
// an input object space spawn point, normal and offset. The transformation is specified as a single
// world to object and corresponding object to world matrix transformation. The input offset is assumed to be sufficient for object
// space self intersection avoidance (\see getSafeTriangleSpawnOffset). The resulting world space offset accounts for
// both the object space offset and any error introduced by the object to world and traversal world to object transformation.
// Offsetting the world space spawn point away along the world space normal by the safe offset in either direction
// gives safe front and back spawn points. Rays starting at these safe spawn points and directed away from the spawn point
// are guaranteed not to self-intersect with the source primitive.
//
// \paramp[out] outPosition     surface position in world space
// \paramp[out] outNormal       unit length spawn point normal in world space
// \paramp[out] outOffset       safe offset along normal in world space to avoid self intersection
// \paramp[in]  inPosition      object space spawn point
// \paramp[in]  inNormal        object space normal
// \paramp[in]  inOffset        object space offset
// \paramp[in]  o2w             pointer to constant object to world transform in global device memory
// \paramp[in]  w2o             pointer to constant world to object transform in global device memory
OTK_INLINE __device__ void transformSafeSpawnOffset( float3&            outPosition,
                                                     float3&            outNormal,
                                                     float&             outOffset,
                                                     const float3&      inPosition,
                                                     const float3&      inNormal,
                                                     const float        inOffset,
                                                     const Matrix3x4* const __restrict o2w,
                                                     const Matrix3x4* const __restrict w2o );

// Transform object space spawn point and safe offset into world space by chain of generic transforms.
//
// Generates a world space spawn point, normal and offset along the normal from
// an input object space spawn point, normal and offset. The transformation is specified as a sequence of
// generic object to world transformations. The input offset is assumed to be sufficient for object
// space self intersection avoidance (\see getSafeTriangleSpawnOffset). The resulting world space offset accounts for
// both the object space offset and any error introduced by the object to world and traversal world to object transformations.
// Offsetting the world space spawn point away along the world space normal by the safe offset in either direction
// gives safe front and back spawn points. Rays starting at these safe spawn points and directed away from the spawn point
// are guaranteed not to self-intersect with the source primitive.
//
// The T template encapsulates an optix traversable handle.
// The object must implement the following interface
//
// struct Transform
// {
//     // return the type of the transform
//     // if only a single type is used, specialize this function to unconditionally return the exact used type to assist dead-code elimination.
//     __device__ OptixTransformType getTransformTypeFromHandle() const;
//     
//     // assumes the type equals OPTIX_TRANSFORM_TYPE_MATRIX_MOTION_TRANSFORM
//     // \retval pointer to constant matrix motion traversable object in global device memory
//     __device__ const OptixMatrixMotionTransform* getMatrixMotionTransformFromHandle() const;
//     
//     // assumes the type equals OPTIX_TRANSFORM_TYPE_SRT_MOTION_TRANSFORM
//     // \retval pointer to constant srt motion traversable object in global device memory
//     __device__ const OptixSRTMotionTransform* getSRTMotionTransformFromHandle() const;
//     
//     // assumes the type equals OPTIX_TRANSFORM_TYPE_STATIC_TRANSFORM
//     // \retval pointer to constant static transform traversable object in global device memory
//     __device__ const OptixStaticTransform* getStaticTransformFromHandle() const;
//     
//     // assumes the type equals OPTIX_TRANSFORM_TYPE_INSTANCE 
//     __device__ Matrix3x4 getInstanceTransformFromHandle() const;
//     // assumes the type equals OPTIX_TRANSFORM_TYPE_INSTANCE 
//     __device__ Matrix3x4 getInstanceInverseTransformFromHandle() const;
// };
// 
// \paramp[out] outPosition     surface position in world space
// \paramp[out] outNormal       unit length spawn point normal in world space
// \paramp[out] outOffset       safe offset along normal in world space to avoid self intersection
// \paramp[in]  inPosition      object space spawn point
// \paramp[in]  inNormal        object space normal
// \paramp[in]  inOffset        object space offset
// \paramp[in]  time            motion time
// \paramp[in]  numTransforms   length of the chain of transforms
// \paramp[in]  transforms      transforms
template <typename T>
OTK_INLINE __device__ void transformSafeSpawnOffset( float3& outPosition,
                                                     float3& outNormal,
                                                     float& outOffset,
                                                     const float3& obj_p,
                                                     const float3& obj_n,
                                                     const float   obj_offset,
                                                     const float   time,
                                                     const unsigned int numTransforms,
                                                     const T* const __restrict transforms );

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
#include "SelfIntersectionAvoidance/CudaSelfIntersectionAvoidanceImpl.h"
#endif
