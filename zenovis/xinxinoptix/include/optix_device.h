/*
* SPDX-FileCopyrightText: Copyright (c) 2010 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
* SPDX-License-Identifier: LicenseRef-NvidiaProprietary
*
* NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
* property and proprietary rights in and to this material, related
* documentation and any modifications thereto. Any use, reproduction,
* disclosure or distribution of this material and related documentation
* without an express license agreement from NVIDIA CORPORATION or
* its affiliates is strictly prohibited.
*/
/// @file
/// @author NVIDIA Corporation
/// @brief  OptiX public API header
///
/// OptiX public API Reference - Device API declarations

#ifndef OPTIX_OPTIX_DEVICE_H
#define OPTIX_OPTIX_DEVICE_H

#if defined( __cplusplus ) && ( __cplusplus < 201103L ) && !defined( _WIN32 )
#error Device code for OptiX requires at least C++11. Consider adding "--std c++11" to the nvcc command-line.
#endif

#include "optix_types.h"

/// \defgroup optix_device_api Device API
/// \brief OptiX Device API

/** \addtogroup optix_device_api
@{
*/


/// Initiates a ray tracing query starting with the given traversable.
///
/// \param[in] handle
/// \param[in] rayOrigin
/// \param[in] rayDirection
/// \param[in] tmin
/// \param[in] tmax
/// \param[in] rayTime
/// \param[in] visibilityMask really only 8 bits
/// \param[in] rayFlags       really only 16 bits, combination of OptixRayFlags
/// \param[in] SBToffset      really only 4 bits
/// \param[in] SBTstride      really only 4 bits
/// \param[in] missSBTIndex   specifies the miss program invoked on a miss
/// \param[in,out] payload    up to 32 unsigned int values that hold the payload
///
/// Available in RG, CH, MS, CC
template <typename... Payload>
static __forceinline__ __device__ void optixTrace( OptixTraversableHandle handle,
                                                   float3                 rayOrigin,
                                                   float3                 rayDirection,
                                                   float                  tmin,
                                                   float                  tmax,
                                                   float                  rayTime,
                                                   OptixVisibilityMask    visibilityMask,
                                                   unsigned int           rayFlags,
                                                   unsigned int           SBToffset,
                                                   unsigned int           SBTstride,
                                                   unsigned int           missSBTIndex,
                                                   Payload&...            payload );

/// Similar to optixTrace, but does not invoke closesthit or miss. Instead, it overwrites the
/// current outgoing hit object with the results of traversing the ray. The outgoing hit object may
/// be invoked at some later point with optixInvoke. The outgoing hit object can also be queried
/// through various functions such as optixHitObjectIsHit or optixHitObjectGetAttribute_0.
///
/// \param[in] handle
/// \param[in] rayOrigin
/// \param[in] rayDirection
/// \param[in] tmin
/// \param[in] tmax
/// \param[in] rayTime
/// \param[in] visibilityMask really only 8 bits
/// \param[in] rayFlags       really only 16 bits, combination of OptixRayFlags
/// \param[in] SBToffset      really only 4 bits
/// \param[in] SBTstride      really only 4 bits
/// \param[in] missSBTIndex   specifies the miss program invoked on a miss
/// \param[in,out] payload    up to 32 unsigned int values that hold the payload
///
/// Available in RG, CH, MS, CC, DC
template <typename... Payload>
static __forceinline__ __device__ void optixTraverse( OptixTraversableHandle handle,
                                                      float3                 rayOrigin,
                                                      float3                 rayDirection,
                                                      float                  tmin,
                                                      float                  tmax,
                                                      float                  rayTime,
                                                      OptixVisibilityMask    visibilityMask,
                                                      unsigned int           rayFlags,
                                                      unsigned int           SBToffset,
                                                      unsigned int           SBTstride,
                                                      unsigned int           missSBTIndex,
                                                      Payload&... payload );

/// Initiates a ray tracing query starting with the given traversable.
///
/// \param[in] type
/// \param[in] handle
/// \param[in] rayOrigin
/// \param[in] rayDirection
/// \param[in] tmin
/// \param[in] tmax
/// \param[in] rayTime
/// \param[in] visibilityMask really only 8 bits
/// \param[in] rayFlags       really only 16 bits, combination of OptixRayFlags
/// \param[in] SBToffset      really only 4 bits
/// \param[in] SBTstride      really only 4 bits
/// \param[in] missSBTIndex   specifies the miss program invoked on a miss
/// \param[in,out] payload    up to 32 unsigned int values that hold the payload
///
/// Available in RG, CH, MS, CC
template <typename... Payload>
static __forceinline__ __device__ void optixTrace( OptixPayloadTypeID     type,
                                                   OptixTraversableHandle handle,
                                                   float3                 rayOrigin,
                                                   float3                 rayDirection,
                                                   float                  tmin,
                                                   float                  tmax,
                                                   float                  rayTime,
                                                   OptixVisibilityMask    visibilityMask,
                                                   unsigned int           rayFlags,
                                                   unsigned int           SBToffset,
                                                   unsigned int           SBTstride,
                                                   unsigned int           missSBTIndex,
                                                   Payload&...            payload );

/// Similar to optixTrace, but does not invoke closesthit or miss. Instead, it overwrites the
/// current outgoing hit object with the results of traversing the ray. The outgoing hit object may
/// be invoked at some later point with optixInvoke. The outgoing hit object can also be queried
/// through various functions such as optixHitObjectIsHit or optixHitObjectGetAttribute_0.
///
/// \param[in] type
/// \param[in] handle
/// \param[in] rayOrigin
/// \param[in] rayDirection
/// \param[in] tmin
/// \param[in] tmax
/// \param[in] rayTime
/// \param[in] visibilityMask really only 8 bits
/// \param[in] rayFlags       really only 16 bits, combination of OptixRayFlags
/// \param[in] SBToffset      really only 4 bits
/// \param[in] SBTstride      really only 4 bits
/// \param[in] missSBTIndex   specifies the miss program invoked on a miss
/// \param[in,out] payload    up to 32 unsigned int values that hold the payload
///
/// Available in RG, CH, MS, CC, DC
template <typename... Payload>
static __forceinline__ __device__ void optixTraverse( OptixPayloadTypeID     type,
                                                      OptixTraversableHandle handle,
                                                      float3                 rayOrigin,
                                                      float3                 rayDirection,
                                                      float                  tmin,
                                                      float                  tmax,
                                                      float                  rayTime,
                                                      OptixVisibilityMask    visibilityMask,
                                                      unsigned int           rayFlags,
                                                      unsigned int           SBToffset,
                                                      unsigned int           SBTstride,
                                                      unsigned int           missSBTIndex,
                                                      Payload&... payload );

/// Reorder the current thread using the current outgoing hit object and the coherence hint bits
/// provided.  Note that the coherence hint will take away some of the bits used in the hit object
/// for sorting, so care should be made to reduce the number of hint bits as much as possible. Nop
/// hit objects can use more coherence hint bits. Bits are taken from the lowest significant bit
/// range. The maximum value of numCoherenceHintBitsFromLSB is implementation defined and can vary.
///
/// \param[in] coherenceHint
/// \param[in] numCoherenceHintBitsFromLSB
///
/// Available in RG
static __forceinline__ __device__ void optixReorder( unsigned int coherenceHint, unsigned int numCoherenceHintBitsFromLSB );

/// Reorder the current thread using the hit object only, ie without further coherence hints.
///
/// Available in RG
static __forceinline__ __device__ void optixReorder();

/// Invokes closesthit, miss or nop based on the current outgoing hit object. After execution the
/// current outgoing hit object will be set to nop. An implied nop hit object is always assumed to
/// exist even if there are no calls to optixTraverse, optixMakeHitObject, optixMakeMissHitObject,
/// or optixMakeNopHitObject.
///
/// \param[in,out] payload       up to 32 unsigned int values that hold the payload
///
/// Available in RG, CH, MS, CC
template <typename... Payload>
static __forceinline__ __device__ void optixInvoke( Payload&... payload );

/// Invokes closesthit, miss or nop based on the current outgoing hit object. After execution the
/// current outgoing hit object will be set to nop. An implied nop hit object is always assumed to
/// exist even if there are no calls to optixTraverse, optixMakeHitObject, optixMakeMissHitObject,
/// or optixMakeNopHitObject.
///
/// \param[in] type
/// \param[in,out] payload       up to 32 unsigned int values that hold the payload
///
/// Available in RG, CH, MS, CC
template <typename... Payload>
static __forceinline__ __device__ void optixInvoke( OptixPayloadTypeID type, Payload&... payload );

/// Constructs an outgoing hit object from the hit information provided. This hit object will now
/// become the current outgoing hit object and will overwrite the current outgoing hit object.
///
/// \param[in] handle
/// \param[in] rayOrigin
/// \param[in] rayDirection
/// \param[in] tmin
/// \param[in] tmax
/// \param[in] rayTime
/// \param[in] SBToffset      really only 4 bits
/// \param[in] SBTstride      really only 4 bits
/// \param[in] instIdx
/// \param[in] sbtGASIdx
/// \param[in] primIdx
/// \param[in] hitKind
/// \param[in] regAttributes  up to 8 attribute registers
///
/// Available in RG, CH, MS, CC
template <typename... RegAttributes>
static __forceinline__ __device__ void optixMakeHitObject( OptixTraversableHandle handle,
                                                           float3                 rayOrigin,
                                                           float3                 rayDirection,
                                                           float                  tmin,
                                                           float                  tmax,
                                                           float                  rayTime,
                                                           unsigned int           SBToffset,
                                                           unsigned int           SBTstride,
                                                           unsigned int           instIdx,
                                                           unsigned int           sbtGASIdx,
                                                           unsigned int           primIdx,
                                                           unsigned int           hitKind,
                                                           RegAttributes... regAttributes );

/// Constructs an outgoing hit object from the hit information provided. This hit object will now
/// become the current outgoing hit object and will overwrite the current outgoing hit object. This
/// method includes the ability to specify arbitrary numbers of OptixTraversableHandle pointers for
/// scenes with 0 to OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRAVERSABLE_GRAPH_DEPTH levels of transforms.
///
/// \param[in] handle
/// \param[in] rayOrigin
/// \param[in] rayDirection
/// \param[in] tmin
/// \param[in] tmax
/// \param[in] rayTime
/// \param[in] SBToffset      really only 4 bits
/// \param[in] SBTstride      really only 4 bits
/// \param[in] instIdx
/// \param[in] transforms
/// \param[in] numTransforms
/// \param[in] sbtGASIdx
/// \param[in] primIdx
/// \param[in] hitKind
/// \param[in] regAttributes  up to 8 attribute registers
///
/// Available in RG, CH, MS, CC
template <typename... RegAttributes>
static __forceinline__ __device__ void optixMakeHitObject( OptixTraversableHandle        handle,
                                                           float3                        rayOrigin,
                                                           float3                        rayDirection,
                                                           float                         tmin,
                                                           float                         tmax,
                                                           float                         rayTime,
                                                           unsigned int                  SBToffset,
                                                           unsigned int                  SBTstride,
                                                           unsigned int                  instIdx,
                                                           const OptixTraversableHandle* transforms,
                                                           unsigned int                  numTransforms,
                                                           unsigned int                  sbtGASIdx,
                                                           unsigned int                  primIdx,
                                                           unsigned int                  hitKind,
                                                           RegAttributes... regAttributes );

/// Constructs an outgoing hit object from the hit information provided. The SBT record index is
/// explicitly specified. This hit object will now become the current outgoing hit object and will
/// overwrite the current outgoing hit object.
///
/// \param[in] handle
/// \param[in] rayOrigin
/// \param[in] rayDirection
/// \param[in] tmin
/// \param[in] tmax
/// \param[in] rayTime
/// \param[in] sbtRecordIndex 32 bits
/// \param[in] instIdx
/// \param[in] transforms
/// \param[in] numTransforms
/// \param[in] sbtGASIdx
/// \param[in] primIdx
/// \param[in] hitKind
/// \param[in] regAttributes  up to 8 attribute registers
///
/// Available in RG, CH, MS, CC
template <typename... RegAttributes>
static __forceinline__ __device__ void optixMakeHitObjectWithRecord( OptixTraversableHandle        handle,
                                                                     float3                        rayOrigin,
                                                                     float3                        rayDirection,
                                                                     float                         tmin,
                                                                     float                         tmax,
                                                                     float                         rayTime,
                                                                     unsigned int                  sbtRecordIndex,
                                                                     unsigned int                  instIdx,
                                                                     const OptixTraversableHandle* transforms,
                                                                     unsigned int                  numTransforms,
                                                                     unsigned int                  sbtGASIdx,
                                                                     unsigned int                  primIdx,
                                                                     unsigned int                  hitKind,
                                                                     RegAttributes... regAttributes );

/// Constructs an outgoing hit object from the miss information provided. The SBT record index is
/// explicitly specified as an argument. This hit object will now become the current outgoing hit
/// object and will overwrite the current outgoing hit object.
///
/// \param[in] missSBTIndex   specifies the miss program invoked on a miss
/// \param[in] rayOrigin
/// \param[in] rayDirection
/// \param[in] tmin
/// \param[in] tmax
/// \param[in] rayTime
///
/// Available in RG, CH, MS, CC
static __forceinline__ __device__ void optixMakeMissHitObject( unsigned int missSBTIndex,
                                                               float3       rayOrigin,
                                                               float3       rayDirection,
                                                               float        tmin,
                                                               float        tmax,
                                                               float        rayTime );

/// Constructs an outgoing hit object that when invoked does nothing (neither the miss nor the
/// closest hit shader will be invoked). This hit object will now become the current outgoing hit
/// object and will overwrite the current outgoing hit object. Accessors such as
/// optixHitObjectGetInstanceId will return 0 or 0 filled structs. Only optixHitObjectGetIsNop()
/// will return a non-zero result.
///
/// Available in RG, CH, MS, CC
static __forceinline__ __device__ void optixMakeNopHitObject();

/// Returns true if the current outgoing hit object contains a hit.
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ bool optixHitObjectIsHit();

/// Returns true if the current outgoing hit object contains a miss.
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ bool optixHitObjectIsMiss();

/// Returns true if the current outgoing hit object contains neither a hit nor miss. If executed
/// with optixInvoke, no operation will result. An implied nop hit object is always assumed to exist
/// even if there are no calls such as optixTraverse to explicitly create one.
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ bool optixHitObjectIsNop();

/// Returns the SBT record index associated with the hit or miss program for the current outgoing
/// hit object.
///
/// Returns 0 for nop hit objects.
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ unsigned int optixHitObjectGetSbtRecordIndex();

/// Writes the 32-bit payload at the given slot index. There are up to 32 attributes available. The
/// number of attributes is configured with OptixPipelineCompileOptions::numPayloadValues or with
/// OptixPayloadType parameters set in OptixModuleCompileOptions.
///
/// Available in IS, AH, CH, MS
static __forceinline__ __device__ void optixSetPayload_0( unsigned int p );
static __forceinline__ __device__ void optixSetPayload_1( unsigned int p );
static __forceinline__ __device__ void optixSetPayload_2( unsigned int p );
static __forceinline__ __device__ void optixSetPayload_3( unsigned int p );
static __forceinline__ __device__ void optixSetPayload_4( unsigned int p );
static __forceinline__ __device__ void optixSetPayload_5( unsigned int p );
static __forceinline__ __device__ void optixSetPayload_6( unsigned int p );
static __forceinline__ __device__ void optixSetPayload_7( unsigned int p );
static __forceinline__ __device__ void optixSetPayload_8( unsigned int p );
static __forceinline__ __device__ void optixSetPayload_9( unsigned int p );
static __forceinline__ __device__ void optixSetPayload_10( unsigned int p );
static __forceinline__ __device__ void optixSetPayload_11( unsigned int p );
static __forceinline__ __device__ void optixSetPayload_12( unsigned int p );
static __forceinline__ __device__ void optixSetPayload_13( unsigned int p );
static __forceinline__ __device__ void optixSetPayload_14( unsigned int p );
static __forceinline__ __device__ void optixSetPayload_15( unsigned int p );
static __forceinline__ __device__ void optixSetPayload_16( unsigned int p );
static __forceinline__ __device__ void optixSetPayload_17( unsigned int p );
static __forceinline__ __device__ void optixSetPayload_18( unsigned int p );
static __forceinline__ __device__ void optixSetPayload_19( unsigned int p );
static __forceinline__ __device__ void optixSetPayload_20( unsigned int p );
static __forceinline__ __device__ void optixSetPayload_21( unsigned int p );
static __forceinline__ __device__ void optixSetPayload_22( unsigned int p );
static __forceinline__ __device__ void optixSetPayload_23( unsigned int p );
static __forceinline__ __device__ void optixSetPayload_24( unsigned int p );
static __forceinline__ __device__ void optixSetPayload_25( unsigned int p );
static __forceinline__ __device__ void optixSetPayload_26( unsigned int p );
static __forceinline__ __device__ void optixSetPayload_27( unsigned int p );
static __forceinline__ __device__ void optixSetPayload_28( unsigned int p );
static __forceinline__ __device__ void optixSetPayload_29( unsigned int p );
static __forceinline__ __device__ void optixSetPayload_30( unsigned int p );
static __forceinline__ __device__ void optixSetPayload_31( unsigned int p );

/// Returns the 32-bit payload at the given slot index. There are up to 32 attributes available. The
/// number of attributes is configured with OptixPipelineCompileOptions::numPayloadValues or with
/// OptixPayloadType parameters set in OptixModuleCompileOptions.
///
/// Available in IS, AH, CH, MS
static __forceinline__ __device__ unsigned int optixGetPayload_0();
static __forceinline__ __device__ unsigned int optixGetPayload_1();
static __forceinline__ __device__ unsigned int optixGetPayload_2();
static __forceinline__ __device__ unsigned int optixGetPayload_3();
static __forceinline__ __device__ unsigned int optixGetPayload_4();
static __forceinline__ __device__ unsigned int optixGetPayload_5();
static __forceinline__ __device__ unsigned int optixGetPayload_6();
static __forceinline__ __device__ unsigned int optixGetPayload_7();
static __forceinline__ __device__ unsigned int optixGetPayload_8();
static __forceinline__ __device__ unsigned int optixGetPayload_9();
static __forceinline__ __device__ unsigned int optixGetPayload_10();
static __forceinline__ __device__ unsigned int optixGetPayload_11();
static __forceinline__ __device__ unsigned int optixGetPayload_12();
static __forceinline__ __device__ unsigned int optixGetPayload_13();
static __forceinline__ __device__ unsigned int optixGetPayload_14();
static __forceinline__ __device__ unsigned int optixGetPayload_15();
static __forceinline__ __device__ unsigned int optixGetPayload_16();
static __forceinline__ __device__ unsigned int optixGetPayload_17();
static __forceinline__ __device__ unsigned int optixGetPayload_18();
static __forceinline__ __device__ unsigned int optixGetPayload_19();
static __forceinline__ __device__ unsigned int optixGetPayload_20();
static __forceinline__ __device__ unsigned int optixGetPayload_21();
static __forceinline__ __device__ unsigned int optixGetPayload_22();
static __forceinline__ __device__ unsigned int optixGetPayload_23();
static __forceinline__ __device__ unsigned int optixGetPayload_24();
static __forceinline__ __device__ unsigned int optixGetPayload_25();
static __forceinline__ __device__ unsigned int optixGetPayload_26();
static __forceinline__ __device__ unsigned int optixGetPayload_27();
static __forceinline__ __device__ unsigned int optixGetPayload_28();
static __forceinline__ __device__ unsigned int optixGetPayload_29();
static __forceinline__ __device__ unsigned int optixGetPayload_30();
static __forceinline__ __device__ unsigned int optixGetPayload_31();

/// Specify the supported payload types for a program.
///
/// The supported types are specified as a bitwise combination of payload types. (See
/// OptixPayloadTypeID) May only be called once per program.
///
/// Must be called at the top of the program.
///
/// Available in IS, AH, CH, MS
static __forceinline__ __device__ void optixSetPayloadTypes( unsigned int typeMask );

/// Returns an undefined value.
///
/// Available anywhere
static __forceinline__ __device__ unsigned int optixUndefinedValue();

/// Returns the rayOrigin passed into optixTrace.
///
/// May be more expensive to call in IS and AH than their object space counterparts, so effort
/// should be made to use the object space ray in those programs.
///
/// Available in IS, AH, CH, MS
static __forceinline__ __device__ float3 optixGetWorldRayOrigin();

/// Returns the rayOrigin passed into optixTraverse, optixMakeHitObject,
/// optixMakeHitObjectWithRecord, or optixMakeMissHitObject.
///
/// Returns [0, 0, 0] for nop hit objects.
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ float3 optixHitObjectGetWorldRayOrigin();

/// Returns the rayDirection passed into optixTrace.
///
/// May be more expensive to call in IS and AH than their object space counterparts, so effort
/// should be made to use the object space ray in those programs.
///
/// Available in IS, AH, CH, MS
static __forceinline__ __device__ float3 optixGetWorldRayDirection();

/// Returns the rayDirection passed into optixTraverse, optixMakeHitObject,
/// optixMakeHitObjectWithRecord, or optixMakeMissHitObject.
///
/// Returns [0, 0, 0] for nop hit objects.
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ float3 optixHitObjectGetWorldRayDirection();

/// Returns the current object space ray origin based on the current transform stack.
///
/// Available in IS and AH
static __forceinline__ __device__ float3 optixGetObjectRayOrigin();

/// Returns the current object space ray direction based on the current transform stack.
///
/// Available in IS and AH
static __forceinline__ __device__ float3 optixGetObjectRayDirection();

/// Returns the tmin passed into optixTrace.
///
/// Available in IS, AH, CH, MS
static __forceinline__ __device__ float optixGetRayTmin();

/// Returns the tmin passed into optixTraverse, optixMakeHitObject,
/// optixMakeHitObjectWithRecord, or optixMakeMissHitObject.
///
/// Returns 0.0f for nop hit objects.
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ float optixHitObjectGetRayTmin();

/// In IS and CH returns the current smallest reported hitT or the tmax passed into optixTrace if no
/// hit has been reported
///
/// In AH returns the hitT value as passed in to optixReportIntersection
///
/// In MS returns the tmax passed into optixTrace
///
/// Available in IS, AH, CH, MS
static __forceinline__ __device__ float optixGetRayTmax();

/// If the hit object is a hit, returns the smallest reported hitT
///
/// If the hit object is a miss, returns the tmax passed into optixTraverse or
/// optixMakeMissHitObject.
///
/// Returns 0 for nop hit objects.
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ float optixHitObjectGetRayTmax();

/// Returns the rayTime passed into optixTrace.
///
/// Returns 0 if motion is disabled.
///
/// Available in IS, AH, CH, MS
static __forceinline__ __device__ float optixGetRayTime();

/// Returns the rayTime passed into optixTraverse, optixMakeHitObject,
/// optixMakeHitObjectWithRecord, or optixMakeMissHitObject.
///
/// Returns 0 for nop hit objects or when motion is disabled.
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ float optixHitObjectGetRayTime();

/// Returns the rayFlags passed into optixTrace
///
/// Available in IS, AH, CH, MS
static __forceinline__ __device__ unsigned int optixGetRayFlags();

/// Returns the visibilityMask passed into optixTrace
///
/// Available in IS, AH, CH, MS
static __forceinline__ __device__ unsigned int optixGetRayVisibilityMask();

/// Return the traversable handle of a given instance in an Instance Acceleration Structure (IAS)
///
/// To obtain instance traversables by index, the IAS must be built using the flag
/// OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS.
///
/// Available in all OptiX program types
static __forceinline__ __device__ OptixTraversableHandle optixGetInstanceTraversableFromIAS( OptixTraversableHandle ias, unsigned int instIdx );

/// Return the object space triangle vertex positions of a given triangle in a Geometry Acceleration
/// Structure (GAS) at a given motion time.
///
/// To access vertex data, the GAS must be built using the flag
/// OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS.
///
/// If motion is disabled via OptixPipelineCompileOptions::usesMotionBlur, or the GAS does not
/// contain motion, the time parameter is ignored.
///
/// Available in all OptiX program types
static __forceinline__ __device__ void optixGetTriangleVertexData( OptixTraversableHandle gas, unsigned int primIdx, unsigned int sbtGASIndex, float time, float3 data[3]);

/// Return the object space micro triangle vertex positions of the current hit.  The current hit
/// must be a displacement micromap triangle hit.
///
/// Available in all OptiX program types
static __forceinline__ __device__ void optixGetMicroTriangleVertexData( float3 data[3] );

/// Returns the barycentrics of the vertices of the currently intersected micro triangle with
/// respect to the base triangle.
///
/// Available in all OptiX program types
static __forceinline__ __device__ void optixGetMicroTriangleBarycentricsData( float2 data[3] );

/// Return the object space curve control vertex data of a linear curve in a Geometry Acceleration
/// Structure (GAS) at a given motion time.
///
/// To access vertex data, the GAS must be built using the flag
/// OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS.
///
/// data[i] = {x,y,z,w} with {x,y,z} the position and w the radius of control vertex i.
///
/// If motion is disabled via OptixPipelineCompileOptions::usesMotionBlur, or the GAS does not
/// contain motion, the time parameter is ignored.
///
/// Available in all OptiX program types
static __forceinline__ __device__ void optixGetLinearCurveVertexData( OptixTraversableHandle gas, unsigned int primIdx, unsigned int sbtGASIndex, float time, float4 data[2] );

/// Return the object space curve control vertex data of a quadratic BSpline curve in a Geometry
/// Acceleration Structure (GAS) at a given motion time.
///
/// To access vertex data, the GAS must be built using the flag
/// OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS.
///
/// data[i] = {x,y,z,w} with {x,y,z} the position and w the radius of control vertex i.
///
/// If motion is disabled via OptixPipelineCompileOptions::usesMotionBlur, or the GAS does not
/// contain motion, the time parameter is ignored.
///
/// Available in all OptiX program types
static __forceinline__ __device__ void optixGetQuadraticBSplineVertexData( OptixTraversableHandle gas, unsigned int primIdx, unsigned int sbtGASIndex, float time, float4 data[3] );

/// Return the object space curve control vertex data of a cubic BSpline curve in a Geometry
/// Acceleration Structure (GAS) at a given motion time.
///
/// To access vertex data, the GAS must be built using the flag
/// OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS.
///
/// data[i] = {x,y,z,w} with {x,y,z} the position and w the radius of control vertex i.
///
/// If motion is disabled via OptixPipelineCompileOptions::usesMotionBlur, or the GAS does not
/// contain motion, the time parameter is ignored.
///
/// Available in all OptiX program types
static __forceinline__ __device__ void optixGetCubicBSplineVertexData( OptixTraversableHandle gas, unsigned int primIdx, unsigned int sbtGASIndex, float time, float4 data[4] );

/// Return the object space curve control vertex data of a CatmullRom spline curve in a Geometry
/// Acceleration Structure (GAS) at a given motion time.
///
/// To access vertex data, the GAS must be built using the flag
/// OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS.
///
/// data[i] = {x,y,z,w} with {x,y,z} the position and w the radius of control vertex i.
///
/// If motion is disabled via OptixPipelineCompileOptions::usesMotionBlur, or the GAS does not
/// contain motion, the time parameter is ignored.
///
/// Available in all OptiX program types
static __forceinline__ __device__ void optixGetCatmullRomVertexData( OptixTraversableHandle gas, unsigned int primIdx, unsigned int sbtGASIndex, float time, float4 data[4] );

/// Return the object space curve control vertex data of a cubic Bezier curve in a Geometry
/// Acceleration Structure (GAS) at a given motion time.
///
/// To access vertex data, the GAS must be built using the flag
/// OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS.
///
/// data[i] = {x,y,z,w} with {x,y,z} the position and w the radius of control vertex i.
///
/// If motion is disabled via OptixPipelineCompileOptions::usesMotionBlur, or the GAS does not
/// contain motion, the time parameter is ignored.
///
/// Available in all OptiX program types
static __forceinline__ __device__ void optixGetCubicBezierVertexData( OptixTraversableHandle gas, unsigned int primIdx, unsigned int sbtGASIndex, float time, float4 data[4] );

/// Return the object space curve control vertex data of a ribbon (flat quadratic BSpline) in a
/// Geometry Acceleration Structure (GAS) at a given motion time.
///
/// To access vertex data, the GAS must be built using the flag
/// OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS.
///
/// data[i] = {x,y,z,w} with {x,y,z} the position and w the radius of control vertex i.
///
/// If motion is disabled via OptixPipelineCompileOptions::usesMotionBlur, or the GAS does not
/// contain motion, the time parameter is ignored.
///
/// Available in all OptiX program types
static __forceinline__ __device__ void optixGetRibbonVertexData( OptixTraversableHandle gas, unsigned int primIdx, unsigned int sbtGASIndex, float time, float4 data[3] );

/// Return ribbon normal at intersection reported by optixReportIntersection.
///
/// Available in all OptiX program types
static __forceinline__ __device__ float3 optixGetRibbonNormal( OptixTraversableHandle gas, unsigned int primIdx, unsigned int sbtGASIndex, float time, float2 ribbonParameters );

/// Return the object space sphere data, center point and radius, in a Geometry Acceleration
/// Structure (GAS) at a given motion time.
///
/// To access sphere data, the GAS must be built using the flag
/// OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS.
///
/// data[0] = {x,y,z,w} with {x,y,z} the position of the sphere center and w the radius.
///
/// If motion is disabled via OptixPipelineCompileOptions::usesMotionBlur, or the GAS does not
/// contain motion, the time parameter is ignored.
///
/// Available in all OptiX program types
static __forceinline__ __device__ void optixGetSphereData( OptixTraversableHandle gas, unsigned int primIdx, unsigned int sbtGASIndex, float time, float4 data[1] );

/// Returns the traversable handle for the Geometry Acceleration Structure (GAS) containing the
/// current hit.
///
/// Available in IS, AH, CH
static __forceinline__ __device__ OptixTraversableHandle optixGetGASTraversableHandle();

/// Returns the motion begin time of a GAS (see OptixMotionOptions)
///
/// Available in all OptiX program types
static __forceinline__ __device__ float optixGetGASMotionTimeBegin( OptixTraversableHandle gas );

/// Returns the motion end time of a GAS (see OptixMotionOptions)
///
/// Available in all OptiX program types
static __forceinline__ __device__ float optixGetGASMotionTimeEnd( OptixTraversableHandle gas );

/// Returns the number of motion steps of a GAS (see OptixMotionOptions)
///
/// Available in all OptiX program types
static __forceinline__ __device__ unsigned int optixGetGASMotionStepCount( OptixTraversableHandle gas );

/// Returns the world-to-object transformation matrix resulting from the current active
/// transformation list.
///
/// The cost of this function may be proportional to the size of the transformation list.
///
/// Available in IS, AH, CH
static __forceinline__ __device__ void optixGetWorldToObjectTransformMatrix( float m[12] );

/// Returns the object-to-world transformation matrix resulting from the current active
/// transformation list.
///
/// The cost of this function may be proportional to the size of the transformation list.
///
/// Available in IS, AH, CH
static __forceinline__ __device__ void optixGetObjectToWorldTransformMatrix( float m[12] );

/// Transforms the point using world-to-object transformation matrix resulting from the current
/// active transformation list.
///
/// The cost of this function may be proportional to the size of the transformation list.
///
/// Available in IS, AH, CH
static __forceinline__ __device__ float3 optixTransformPointFromWorldToObjectSpace( float3 point );

/// Transforms the vector using world-to-object transformation matrix resulting from the current
/// active transformation list.
///
/// The cost of this function may be proportional to the size of the transformation list.
///
/// Available in IS, AH, CH
static __forceinline__ __device__ float3 optixTransformVectorFromWorldToObjectSpace( float3 vec );

/// Transforms the normal using world-to-object transformation matrix resulting from the current
/// active transformation list.
///
/// The cost of this function may be proportional to the size of the transformation list.
///
/// Available in IS, AH, CH
static __forceinline__ __device__ float3 optixTransformNormalFromWorldToObjectSpace( float3 normal );

/// Transforms the point using object-to-world transformation matrix resulting from the current
/// active transformation list.
///
/// The cost of this function may be proportional to the size of the transformation list.
///
/// Available in IS, AH, CH
static __forceinline__ __device__ float3 optixTransformPointFromObjectToWorldSpace( float3 point );

/// Transforms the vector using object-to-world transformation matrix resulting from the current
/// active transformation list.
///
/// The cost of this function may be proportional to the size of the transformation list.
///
/// Available in IS, AH, CH
static __forceinline__ __device__ float3 optixTransformVectorFromObjectToWorldSpace( float3 vec );

/// Transforms the normal using object-to-world transformation matrix resulting from the current
/// active transformation list.
///
/// The cost of this function may be proportional to the size of the transformation list.
///
/// Available in IS, AH, CH
static __forceinline__ __device__ float3 optixTransformNormalFromObjectToWorldSpace( float3 normal );

/// Returns the number of transforms on the current transform list.
///
/// Available in IS, AH, CH
static __forceinline__ __device__ unsigned int optixGetTransformListSize();

/// Returns the number of transforms associated with the current outgoing hit object's transform
/// list.
///
/// Returns zero when there is no hit (miss and nop).
///
/// See #optixGetTransformListSize()
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ unsigned int optixHitObjectGetTransformListSize();

/// Returns the traversable handle for a transform in the current transform list.
///
/// Available in IS, AH, CH
static __forceinline__ __device__ OptixTraversableHandle optixGetTransformListHandle( unsigned int index );

/// Returns the traversable handle for a transform in the current transform list associated with the
/// outgoing hit object.
///
/// Results are undefined if the hit object is a miss.
///
/// See #optixGetTransformListHandle()
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ OptixTraversableHandle optixHitObjectGetTransformListHandle( unsigned int index );

/// Returns the transform type of a traversable handle from a transform list.
///
/// Available in all OptiX program types
static __forceinline__ __device__ OptixTransformType optixGetTransformTypeFromHandle( OptixTraversableHandle handle );

/// Returns a pointer to a OptixStaticTransform from its traversable handle.
///
/// Returns 0 if the traversable is not of type OPTIX_TRANSFORM_TYPE_STATIC_TRANSFORM.
///
/// Available in all OptiX program types
static __forceinline__ __device__ const OptixStaticTransform* optixGetStaticTransformFromHandle( OptixTraversableHandle handle );

/// Returns a pointer to a OptixSRTMotionTransform from its traversable handle.
///
/// Returns 0 if the traversable is not of type OPTIX_TRANSFORM_TYPE_SRT_MOTION_TRANSFORM.
///
/// Available in all OptiX program types
static __forceinline__ __device__ const OptixSRTMotionTransform* optixGetSRTMotionTransformFromHandle( OptixTraversableHandle handle );

/// Returns a pointer to a OptixMatrixMotionTransform from its traversable handle.
///
/// Returns 0 if the traversable is not of type OPTIX_TRANSFORM_TYPE_MATRIX_MOTION_TRANSFORM.
///
/// Available in all OptiX program types
static __forceinline__ __device__ const OptixMatrixMotionTransform* optixGetMatrixMotionTransformFromHandle( OptixTraversableHandle handle );

/// Returns instanceId from an OptixInstance traversable.
///
/// Returns 0 if the traversable handle does not reference an OptixInstance.
///
/// Available in all OptiX program types
static __forceinline__ __device__ unsigned int optixGetInstanceIdFromHandle( OptixTraversableHandle handle );

/// Returns child traversable handle from an OptixInstance traversable.
///
/// Returns 0 if the traversable handle does not reference an OptixInstance.
///
/// Available in all OptiX program types
static __forceinline__ __device__ OptixTraversableHandle optixGetInstanceChildFromHandle( OptixTraversableHandle handle );

/// Returns object-to-world transform from an OptixInstance traversable.
///
/// Returns 0 if the traversable handle does not reference an OptixInstance.
///
/// Available in all OptiX program types
static __forceinline__ __device__ const float4* optixGetInstanceTransformFromHandle( OptixTraversableHandle handle );

/// Returns world-to-object transform from an OptixInstance traversable.
///
/// Returns 0 if the traversable handle does not reference an OptixInstance.
///
/// Available in all OptiX program types
static __forceinline__ __device__ const float4* optixGetInstanceInverseTransformFromHandle( OptixTraversableHandle handle );

/// Returns a pointer to the geometry acceleration structure from its traversable handle.
///
/// Returns 0 if the traversable is not a geometry acceleration structure.
///
/// Available in all OptiX program types
static __device__ __forceinline__ CUdeviceptr optixGetGASPointerFromHandle( OptixTraversableHandle handle );
/// Reports an intersections (overload without attributes).
///
/// If optixGetRayTmin() <= hitT <= optixGetRayTmax(), the any hit program associated with this
/// intersection program (via the SBT entry) is called.
///
/// The AH program can do one of three things:
/// 1. call optixIgnoreIntersection - no hit is recorded, optixReportIntersection returns false
/// 2. call optixTerminateRay       -    hit is recorded, optixReportIntersection does not return, no further traversal occurs,
///                                                       and the associated closest hit program is called
/// 3. neither                      -    hit is recorded, optixReportIntersection returns true
///
/// hitKind - Only the 7 least significant bits should be written [0..127].  Any values above 127
/// are reserved for built in intersection.  The value can be queried with optixGetHitKind() in AH
/// and CH.
///
/// The attributes specified with a0..a7 are available in the AH and CH programs.  Note that the
/// attributes available in the CH program correspond to the closest recorded intersection.  The
/// number of attributes in registers and memory can be configured in the pipeline.
///
/// \param[in] hitT
/// \param[in] hitKind
///
/// Available in IS
static __forceinline__ __device__ bool optixReportIntersection( float hitT, unsigned int hitKind );

/// Reports an intersection (overload with 1 attribute register).
///
/// \see #optixReportIntersection(float,unsigned int)
///
/// Available in IS
static __forceinline__ __device__ bool optixReportIntersection( float hitT, unsigned int hitKind, unsigned int a0 );

/// Reports an intersection (overload with 2 attribute registers).
///
/// \see #optixReportIntersection(float,unsigned int)
///
/// Available in IS
static __forceinline__ __device__ bool optixReportIntersection( float hitT, unsigned int hitKind, unsigned int a0, unsigned int a1 );

/// Reports an intersection (overload with 3 attribute registers).
///
/// \see #optixReportIntersection(float,unsigned int)
///
/// Available in IS
static __forceinline__ __device__ bool optixReportIntersection( float hitT, unsigned int hitKind, unsigned int a0, unsigned int a1, unsigned int a2 );

/// Reports an intersection (overload with 4 attribute registers).
///
/// \see #optixReportIntersection(float,unsigned int)
///
/// Available in IS
static __forceinline__ __device__ bool optixReportIntersection( float        hitT,
                                                                unsigned int hitKind,
                                                                unsigned int a0,
                                                                unsigned int a1,
                                                                unsigned int a2,
                                                                unsigned int a3 );

/// Reports an intersection (overload with 5 attribute registers).
///
/// \see #optixReportIntersection(float,unsigned int)
///
/// Available in IS
static __forceinline__ __device__ bool optixReportIntersection( float        hitT,
                                                                unsigned int hitKind,
                                                                unsigned int a0,
                                                                unsigned int a1,
                                                                unsigned int a2,
                                                                unsigned int a3,
                                                                unsigned int a4 );

/// Reports an intersection (overload with 6 attribute registers).
///
/// \see #optixReportIntersection(float,unsigned int)
///
/// Available in IS
static __forceinline__ __device__ bool optixReportIntersection( float        hitT,
                                                                unsigned int hitKind,
                                                                unsigned int a0,
                                                                unsigned int a1,
                                                                unsigned int a2,
                                                                unsigned int a3,
                                                                unsigned int a4,
                                                                unsigned int a5 );

/// Reports an intersection (overload with 7 attribute registers).
///
/// \see #optixReportIntersection(float,unsigned int)
///
/// Available in IS
static __forceinline__ __device__ bool optixReportIntersection( float        hitT,
                                                                unsigned int hitKind,
                                                                unsigned int a0,
                                                                unsigned int a1,
                                                                unsigned int a2,
                                                                unsigned int a3,
                                                                unsigned int a4,
                                                                unsigned int a5,
                                                                unsigned int a6 );

/// Reports an intersection (overload with 8 attribute registers).
///
/// \see #optixReportIntersection(float,unsigned int)
///
/// Available in IS
static __forceinline__ __device__ bool optixReportIntersection( float        hitT,
                                                                unsigned int hitKind,
                                                                unsigned int a0,
                                                                unsigned int a1,
                                                                unsigned int a2,
                                                                unsigned int a3,
                                                                unsigned int a4,
                                                                unsigned int a5,
                                                                unsigned int a6,
                                                                unsigned int a7 );

/// Returns the attribute at the given slot index. There are up to 8 attributes available. The
/// number of attributes is configured with OptixPipelineCompileOptions::numAttributeValues.
///
/// Available in AH, CH
static __forceinline__ __device__ unsigned int optixGetAttribute_0();
static __forceinline__ __device__ unsigned int optixGetAttribute_1();
static __forceinline__ __device__ unsigned int optixGetAttribute_2();
static __forceinline__ __device__ unsigned int optixGetAttribute_3();
static __forceinline__ __device__ unsigned int optixGetAttribute_4();
static __forceinline__ __device__ unsigned int optixGetAttribute_5();
static __forceinline__ __device__ unsigned int optixGetAttribute_6();
static __forceinline__ __device__ unsigned int optixGetAttribute_7();


/// Return the attribute at the given slot index for the current outgoing hit object. There are up
/// to 8 attributes available. The number of attributes is configured with
/// OptixPipelineCompileOptions::numAttributeValues.
///
/// Results are undefined if the hit object is a miss.
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ unsigned int optixHitObjectGetAttribute_0();
static __forceinline__ __device__ unsigned int optixHitObjectGetAttribute_1();
static __forceinline__ __device__ unsigned int optixHitObjectGetAttribute_2();
static __forceinline__ __device__ unsigned int optixHitObjectGetAttribute_3();
static __forceinline__ __device__ unsigned int optixHitObjectGetAttribute_4();
static __forceinline__ __device__ unsigned int optixHitObjectGetAttribute_5();
static __forceinline__ __device__ unsigned int optixHitObjectGetAttribute_6();
static __forceinline__ __device__ unsigned int optixHitObjectGetAttribute_7();

/// Record the hit, stops traversal, and proceeds to CH.
///
/// Available in AH
static __forceinline__ __device__ void optixTerminateRay();

/// Discards the hit, and returns control to the calling optixReportIntersection or built-in
/// intersection routine.
///
/// Available in AH
static __forceinline__ __device__ void optixIgnoreIntersection();


/// For a given OptixBuildInputTriangleArray the number of primitives is defined as
///
/// "(OptixBuildInputTriangleArray::indexBuffer == 0) ? OptixBuildInputTriangleArray::numVertices/3 :
///                                                     OptixBuildInputTriangleArray::numIndexTriplets;".
///
/// For a given OptixBuildInputCustomPrimitiveArray the number of primitives is defined as numAabbs.
///
/// The primitive index returns the index into the array of primitives plus the
/// primitiveIndexOffset.
///
/// In IS and AH this corresponds to the currently intersected primitive.
///
/// In CH this corresponds to the primitive index of the closest intersected primitive.
///
/// Available in IS, AH, CH
static __forceinline__ __device__ unsigned int optixGetPrimitiveIndex();

/// Return the primitive index associated with the current outgoing hit object.
///
/// Results are undefined if the hit object is a miss.
///
/// See #optixGetPrimitiveIndex() for more details.
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ unsigned int optixHitObjectGetPrimitiveIndex();

/// Returns the Sbt GAS index of the primitive associated with the current intersection.
///
/// In IS and AH this corresponds to the currently intersected primitive.
///
/// In CH this corresponds to the SBT GAS index of the closest intersected primitive.
///
/// Available in IS, AH, CH
static __forceinline__ __device__ unsigned int optixGetSbtGASIndex();

/// Return the SBT GAS index of the closest intersected primitive associated with the current
/// outgoing hit object.
///
/// Results are undefined if the hit object is a miss.
///
/// See #optixGetSbtGASIndex() for details on the version for the incoming hit object.
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ unsigned int optixHitObjectGetSbtGASIndex();


/// Returns the OptixInstance::instanceId of the instance within the top level acceleration
/// structure associated with the current intersection.
///
/// When building an acceleration structure using OptixBuildInputInstanceArray each OptixInstance
/// has a user supplied instanceId.  OptixInstance objects reference another acceleration structure.
/// During traversal the acceleration structures are visited top down.  In the IS and AH programs
/// the OptixInstance::instanceId corresponding to the most recently visited OptixInstance is
/// returned when calling optixGetInstanceId().  In CH optixGetInstanceId() returns the
/// OptixInstance::instanceId when the hit was recorded with optixReportIntersection.  In the case
/// where there is no OptixInstance visited, optixGetInstanceId returns 0
///
/// Available in IS, AH, CH
static __forceinline__ __device__ unsigned int optixGetInstanceId();

/// Returns the OptixInstance::instanceId of the instance within the top level acceleration
/// structure associated with the outgoing hit object.
///
/// Results are undefined if the hit object is a miss.
///
/// See #optixGetInstanceId().
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ unsigned int optixHitObjectGetInstanceId();

/// Returns the zero-based index of the instance within its instance acceleration structure
/// associated with the current intersection.
///
/// In the IS and AH programs the index corresponding to the most recently visited OptixInstance is
/// returned when calling optixGetInstanceIndex().  In CH optixGetInstanceIndex() returns the index
/// when the hit was recorded with optixReportIntersection.  In the case where there is no
/// OptixInstance visited, optixGetInstanceIndex returns 0
///
/// Available in IS, AH, CH
static __forceinline__ __device__ unsigned int optixGetInstanceIndex();

/// Returns the zero-based index of the instance within its instance acceleration structure
/// associated with the outgoing hit object.
///
/// Results are undefined if the hit object is a miss.
///
/// See #optixGetInstanceIndex().
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ unsigned int optixHitObjectGetInstanceIndex();

/// Returns the 8 bit hit kind associated with the current hit.
///
/// Use optixGetPrimitiveType() to interpret the hit kind.  For custom intersections (primitive type
/// OPTIX_PRIMITIVE_TYPE_CUSTOM), this is the 7-bit hitKind passed to optixReportIntersection().
/// Hit kinds greater than 127 are reserved for built-in primitives.
///
/// Available in AH and CH
static __forceinline__ __device__ unsigned int optixGetHitKind();

/// Returns the 8 bit hit kind associated with the current outgoing hit object.
///
/// Results are undefined if the hit object is a miss.
///
/// See #optixGetHitKind().
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ unsigned int optixHitObjectGetHitKind();

/// Function interpreting the result of #optixGetHitKind().
///
/// Available in all OptiX program types
static __forceinline__ __device__ OptixPrimitiveType optixGetPrimitiveType( unsigned int hitKind );

/// Function interpreting the result of #optixGetHitKind().
///
/// Available in all OptiX program types
static __forceinline__ __device__ bool optixIsFrontFaceHit( unsigned int hitKind );

/// Function interpreting the result of #optixGetHitKind().
///
/// Available in all OptiX program types
static __forceinline__ __device__ bool optixIsBackFaceHit( unsigned int hitKind );

/// Function interpreting the hit kind associated with the current optixReportIntersection.
///
/// Available in AH, CH
static __forceinline__ __device__ OptixPrimitiveType optixGetPrimitiveType();

/// Function interpreting the hit kind associated with the current optixReportIntersection.
///
/// Available in AH, CH
static __forceinline__ __device__ bool optixIsFrontFaceHit();

/// Function interpreting the hit kind associated with the current optixReportIntersection.
///
/// Available in AH, CH
static __forceinline__ __device__ bool optixIsBackFaceHit();

/// Convenience function interpreting the result of #optixGetHitKind().
///
/// Available in AH, CH
static __forceinline__ __device__ bool optixIsTriangleHit();

/// Convenience function interpreting the result of #optixGetHitKind().
///
/// Available in AH, CH
static __forceinline__ __device__ bool optixIsTriangleFrontFaceHit();

/// Convenience function interpreting the result of #optixGetHitKind().
///
/// Available in AH, CH
static __forceinline__ __device__ bool optixIsTriangleBackFaceHit();

/// Convenience function interpreting the result of #optixGetHitKind().
///
/// Available in AH, CH
static __forceinline__ __device__ bool optixIsDisplacedMicromeshTriangleHit();

/// Convenience function interpreting the result of #optixGetHitKind().
///
/// Available in AH, CH
static __forceinline__ __device__ bool optixIsDisplacedMicromeshTriangleFrontFaceHit();

/// Convenience function interpreting the result of #optixGetHitKind().
///
/// Available in AH, CH
static __forceinline__ __device__ bool optixIsDisplacedMicromeshTriangleBackFaceHit();

/// Convenience function that returns the first two attributes as floats.
///
/// When using OptixBuildInputTriangleArray objects, during intersection the barycentric coordinates
/// are stored into the first two attribute registers.
///
/// Available in AH, CH
static __forceinline__ __device__ float2 optixGetTriangleBarycentrics();

/// Returns the curve parameter associated with the current intersection when using
/// OptixBuildInputCurveArray objects.
///
/// Available in AH, CH
static __forceinline__ __device__ float optixGetCurveParameter();

/// Returns the ribbon parameters along directrix (length) and generator (width) of the current
/// intersection when using OptixBuildInputCurveArray objects with curveType
/// OPTIX_PRIMITIVE_TYPE_FLAT_QUADRATIC_BSPLINE.
///
/// Available in AH, CH
static __forceinline__ __device__ float2 optixGetRibbonParameters();

/// Available in any program, it returns the current launch index within the launch dimensions
/// specified by optixLaunch on the host.
///
/// The raygen program is typically only launched once per launch index.
///
/// Available in all OptiX program types
static __forceinline__ __device__ uint3 optixGetLaunchIndex();

/// Available in any program, it returns the dimensions of the current launch specified by
/// optixLaunch on the host.
///
/// Available in all OptiX program types
static __forceinline__ __device__ uint3 optixGetLaunchDimensions();

/// Returns the generic memory space pointer to the data region (past the header) of the
/// currently active SBT record corresponding to the current program.
///
/// Note that optixGetSbtDataPointer is not available in OptiX-enabled functions, because
/// there is no SBT entry associated with the function.
///
/// Available in RG, IS, AH, CH, MS, EX, DC, CC
static __forceinline__ __device__ CUdeviceptr optixGetSbtDataPointer();

/// Device pointer address for the SBT associated with the hit or miss program for the current
/// outgoing hit object.
///
/// Returns 0 for nop hit objects.
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ CUdeviceptr optixHitObjectGetSbtDataPointer();

/// Throws a user exception with the given exception code (overload without exception details).
///
/// The exception code must be in the range from 0 to 2^30 - 1. Up to 8 optional exception details
/// can be passed. They can be queried in the EX program using optixGetExceptionDetail_0() to
/// ..._8().
///
/// The exception details must not be used to encode pointers to the stack since the current stack
/// is not preserved in the EX program.
///
/// Not available in EX
///
/// \param[in] exceptionCode The exception code to be thrown.
///
/// Available in RG, IS, AH, CH, MS, DC, CC
static __forceinline__ __device__ void optixThrowException( int exceptionCode );

/// Throws a user exception with the given exception code (overload with 1 exception detail).
///
/// \see #optixThrowException(int)
///
/// Available in RG, IS, AH, CH, MS, DC, CC
static __forceinline__ __device__ void optixThrowException( int exceptionCode, unsigned int exceptionDetail0 );

/// Throws a user exception with the given exception code (overload with 2 exception details).
///
/// \see #optixThrowException(int)
///
/// Available in RG, IS, AH, CH, MS, DC, CC
static __forceinline__ __device__ void optixThrowException( int exceptionCode,
                                                            unsigned int exceptionDetail0,
                                                            unsigned int exceptionDetail1 );

/// Throws a user exception with the given exception code (overload with 3 exception details).
///
/// \see #optixThrowException(int)
///
/// Available in RG, IS, AH, CH, MS, DC, CC
static __forceinline__ __device__ void optixThrowException( int exceptionCode,
                                                            unsigned int exceptionDetail0,
                                                            unsigned int exceptionDetail1,
                                                            unsigned int exceptionDetail2 );

/// Throws a user exception with the given exception code (overload with 4 exception details).
///
/// \see #optixThrowException(int)
///
/// Available in RG, IS, AH, CH, MS, DC, CC
static __forceinline__ __device__ void optixThrowException( int exceptionCode,
                                                            unsigned int exceptionDetail0,
                                                            unsigned int exceptionDetail1,
                                                            unsigned int exceptionDetail2,
                                                            unsigned int exceptionDetail3 );

/// Throws a user exception with the given exception code (overload with 5 exception details).
///
/// \see #optixThrowException(int)
///
/// Available in RG, IS, AH, CH, MS, DC, CC
static __forceinline__ __device__ void optixThrowException( int exceptionCode,
                                                            unsigned int exceptionDetail0,
                                                            unsigned int exceptionDetail1,
                                                            unsigned int exceptionDetail2,
                                                            unsigned int exceptionDetail3,
                                                            unsigned int exceptionDetail4 );

/// Throws a user exception with the given exception code (overload with 6 exception details).
///
/// \see #optixThrowException(int)
///
/// Available in RG, IS, AH, CH, MS, DC, CC
static __forceinline__ __device__ void optixThrowException( int exceptionCode,
                                                            unsigned int exceptionDetail0,
                                                            unsigned int exceptionDetail1,
                                                            unsigned int exceptionDetail2,
                                                            unsigned int exceptionDetail3,
                                                            unsigned int exceptionDetail4,
                                                            unsigned int exceptionDetail5 );

/// Throws a user exception with the given exception code (overload with 7 exception
/// details).
///
/// \see #optixThrowException(int)
///
/// Available in RG, IS, AH, CH, MS, DC, CC
static __forceinline__ __device__ void optixThrowException( int exceptionCode,
                                                            unsigned int exceptionDetail0,
                                                            unsigned int exceptionDetail1,
                                                            unsigned int exceptionDetail2,
                                                            unsigned int exceptionDetail3,
                                                            unsigned int exceptionDetail4,
                                                            unsigned int exceptionDetail5,
                                                            unsigned int exceptionDetail6 );

/// Throws a user exception with the given exception code (overload with 8 exception details).
///
/// \see #optixThrowException(int)
///
/// Available in RG, IS, AH, CH, MS, DC, CC
static __forceinline__ __device__ void optixThrowException( int exceptionCode,
                                                            unsigned int exceptionDetail0,
                                                            unsigned int exceptionDetail1,
                                                            unsigned int exceptionDetail2,
                                                            unsigned int exceptionDetail3,
                                                            unsigned int exceptionDetail4,
                                                            unsigned int exceptionDetail5,
                                                            unsigned int exceptionDetail6,
                                                            unsigned int exceptionDetail7 );

/// Returns the exception code.
///
/// Available in EX
static __forceinline__ __device__ int optixGetExceptionCode();

/// Returns the 32-bit exception detail at slot 0.
///
/// The behavior is undefined if the exception is not a user exception, or the used overload
/// #optixThrowException() did not provide the queried exception detail.
///
/// Available in EX
static __forceinline__ __device__ unsigned int optixGetExceptionDetail_0();

/// Returns the 32-bit exception detail at slot 1.
///
/// \see #optixGetExceptionDetail_0()
///
/// Available in EX
static __forceinline__ __device__ unsigned int optixGetExceptionDetail_1();

/// Returns the 32-bit exception detail at slot 2.
///
/// \see #optixGetExceptionDetail_0()
///
/// Available in EX
static __forceinline__ __device__ unsigned int optixGetExceptionDetail_2();

/// Returns the 32-bit exception detail at slot 3.
///
/// \see #optixGetExceptionDetail_0()
///
/// Available in EX
static __forceinline__ __device__ unsigned int optixGetExceptionDetail_3();

/// Returns the 32-bit exception detail at slot 4.
///
/// \see #optixGetExceptionDetail_0()
///
/// Available in EX
static __forceinline__ __device__ unsigned int optixGetExceptionDetail_4();

/// Returns the 32-bit exception detail at slot 5.
///
/// \see #optixGetExceptionDetail_0()
///
/// Available in EX
static __forceinline__ __device__ unsigned int optixGetExceptionDetail_5();

/// Returns the 32-bit exception detail at slot 6.
///
/// \see #optixGetExceptionDetail_0()
///
/// Available in EX
static __forceinline__ __device__ unsigned int optixGetExceptionDetail_6();

/// Returns the 32-bit exception detail at slot 7.
///
/// \see #optixGetExceptionDetail_0()
///
/// Available in EX
static __forceinline__ __device__ unsigned int optixGetExceptionDetail_7();


/// Returns a string that includes information about the source location that caused the current
/// exception.
///
/// The source location is only available for user exceptions.
/// Line information needs to be present in the input PTX and
/// OptixModuleCompileOptions::debugLevel may not be set to OPTIX_COMPILE_DEBUG_LEVEL_NONE.
///
/// Returns a NULL pointer if no line information is available.
///
/// Available in EX
static __forceinline__ __device__ char* optixGetExceptionLineInfo();

/// Creates a call to the direct callable program at the specified SBT entry.
///
/// This will call the program that was specified in the
/// OptixProgramGroupCallables::entryFunctionNameDC in the module specified by
/// OptixProgramGroupCallables::moduleDC.
///
/// The address of the SBT entry is calculated by:
///  OptixShaderBindingTable::callablesRecordBase + ( OptixShaderBindingTable::callablesRecordStrideInBytes * sbtIndex ).
///
/// Direct callable programs are allowed to call optixTrace, but any secondary trace calls invoked
/// from subsequently called CH, MS and callable programs will result an an error.
///
/// Behavior is undefined if there is no direct callable program at the specified SBT entry.
///
/// Behavior is undefined if the number of arguments that are being passed in does not match the
/// number of parameters expected by the program that is called. In validation mode an exception
/// will be generated.
///
/// \param[in] sbtIndex The offset of the SBT entry of the direct callable program to call relative
/// to OptixShaderBindingTable::callablesRecordBase.  \param[in] args The arguments to pass to the
/// direct callable program.
///
/// Available in RG, IS, AH, CH, MS, DC, CC
template <typename ReturnT, typename... ArgTypes>
static __forceinline__ __device__ ReturnT optixDirectCall( unsigned int sbtIndex, ArgTypes... args );


/// Creates a call to the continuation callable program at the specified SBT entry.
///
/// This will call the program that was specified in the
/// OptixProgramGroupCallables::entryFunctionNameCC in the module specified by
/// OptixProgramGroupCallables::moduleCC.
///
/// The address of the SBT entry is calculated by:
///  OptixShaderBindingTable::callablesRecordBase + ( OptixShaderBindingTable::callablesRecordStrideInBytes * sbtIndex ).
///
/// As opposed to direct callable programs, continuation callable programs are allowed to make
/// secondary optixTrace calls.
///
/// Behavior is undefined if there is no continuation callable program at the specified SBT entry.
///
/// Behavior is undefined if the number of arguments that are being passed in does not match the
/// number of parameters expected by the program that is called. In validation mode an exception
/// will be generated.
///
/// \param[in] sbtIndex The offset of the SBT entry of the continuation callable program to call relative to OptixShaderBindingTable::callablesRecordBase.
/// \param[in] args The arguments to pass to the continuation callable program.
///
/// Available in RG, CH, MS, CC
template <typename ReturnT, typename... ArgTypes>
static __forceinline__ __device__ ReturnT optixContinuationCall( unsigned int sbtIndex, ArgTypes... args );


/// optixTexFootprint2D calculates the footprint of a corresponding 2D texture fetch (non-mipmapped).
///
/// On Turing and subsequent architectures, a texture footprint instruction allows user programs to
/// determine the set of texels that would be accessed by an equivalent filtered texture lookup.
///
/// \param[in] tex      CUDA texture object (cast to 64-bit integer)
/// \param[in] texInfo  Texture info packed into 32-bit integer, described below.
/// \param[in] x        Texture coordinate
/// \param[in] y        Texture coordinate
/// \param[out] singleMipLevel  Result indicating whether the footprint spans only a single miplevel.
///
/// The texture info argument is a packed 32-bit integer with the following layout:
///
///   texInfo[31:29] = reserved (3 bits)
///   texInfo[28:24] = miplevel count (5 bits)
///   texInfo[23:20] = log2 of tile width (4 bits)
///   texInfo[19:16] = log2 of tile height (4 bits)
///   texInfo[15:10] = reserved (6 bits)
///   texInfo[9:8]   = horizontal wrap mode (2 bits) (CUaddress_mode)
///   texInfo[7:6]   = vertical wrap mode (2 bits) (CUaddress_mode)
///   texInfo[5]     = mipmap filter mode (1 bit) (CUfilter_mode)
///   texInfo[4:0]   = maximum anisotropy (5 bits)
///
/// Returns a 16-byte structure (as a uint4) that stores the footprint of a texture request at a
/// particular "granularity", which has the following layout:
///
///    struct Texture2DFootprint
///    {
///        unsigned long long mask;
///        unsigned int tileY : 12;
///        unsigned int reserved1 : 4;
///        unsigned int dx : 3;
///        unsigned int dy : 3;
///        unsigned int reserved2 : 2;
///        unsigned int granularity : 4;
///        unsigned int reserved3 : 4;
///        unsigned int tileX : 12;
///        unsigned int level : 4;
///        unsigned int reserved4 : 16;
///    };
///
/// The granularity indicates the size of texel groups that are represented by an 8x8 bitmask. For
/// example, a granularity of 12 indicates texel groups that are 128x64 texels in size. In a
/// footprint call, The returned granularity will either be the actual granularity of the result, or
/// 0 if the footprint call was able to honor the requested granularity (the usual case).
///
/// level is the mip level of the returned footprint. Two footprint calls are needed to get the
/// complete footprint when a texture call spans multiple mip levels.
///
/// mask is an 8x8 bitmask of texel groups that are covered, or partially covered, by the footprint.
/// tileX and tileY give the starting position of the mask in 8x8 texel-group blocks.  For example,
/// suppose a granularity of 12 (128x64 texels), and tileX=3 and tileY=4. In this case, bit 0 of the
/// mask (the low order bit) corresponds to texel group coordinates (3*8, 4*8), and texel
/// coordinates (3*8*128, 4*8*64), within the specified mip level.
///
/// If nonzero, dx and dy specify a "toroidal rotation" of the bitmask.  Toroidal rotation of a
/// coordinate in the mask simply means that its value is reduced by 8.  Continuing the example from
/// above, if dx=0 and dy=0 the mask covers texel groups (3*8, 4*8) to (3*8+7, 4*8+7) inclusive.
/// If, on the other hand, dx=2, the rightmost 2 columns in the mask have their x coordinates
/// reduced by 8, and similarly for dy.
///
/// See the OptiX SDK for sample code that illustrates how to unpack the result.
///
/// Available anywhere
static __forceinline__ __device__ uint4 optixTexFootprint2D( unsigned long long tex, unsigned int texInfo, float x, float y, unsigned int* singleMipLevel );

/// optixTexFootprint2DLod calculates the footprint of a corresponding 2D texture fetch (tex2DLod)
/// \param[in] tex      CUDA texture object (cast to 64-bit integer)
/// \param[in] texInfo  Texture info packed into 32-bit integer, described below.
/// \param[in] x        Texture coordinate
/// \param[in] y        Texture coordinate
/// \param[in] level    Level of detail (lod)
/// \param[in] coarse   Requests footprint from coarse miplevel, when the footprint spans two levels.
/// \param[out] singleMipLevel  Result indicating whether the footprint spans only a single miplevel.
/// \see #optixTexFootprint2D(unsigned long long,unsigned int,float,float,unsigned int*)
///
/// Available anywhere
static __forceinline__ __device__ uint4
optixTexFootprint2DLod( unsigned long long tex, unsigned int texInfo, float x, float y, float level, bool coarse, unsigned int* singleMipLevel );

/// optixTexFootprint2DGrad calculates the footprint of a corresponding 2D texture fetch (tex2DGrad)
/// \param[in] tex      CUDA texture object (cast to 64-bit integer)
/// \param[in] texInfo  Texture info packed into 32-bit integer, described below.
/// \param[in] x        Texture coordinate
/// \param[in] y        Texture coordinate
/// \param[in] dPdx_x   Derivative of x coordinte, which determines level of detail.
/// \param[in] dPdx_y   Derivative of x coordinte, which determines level of detail.
/// \param[in] dPdy_x   Derivative of y coordinte, which determines level of detail.
/// \param[in] dPdy_y   Derivative of y coordinte, which determines level of detail.
/// \param[in] coarse   Requests footprint from coarse miplevel, when the footprint spans two levels.
/// \param[out] singleMipLevel  Result indicating whether the footprint spans only a single miplevel.
/// \see #optixTexFootprint2D(unsigned long long,unsigned int,float,float,unsigned int*)
///
/// Available anywhere
static __forceinline__ __device__ uint4 optixTexFootprint2DGrad( unsigned long long tex,
                                                                 unsigned int       texInfo,
                                                                 float              x,
                                                                 float              y,
                                                                 float              dPdx_x,
                                                                 float              dPdx_y,
                                                                 float              dPdy_x,
                                                                 float              dPdy_y,
                                                                 bool               coarse,
                                                                 unsigned int*      singleMipLevel );

/**@}*/  // end group optix_device_api

#define __OPTIX_INCLUDE_INTERNAL_HEADERS__

#include "internal/optix_device_impl.h"

#endif  // OPTIX_OPTIX_DEVICE_H
