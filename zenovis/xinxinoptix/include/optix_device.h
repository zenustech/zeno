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
/// exist even if there are no calls to optixTraverse, optixMakeMissHitObject, optixMakeHitObject
/// or optixMakeNopHitObject.
///
/// \param[in,out] payload       up to 32 unsigned int values that hold the payload
///
/// Available in RG, CH, MS, CC
template <typename... Payload>
static __forceinline__ __device__ void optixInvoke( Payload&... payload );

/// Invokes closesthit, miss or nop based on the current outgoing hit object. After execution the
/// current outgoing hit object will be set to nop. An implied nop hit object is always assumed to
/// exist even if there are no calls to optixTraverse, optixMakeMissHitObject, optixMakeHitObject
/// or optixMakeNopHitObject.
///
/// \param[in] type
/// \param[in,out] payload       up to 32 unsigned int values that hold the payload
///
/// Available in RG, CH, MS, CC
template <typename... Payload>
static __forceinline__ __device__ void optixInvoke( OptixPayloadTypeID type, Payload&... payload );

/// Constructs an outgoing hit object from the hit object data provided. The traverseData needs to be collected from a previous hit
/// object using #optixHitObjectGetTraverseData.
/// This hit object will now become the current outgoing hit object and will overwrite the current outgoing hit object.
///
/// \param[in] handle
/// \param[in] rayOrigin
/// \param[in] rayDirection
/// \param[in] tmin
/// \param[in] rayTime
/// \param[in] rayFlags       really only 16 bits, combination of OptixRayFlags
/// \param[in] traverseData
/// \param[in] transforms
/// \param[in] numTransforms
///
/// Available in RG, CH, MS, CC
static __forceinline__ __device__ void optixMakeHitObject( OptixTraversableHandle        handle,
                                                           float3                        rayOrigin,
                                                           float3                        rayDirection,
                                                           float                         tmin,
                                                           float                         rayTime,
                                                           unsigned int                  rayFlags,
                                                           OptixTraverseData             traverseData,
                                                           const OptixTraversableHandle* transforms,
                                                           unsigned int                  numTransforms );

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
/// \param[in] rayFlags       really only 16 bits, combination of OptixRayFlags
///
/// Available in RG, CH, MS, CC
static __forceinline__ __device__ void optixMakeMissHitObject( unsigned int missSBTIndex,
                                                               float3       rayOrigin,
                                                               float3       rayDirection,
                                                               float        tmin,
                                                               float        tmax,
                                                               float        rayTime,
                                                               unsigned int rayFlags );

/// Constructs an outgoing hit object that when invoked does nothing (neither the miss nor the
/// closest hit shader will be invoked). This hit object will now become the current outgoing hit
/// object and will overwrite the current outgoing hit object. Accessors such as
/// #optixHitObjectGetInstanceId will return 0 or 0 filled structs. Only #optixHitObjectIsNop
/// will return a non-zero result.
///
/// Available in RG, CH, MS, CC
static __forceinline__ __device__ void optixMakeNopHitObject();

/// Serializes the current outgoing hit object which allows to recreate it at a later
/// point using #optixMakeHitObject.
///
/// \param[out] data
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ void optixHitObjectGetTraverseData( OptixTraverseData* data );

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

/// Sets the SBT record index in the current outgoing hit object.
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ void optixHitObjectSetSbtRecordIndex( unsigned int sbtRecordIndex );

/// Returns the traversable handle for the Geometry Acceleration Structure (GAS) associated
/// with the current outgoing hit object.
/// Returns 0 if the hit object is not a hit.
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ OptixTraversableHandle optixHitObjectGetGASTraversableHandle();

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

/// If non-zero it is legal to call optixTrace or optixTraverse without triggering an
/// OPTIX_EXCEPTION_CODE_TRACE_DEPTH_EXCEEDED exception. In the case of optixTrace it
/// represents the number of recursive calls that are remaining and counts down.
///
/// Value is in the range of [0..OptixPipelineLinkOptions::maxTraceDepth], and
/// maxTraceDepth has a maximum value of 31.
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ unsigned int optixGetRemainingTraceDepth();

/// Returns the rayOrigin passed into optixTrace.
///
/// May be more expensive to call in IS and AH than their object space counterparts, so effort
/// should be made to use the object space ray in those programs.
///
/// Available in IS, AH, CH, MS
static __forceinline__ __device__ float3 optixGetWorldRayOrigin();

/// Returns the rayOrigin passed into optixTraverse, optixMakeHitObject or optixMakeMissHitObject.
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

/// Returns the rayDirection passed into optixTraverse, optixMakeHitObject or optixMakeMissHitObject.
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

/// Returns the tmin passed into optixTraverse, optixMakeHitObject or optixMakeMissHitObject.
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
/// If the hit object is a miss, returns the tmax passed into optixTraverse, optixMakeHitObject or
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

/// Returns the rayTime passed into optixTraverse, optixMakeHitObject or optixMakeMissHitObject.
///
/// Returns 0 for nop hit objects or when motion is disabled.
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ float optixHitObjectGetRayTime();

/// Returns the rayFlags passed into optixTrace
///
/// Available in IS, AH, CH, MS
static __forceinline__ __device__ unsigned int optixGetRayFlags();

/// Returns the rayFlags passed into optixTrace associated with the current outgoing hit object.
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ unsigned int optixHitObjectGetRayFlags();

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

/// [DEPRECATED] Returns the object space triangle vertex positions of a given triangle in a Geometry Acceleration
/// Structure (GAS) at a given motion time.
/// This function is deprecated, use optixGetTriangleVertexDataFromHandle for random access triangle vertex data fetch or
/// the overload optixGetTriangleVertexData( float3 data[3] ) for a current triangle hit vertex data fetch.
///
/// To access vertex data, the GAS must be built using the flag
/// OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS.
///
/// If motion is disabled via OptixPipelineCompileOptions::usesMotionBlur, or the GAS does not
/// contain motion, the time parameter is ignored.
///
/// Available in all OptiX program types
static __forceinline__ __device__ void optixGetTriangleVertexData( OptixTraversableHandle gas,
                                                                   unsigned int           primIdx,
                                                                   unsigned int           sbtGASIndex,
                                                                   float                  time,
                                                                   float3                 data[3] );

/// Performs a random access data fetch object space vertex position of a given triangle in a Geometry Acceleration
/// Structure (GAS) at a given motion time.
///
/// To access vertex data of any triangle, the GAS must be built using the flag
/// OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS.
/// If only the vertex data of a currently intersected triangle is required, it is recommended to
/// use function optixGetTriangleVertexData. A data fetch of the currently hit primitive does NOT
/// require building the corresponding GAS with flag OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS.
///
/// If motion is disabled via OptixPipelineCompileOptions::usesMotionBlur, or the GAS does not
/// contain motion, the time parameter is ignored.
///
/// Available in all OptiX program types
static __forceinline__ __device__ void optixGetTriangleVertexDataFromHandle( OptixTraversableHandle gas,
                                                                             unsigned int           primIdx,
                                                                             unsigned int           sbtGASIndex,
                                                                             float                  time,
                                                                             float3                 data[3] );

/// Returns the object space triangle vertex positions of the currently intersected triangle at the current ray time.
///
/// Similar to the random access variant optixGetTriangleVertexDataFromHandle, but does not require setting flag
/// OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS when building the corresponding GAS.
///
/// It is only valid to call this function if the return value of optixGetPrimitiveType( optixGetHitKind() ) equals OPTIX_PRIMITIVE_TYPE_TRIANGLE.
///
/// Available in AH, CH
static __forceinline__ __device__ void optixGetTriangleVertexData( float3 data[3] );

/// Returns the object space triangle vertex positions of the intersected triangle for a valid outgoing hit object.
/// It is the hit object's pendant of optixGetTriangleVertexData( float3 data[3] ).
///
/// It is only valid to call this function if the return value of optixGetPrimitiveType( optixHitObjectGetHitKind() ) equals OPTIX_PRIMITIVE_TYPE_TRIANGLE.
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ void optixHitObjectGetTriangleVertexData( float3 data[3] );


/// Deprecated. Call either optixGetLinearCurveVertexData( float4 data[2] ) for a current-hit data fetch,
///  or optixGetLinearCurveVertexDataFromHandle( ... ) for a random-access data fetch.
///
/// Returns the object space curve control vertex data of a linear curve in a Geometry Acceleration
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
static __forceinline__ __device__ void optixGetLinearCurveVertexData( OptixTraversableHandle gas,
                                                                      unsigned int           primIdx,
                                                                      unsigned int           sbtGASIndex,
                                                                      float                  time,
                                                                      float4                 data[2] );

/// Performs a random access fetch of the object space curve control vertex data of a linear curve in a Geometry Acceleration
/// Structure (GAS) at a given motion time.
///
/// To access vertex data of any curve, the GAS must be built using the flag
/// OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS.
/// If only the vertex data of a currently intersected linear curve is required, it is recommended to
/// use function optixGetLinearCurveVertexData. A data fetch of the currently hit primitive does NOT
/// require building the corresponding GAS with flag OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS.
///
/// data[i] = {x,y,z,w} with {x,y,z} the position and w the radius of control vertex i.
///
/// If motion is disabled via OptixPipelineCompileOptions::usesMotionBlur, or the GAS does not
/// contain motion, the time parameter is ignored.
///
/// Available in all OptiX program types
static __forceinline__ __device__ void optixGetLinearCurveVertexDataFromHandle( OptixTraversableHandle gas,
                                                                                unsigned int           primIdx,
                                                                                unsigned int           sbtGASIndex,
                                                                                float                  time,
                                                                                float4                 data[2] );

/// Returns the object space control vertex data of the currently intersected linear curve at the current ray time.
///
/// Similar to the random access variant optixGetLinearCurveVertexDataFromHandle, but does not require setting flag
/// OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS when building the corresponding GAS.
///
/// It is only valid to call this function if the return value of optixGetPrimitiveType( optixGetHitKind() ) equals OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR.
///
/// Available in AH, CH
static __forceinline__ __device__ void optixGetLinearCurveVertexData( float4 data[2] );

/// Returns the object space control vertex data of the currently intersected linear curve for a valid outgoing hit object.
/// It is the hit object's pendant of optixGetLinearCurveVertexData( float4 data[2] ).
///
/// It is only valid to call this function if the return value of optixGetPrimitiveType( optixHitObjectGetHitKind() ) equals OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR.
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ void optixHitObjectGetLinearCurveVertexData( float4 data[2] );

/// Returns the object space curve control vertex data of a quadratic BSpline curve in a Geometry
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
static __forceinline__ __device__ void optixGetQuadraticBSplineVertexData( OptixTraversableHandle gas,
                                                                           unsigned int           primIdx,
                                                                           unsigned int           sbtGASIndex,
                                                                           float                  time,
                                                                           float4                 data[3] );

/// Returns the object space curve control vertex data of a quadratic BSpline curve in a Geometry
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
static __forceinline__ __device__ void optixGetQuadraticBSplineVertexDataFromHandle( OptixTraversableHandle gas,
                                                                                     unsigned int           primIdx,
                                                                                     unsigned int           sbtGASIndex,
                                                                                     float                  time,
                                                                                     float4                 data[3] );
static __forceinline__ __device__ void optixGetQuadraticBSplineRocapsVertexDataFromHandle( OptixTraversableHandle gas,
                                                                                           unsigned int primIdx,
                                                                                           unsigned int sbtGASIndex,
                                                                                           float        time,
                                                                                           float4       data[3] );

/// Returns the object space curve control vertex data of a quadratic BSpline curve in a Geometry
/// Acceleration Structure (GAS) at a given motion time.
///
/// data[i] = {x,y,z,w} with {x,y,z} the position and w the radius of control vertex i.
///
/// Available in AH, CH
static __forceinline__ __device__ void optixGetQuadraticBSplineVertexData( float4 data[3] );
static __forceinline__ __device__ void optixGetQuadraticBSplineRocapsVertexData( float4 data[3] );

/// Returns the object space curve control vertex data of a quadratic BSpline curve for a valid outgoing hit object.
///
/// data[i] = {x,y,z,w} with {x,y,z} the position and w the radius of control vertex i.
///
/// It is only valid to call this function if the return value of optixGetPrimitiveType( optixHitObjectGetHitKind() )
/// equals OPTIX_PRIMITIVE_TYPE_FLAT_QUADRATIC_BSPLINE.
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ void optixHitObjectGetQuadraticBSplineVertexData( float4 data[3] );
static __forceinline__ __device__ void optixHitObjectGetQuadraticBSplineRocapsVertexData( float4 data[3] );

/// Deprecated. Call either optixGetCubicBSplineVertexData( float4 data[4] ) for current hit
/// sphere data, or optixGetCubicBSplineVertexDataFromHandle() for random access sphere data.
///
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
static __forceinline__ __device__ void optixGetCubicBSplineVertexData( OptixTraversableHandle gas,
                                                                       unsigned int           primIdx,
                                                                       unsigned int           sbtGASIndex,
                                                                       float                  time,
                                                                       float4                 data[4] );

/// Returns the object space curve control vertex data of a cubic BSpline curve in a Geometry
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
static __forceinline__ __device__ void optixGetCubicBSplineVertexDataFromHandle( OptixTraversableHandle gas,
                                                                                 unsigned int           primIdx,
                                                                                 unsigned int           sbtGASIndex,
                                                                                 float                  time,
                                                                                 float4                 data[4] );
static __forceinline__ __device__ void optixGetCubicBSplineRocapsVertexDataFromHandle( OptixTraversableHandle gas,
                                                                                       unsigned int           primIdx,
                                                                                       unsigned int sbtGASIndex,
                                                                                       float        time,
                                                                                       float4       data[4] );

/// Returns the object space curve control vertex data of a cubic BSpline curve in a Geometry
/// Acceleration Structure (GAS) at a given motion time.
///
/// data[i] = {x,y,z,w} with {x,y,z} the position and w the radius of control vertex i.
///
/// Available in AH, CH
static __forceinline__ __device__ void optixGetCubicBSplineVertexData( float4 data[4] );
static __forceinline__ __device__ void optixGetCubicBSplineRocapsVertexData( float4 data[4] );

/// Returns the object space curve control vertex data of a cubic BSpline curve for a valid
/// outgoing hit object.
///
/// data[i] = {x,y,z,w} with {x,y,z} the position and w the radius of control vertex i.
///
/// It is only valid to call this function if the return value of optixGetPrimitiveType( optixHitObjectGetHitKind() )
/// equals OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE.
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ void optixHitObjectGetCubicBSplineVertexData( float4 data[4] );
/// See #optixHitObjectGetCubicBSplineVertexData for further documentation
///
/// It is only valid to call this function if the return value of optixGetPrimitiveType( optixHitObjectGetHitKind() )
/// equals OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE_ROCAPS.
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ void optixHitObjectGetCubicBSplineRocapsVertexData( float4 data[4] );

/// Deprecated. Call either optixGetCatmullRomVertexData( float4 data[4] ) for current hit
/// data, or optixGetCatmullRomVertexDataFromHandle() for random access sphere data.
///
/// Returns the object space curve control vertex data of a CatmullRom spline curve in a Geometry
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
static __forceinline__ __device__ void optixGetCatmullRomVertexData( OptixTraversableHandle gas,
                                                                     unsigned int           primIdx,
                                                                     unsigned int           sbtGASIndex,
                                                                     float                  time,
                                                                     float4                 data[4] );

/// Returns the object space curve control vertex data of a CatmullRom spline curve in a Geometry
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
static __forceinline__ __device__ void optixGetCatmullRomVertexDataFromHandle( OptixTraversableHandle gas,
                                                                               unsigned int           primIdx,
                                                                               unsigned int           sbtGASIndex,
                                                                               float                  time,
                                                                               float4                 data[4] );
static __forceinline__ __device__ void optixGetCatmullRomRocapsVertexDataFromHandle( OptixTraversableHandle gas,
                                                                                     unsigned int           primIdx,
                                                                                     unsigned int           sbtGASIndex,
                                                                                     float                  time,
                                                                                     float4                 data[4] );

/// Returns the object space curve control vertex data of a CatmullRom spline curve in a Geometry
/// Acceleration Structure (GAS) at a given motion time.
///
/// data[i] = {x,y,z,w} with {x,y,z} the position and w the radius of control vertex i.
///
/// Available in AH, CH
static __forceinline__ __device__ void optixGetCatmullRomVertexData( float4 data[4] );
static __forceinline__ __device__ void optixGetCatmullRomRocapsVertexData( float4 data[4] );

/// Returns the object space curve control vertex data of a CatmullRom spline curve for a valid
/// outgoing hit object.
///
/// data[i] = {x,y,z,w} with {x,y,z} the position and w the radius of control vertex i.
///
/// It is only valid to call this function if the return value of optixGetPrimitiveType( optixHitObjectGetHitKind() )
/// equals OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM.
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ void optixHitObjectGetCatmullRomVertexData( float4 data[4] );
static __forceinline__ __device__ void optixHitObjectGetCatmullRomRocapsVertexData( float4 data[4] );

/// Deprecated. Call either optixGetCubicBezierVertexData( float4 data[4] ) for current hit
/// data, or optixGetCubicBezierVertexDataFromHandle() for random access sphere data.
///
/// Returns the object space curve control vertex data of a cubic Bezier curve in a Geometry
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
static __forceinline__ __device__ void optixGetCubicBezierVertexData( OptixTraversableHandle gas,
                                                                      unsigned int           primIdx,
                                                                      unsigned int           sbtGASIndex,
                                                                      float                  time,
                                                                      float4                 data[4] );

/// Returns the object space curve control vertex data of a cubic Bezier curve in a Geometry
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
static __forceinline__ __device__ void optixGetCubicBezierVertexDataFromHandle( OptixTraversableHandle gas,
                                                                                unsigned int           primIdx,
                                                                                unsigned int           sbtGASIndex,
                                                                                float                  time,
                                                                                float4                 data[4] );
static __forceinline__ __device__ void optixGetCubicBezierRocapsVertexDataFromHandle( OptixTraversableHandle gas,
                                                                                      unsigned int           primIdx,
                                                                                      unsigned int sbtGASIndex,
                                                                                      float        time,
                                                                                      float4       data[4] );

/// Returns the object space curve control vertex data of a cubic Bezier curve in a Geometry
/// Acceleration Structure (GAS) at a given motion time.
///
/// data[i] = {x,y,z,w} with {x,y,z} the position and w the radius of control vertex i.
///
/// Available in AH, CH
static __forceinline__ __device__ void optixGetCubicBezierVertexData( float4 data[4] );
static __forceinline__ __device__ void optixGetCubicBezierRocapsVertexData( float4 data[4] );

/// Returns the object space curve control vertex data of a cubic Bezier curve for a valid
/// outgoing hit object.
///
/// data[i] = {x,y,z,w} with {x,y,z} the position and w the radius of control vertex i.
///
/// It is only valid to call this function if the return value of optixGetPrimitiveType( optixHitObjectGetHitKind() )
/// equals OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER.
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ void optixHitObjectGetCubicBezierVertexData( float4 data[4] );
static __forceinline__ __device__ void optixHitObjectGetCubicBezierRocapsVertexData( float4 data[4] );

/// Deprecated. Call either optixGetRibbonVertexData( float4 data[3] ) for current hit
/// data, or optixGetRibbonVertexDataFromHandle() for random access.
///
/// Returns the object space curve control vertex data of a ribbon (flat quadratic BSpline) in a
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
static __forceinline__ __device__ void optixGetRibbonVertexData( OptixTraversableHandle gas,
                                                                 unsigned int           primIdx,
                                                                 unsigned int           sbtGASIndex,
                                                                 float                  time,
                                                                 float4                 data[3] );

/// Returns the object space curve control vertex data of a ribbon (flat quadratic BSpline) in a
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
static __forceinline__ __device__ void optixGetRibbonVertexDataFromHandle( OptixTraversableHandle gas,
                                                                           unsigned int           primIdx,
                                                                           unsigned int           sbtGASIndex,
                                                                           float                  time,
                                                                           float4                 data[3] );

/// Returns the object space curve control vertex data of a ribbon (flat quadratic BSpline) in a
/// Geometry Acceleration Structure (GAS) at a given motion time.
///
/// data[i] = {x,y,z,w} with {x,y,z} the position and w the radius of control vertex i.
///
/// Available in AH, CH
static __forceinline__ __device__ void optixGetRibbonVertexData( float4 data[3] );

/// Returns the object space curve control vertex data of a ribbon (flat quadratic BSpline) for a valid
/// outgoing hit object.
///
/// data[i] = {x,y,z,w} with {x,y,z} the position and w the radius of control vertex i.
///
/// It is only valid to call this function if the return value of optixGetPrimitiveType( optixHitObjectGetHitKind() )
/// equals OPTIX_PRIMITIVE_TYPE_FLAT_QUADRATIC_BSPLINE.
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ void optixHitObjectGetRibbonVertexData( float4 data[3] );

/// Deprecated. Call either optixGetRibbonNormal( float2 ribbonParameters ) for current hit
/// data, or optixGetRibbonNormalFromHandle() for random access.
///
/// Returns ribbon normal at intersection reported by optixReportIntersection.
///
/// Available in all OptiX program types
static __forceinline__ __device__ float3 optixGetRibbonNormal( OptixTraversableHandle gas,
                                                               unsigned int           primIdx,
                                                               unsigned int           sbtGASIndex,
                                                               float                  time,
                                                               float2                 ribbonParameters );

/// Returns ribbon normal at intersection reported by optixReportIntersection.
///
/// Available in all OptiX program types
static __forceinline__ __device__ float3 optixGetRibbonNormalFromHandle( OptixTraversableHandle gas,
                                                                         unsigned int           primIdx,
                                                                         unsigned int           sbtGASIndex,
                                                                         float                  time,
                                                                         float2                 ribbonParameters );

/// Return ribbon normal at intersection reported by optixReportIntersection.
///
/// Available in AH, CH
static __forceinline__ __device__ float3 optixGetRibbonNormal( float2 ribbonParameters );

/// Return ribbon normal at intersection reported by optixReportIntersection.
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ float3 optixHitObjectGetRibbonNormal( float2 ribbonParameters );

/// Deprecated. Call either optixGetSphereData( float4 data[1] ) for current hit
/// sphere data, or optixGetSphereDataFromHandle() for random access sphere data.
///
/// Returns the object space sphere data, center point and radius, in a Geometry Acceleration
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
static __forceinline__ __device__ void optixGetSphereData( OptixTraversableHandle gas,
                                                           unsigned int           primIdx,
                                                           unsigned int           sbtGASIndex,
                                                           float                  time,
                                                           float4                 data[1] );

/// Performs a random access fetch of the object space sphere data, center point and radius, in a Geometry Acceleration
/// Structure (GAS) at a given motion time.
///
/// To access vertex data of any curve, the GAS must be built using the flag
/// OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS.
/// If only the vertex data of a currently intersected sphere is required, it is recommended to
/// use function optixGetSphereData. A data fetch of the currently hit primitive does NOT
/// require building the corresponding GAS with flag OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS.
///
/// data[0] = {x,y,z,w} with {x,y,z} the position of the sphere center and w the radius.
///
/// If motion is disabled via OptixPipelineCompileOptions::usesMotionBlur, or the GAS does not
/// contain motion, the time parameter is ignored.
///
/// Available in all OptiX program types
static __forceinline__ __device__ void optixGetSphereDataFromHandle( OptixTraversableHandle gas,
                                                                     unsigned int           primIdx,
                                                                     unsigned int           sbtGASIndex,
                                                                     float                  time,
                                                                     float4                 data[1] );

/// Returns the object space sphere data of the currently intersected sphere at the current ray time.
///
/// Similar to the random access variant optixGetSphereDataFromHandle, but does not require setting flag
/// OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS when building the corresponding GAS.
///
/// It is only valid to call this function if the return value of optixGetPrimitiveType( optixGetHitKind() ) equals OPTIX_PRIMITIVE_TYPE_SPHERE.
///
/// Available in AH, CH
static __forceinline__ __device__ void optixGetSphereData( float4 data[1] );

/// Returns the object space sphere data of the currently intersected sphere for a valid outgoing hit object.
/// It is the hit object's pendant of optixGetSphereData( float4 data[1] ).
///
/// It is only valid to call this function if the return value of optixGetPrimitiveType( optixHitObjectGetHitKind() ) equals OPTIX_PRIMITIVE_TYPE_SPHERE.
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ void optixHitObjectGetSphereData( float4 data[1] );

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

/// Returns the world-to-object transformation matrix resulting from the
/// transformation list of the current outgoing hit object.
///
/// The cost of this function may be proportional to the size of the transformation list.
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ void optixHitObjectGetWorldToObjectTransformMatrix( float m[12] );

/// Returns the object-to-world transformation matrix resulting from the
/// transformation list of the current outgoing hit object.
///
/// The cost of this function may be proportional to the size of the transformation list.
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ void optixHitObjectGetObjectToWorldTransformMatrix( float m[12] );

/// Transforms the point using world-to-object transformation matrix resulting from the
/// transformation list of the current outgoing hit object.
///
/// The cost of this function may be proportional to the size of the transformation list.
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ float3 optixHitObjectTransformPointFromWorldToObjectSpace( float3 point );

/// Transforms the vector using world-to-object transformation matrix resulting from the
/// transformation list of the current outgoing hit object.
///
/// The cost of this function may be proportional to the size of the transformation list.
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ float3 optixHitObjectTransformVectorFromWorldToObjectSpace( float3 vec );

/// Transforms the normal using world-to-object transformation matrix resulting from the
/// transformation list of the current outgoing hit object.
///
/// The cost of this function may be proportional to the size of the transformation list.
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ float3 optixHitObjectTransformNormalFromWorldToObjectSpace( float3 normal );

/// Transforms the point using object-to-world transformation matrix resulting from the
/// transformation list of the current outgoing hit object.
///
/// The cost of this function may be proportional to the size of the transformation list.
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ float3 optixHitObjectTransformPointFromObjectToWorldSpace( float3 point );

/// Transforms the vector using object-to-world transformation matrix resulting from the
/// transformation list of the current outgoing hit object.
///
/// The cost of this function may be proportional to the size of the transformation list.
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ float3 optixHitObjectTransformVectorFromObjectToWorldSpace( float3 vec );

/// Transforms the normal using object-to-world transformation matrix resulting from the
/// transformation list of the current outgoing hit object.
///
/// The cost of this function may be proportional to the size of the transformation list.
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ float3 optixHitObjectTransformNormalFromObjectToWorldSpace( float3 normal );

/// Returns the world-to-object transformation matrix resulting from the transformation list of the
/// templated hit object. Users may implement getRayTime, getTransformListSize, and getTransformListHandle
/// in their own structs, or inherit them from Optix[Incoming|Outgoing]HitObject. Here is an example:
///
/// struct FixedTimeHitState : OptixIncomingHitObject {
///   float time;
///   __forceinline__ __device__ float getRayTime() { return time; }
/// };
/// ...
/// optixGetWorldToObjectTransformMatrix( FixedTimeHitState{ 0.4f }, m );
///
/// The cost of this function may be proportional to the size of the transformation list.
///
/// Available in IS, AH, CH
template <typename HitState>
static __forceinline__ __device__ void optixGetWorldToObjectTransformMatrix( const HitState& hs, float m[12] );

/// Returns the object-to-world transformation matrix resulting from the transformation list
/// of the templated hit object (see optixGetWorldToObjectTransformMatrix for example usage).
///
/// The cost of this function may be proportional to the size of the transformation list.
///
/// Available in IS, AH, CH
template <typename HitState>
static __forceinline__ __device__ void optixGetObjectToWorldTransformMatrix( const HitState& hs, float m[12] );

/// Transforms the point using world-to-object transformation matrix resulting from the transformation
/// list of the templated hit object (see optixGetWorldToObjectTransformMatrix for example usage).
///
/// The cost of this function may be proportional to the size of the transformation list.
///
/// Available in IS, AH, CH
template <typename HitState>
static __forceinline__ __device__ float3 optixTransformPointFromWorldToObjectSpace( const HitState& hs, float3 point );

/// Transforms the vector using world-to-object transformation matrix resulting from the transformation
/// list of the templated hit object (see optixGetWorldToObjectTransformMatrix for example usage).
///
/// The cost of this function may be proportional to the size of the transformation list.
///
/// Available in IS, AH, CH
template <typename HitState>
static __forceinline__ __device__ float3 optixTransformVectorFromWorldToObjectSpace( const HitState& hs, float3 vec );

/// Transforms the normal using world-to-object transformation matrix resulting from the transformation
/// list of the templated hit object (see optixGetWorldToObjectTransformMatrix for example usage).
///
/// The cost of this function may be proportional to the size of the transformation list.
///
/// Available in IS, AH, CH
template <typename HitState>
static __forceinline__ __device__ float3 optixTransformNormalFromWorldToObjectSpace( const HitState& hs, float3 normal );

/// Transforms the point using object-to-world transformation matrix resulting from the transformation
/// list of the templated hit object (see optixGetWorldToObjectTransformMatrix for example usage).
///
/// The cost of this function may be proportional to the size of the transformation list.
///
/// Available in IS, AH, CH
template <typename HitState>
static __forceinline__ __device__ float3 optixTransformPointFromObjectToWorldSpace( const HitState& hs, float3 point );

/// Transforms the vector using object-to-world transformation matrix resulting from the transformation
/// list of the templated hit object (see optixGetWorldToObjectTransformMatrix for example usage).
///
/// The cost of this function may be proportional to the size of the transformation list.
///
/// Available in IS, AH, CH
template <typename HitState>
static __forceinline__ __device__ float3 optixTransformVectorFromObjectToWorldSpace( const HitState& hs, float3 vec );

/// Transforms the normal using object-to-world transformation matrix resulting from the transformation
/// list of the templated hit object (see optixGetWorldToObjectTransformMatrix for example usage).
///
/// The cost of this function may be proportional to the size of the transformation list.
///
/// Available in IS, AH, CH
template <typename HitState>
static __forceinline__ __device__ float3 optixTransformNormalFromObjectToWorldSpace( const HitState& hs, float3 normal );

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

struct OptixIncomingHitObject
{
    __forceinline__ __device__ float        getRayTime() const { return optixGetRayTime(); }
    __forceinline__ __device__ unsigned int getTransformListSize() const { return optixGetTransformListSize(); }
    __forceinline__ __device__ OptixTraversableHandle getTransformListHandle( unsigned int index ) const
    {
        return optixGetTransformListHandle( index );
    }
};

struct OptixOutgoingHitObject
{
    __forceinline__ __device__ float        getRayTime() const { return optixHitObjectGetRayTime(); }
    __forceinline__ __device__ unsigned int getTransformListSize() const
    {
        return optixHitObjectGetTransformListSize();
    }
    __forceinline__ __device__ OptixTraversableHandle getTransformListHandle( unsigned int index ) const
    {
        return optixHitObjectGetTransformListHandle( index );
    }
};

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


/// Returns the user-provided cluster ID of the intersected CLAS of a hit.
///
/// Returns OPTIX_CLUSTER_ID_INVALID if the closest (or current) intersection 
/// is not a cluster.
///
/// see also OptixPipelineCompileOptions::allowClusteredGeometry
///
/// Available in AH, CH
static __forceinline__ __device__ unsigned int optixGetClusterId();

/// Returns the user-provided cluster ID associated with the current outgoing hit object.
///
/// Returns OPTIX_CLUSTER_ID_INVALID if the closest intersection is not a cluster,
/// or if the hit object is a miss.
///
/// see also OptixPipelineCompileOptions::allowClusteredGeometry
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ unsigned int optixHitObjectGetClusterId();


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


/// Convenience function that returns the first two attributes as floats.
///
/// When using OptixBuildInputTriangleArray objects, during intersection with a triangle, the barycentric coordinates of the hit
/// are stored into the first two attribute registers.
///
/// Available in AH, CH
static __forceinline__ __device__ float2 optixGetTriangleBarycentrics();

/// Returns the barycentric coordinates of the hit point on an intersected triangle.
///
/// This function is the hit object's equivalent to optixGetTriangleBarycentrics().
/// It is only valid to call this function if the return value of
/// optixGetPrimitiveType( optixHitObjectGetHitKind() ) equals OPTIX_PRIMITIVE_TYPE_TRIANGLE.
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ float2 optixHitObjectGetTriangleBarycentrics();

/// Returns the curve parameter associated with the current intersection when using
/// OptixBuildInputCurveArray objects.
///
/// Available in AH, CH
static __forceinline__ __device__ float optixGetCurveParameter();

/// Returns the curve parameter associated with the intersection of a curve.
///
/// This function is the hit object's equivalent to optixGetCurveParameter().
/// It is only valid to call this function if the return value of
/// optixGetPrimitiveType( optixHitObjectGetHitKind() ) equals a primitive type that can
/// be used to build an AS with OptixBuildInputCurveArray objects.
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ float optixHitObjectGetCurveParameter();

/// Returns the ribbon parameters along directrix (length) and generator (width) of the current
/// intersection when using OptixBuildInputCurveArray objects with curveType
/// OPTIX_PRIMITIVE_TYPE_FLAT_QUADRATIC_BSPLINE.
///
/// Available in AH, CH
static __forceinline__ __device__ float2 optixGetRibbonParameters();

/// Returns the ribbon parameters along directrix (length) and generator (width) of the current
/// curve intersection with primitive type OPTIX_PRIMITIVE_TYPE_FLAT_QUADRATIC_BSPLINE.
///
/// This function is the hit object's equivalent to optixGetRibbonParameters().
/// It is only valid to call this function if the return value of
/// optixGetPrimitiveType( optixHitObjectGetHitKind() ) equals OPTIX_PRIMITIVE_TYPE_FLAT_QUADRATIC_BSPLINE.
///
/// Available in RG, CH, MS, CC, DC
static __forceinline__ __device__ float2 optixHitObjectGetRibbonParameters();

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


// If you manually define OPTIX_INCLUDE_COOPERATIVE_VECTOR to override the default behavior, you must
// set it to 0 or 1 and not simply define it with no value (which will default it have a value of 0).
#ifndef OPTIX_INCLUDE_COOPERATIVE_VECTOR
#  define OPTIX_INCLUDE_COOPERATIVE_VECTOR_UNSET
#  define OPTIX_INCLUDE_COOPERATIVE_VECTOR 1
#endif

#if OPTIX_INCLUDE_COOPERATIVE_VECTOR
/// \addtogroup optix_device_api
/// \defgroup optix_device_api_coop_vec Cooperative Vector
/// \ingroup optix_device_api
///@{
///

/// Load the vector from global memory. The memory address must be 16 byte aligned
/// regardless of the type and number of elements in the vector.
///
/// Available anywhere
template <typename VecTOut>
static __forceinline__ __device__ VecTOut optixCoopVecLoad( CUdeviceptr ptr );
/// Load the vector from global memory. The memory address must be 16 byte aligned
/// regardless of the type and number of elements in the vector.
///
/// Available anywhere
template <typename VecTOut, typename T>
static __forceinline__ __device__ VecTOut optixCoopVecLoad( T* ptr );


/// Following functions are designed to facilitate activation function evaluation between
/// calls to optixCoopVecMatMul. Utilizing only these functions on the activation vectors
/// will typically improve performance.
///
/// Available anywhere
template <typename VecT>
static __forceinline__ __device__ VecT optixCoopVecExp2( const VecT& vec );
///
/// Available anywhere
template <typename VecT>
static __forceinline__ __device__ VecT optixCoopVecLog2( const VecT& vec );
///
/// Available anywhere
template <typename VecT>
static __forceinline__ __device__ VecT optixCoopVecTanh( const VecT& vec );
/// Convert from VecTIn to VecTOut. Not all conversions are supported, only integral to 16
/// or 32-bit floating point.
///
/// Available anywhere
template <typename VecTOut, typename VecTIn>
static __forceinline__ __device__ VecTOut optixCoopVecCvt( const VecTIn& vec );
///
/// Available anywhere
template <typename VecT>
static __forceinline__ __device__ VecT optixCoopVecMin( const VecT& vecA, const VecT& vecB );
///
/// Available anywhere
template <typename VecT>
static __forceinline__ __device__ VecT optixCoopVecMin( const VecT& vecA, typename VecT::value_type B );
///
/// Available anywhere
template <typename VecT>
static __forceinline__ __device__ VecT optixCoopVecMax( const VecT& vecA, const VecT& vecB );
///
/// Available anywhere
template <typename VecT>
static __forceinline__ __device__ VecT optixCoopVecMax( const VecT& vecA, typename VecT::value_type B );
///
/// Available anywhere
template <typename VecT>
static __forceinline__ __device__ VecT optixCoopVecMul( const VecT& vecA, const VecT& vecB );
///
/// Available anywhere
template <typename VecT>
static __forceinline__ __device__ VecT optixCoopVecAdd( const VecT& vecA, const VecT& vecB );
///
/// Available anywhere
template <typename VecT>
static __forceinline__ __device__ VecT optixCoopVecSub( const VecT& vecA, const VecT& vecB );
/// Returns result[i] = ( vecA[i] < vecB[i] ) ? 0 : 1;
///
/// Available anywhere
template <typename VecT>
static __forceinline__ __device__ VecT optixCoopVecStep( const VecT& vecA, const VecT& vecB );
///
/// Available anywhere
template <typename VecT>
static __forceinline__ __device__ VecT optixCoopVecFFMA( const VecT& vecA, const VecT& vecB, const VecT& vecC );

/// Computes a vector matrix multiplication with an addition of a bias.
///
/// \code
///           A * B           + C     = D
/// Does matrix * inputVector + bias  = output
///       [NxK]   [Kx1]         [Nx1] = [Nx1]
/// \endcode
///
/// Not all combinations of inputType and matrixElementType are supported. See the
/// following table for supported configurations.
///
/// inputType  | inputInterpretation | matrixElementType | biasElementType | outputType
/// -----------|---------------------|-------------------|-----------------|-----------
/// FLOAT16    | FLOAT16             | FLOAT16           | FLOAT16         | FLOAT16
/// FLOAT16    | FLOAT8_E4M3         | FLOAT8_E4M3       | FLOAT16         | FLOAT16
/// FLOAT16    | FLOAT8_E5M4         | FLOAT8_E5M4       | FLOAT16         | FLOAT16
/// FLOAT16    | UINT8/INT8          | UINT8/INT8        | UINT32/INT32    | UINT32/INT32
/// FLOAT32    | UINT8/INT8          | UINT8/INT8        | UINT32/INT32    | UINT32/INT32
/// UINT8/INT8 | UINT8/INT8          | UINT8/INT8        | UINT32/INT32    | UINT32/INT32
///
/// If either the input or matrix is signed, then the bias and output must also be signed.
///
/// When matrixElementType is OPTIX_COOP_VEC_ELEM_TYPE_FLOAT8_E4M3 or
/// OPTIX_COOP_VEC_ELEM_TYPE_FLOAT8_E5M2 the matrixLayout must be either
/// OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL or
/// OPTIX_COOP_VEC_MATRIX_LAYOUT_TRAINING_OPTIMAL.
///
/// When the inputVector's element type does not match the inputInterpretation
/// arithmetically casting is performed on the input values to match the
/// inputInterpretation.
///
/// If transpose is true, the matrix is treated as being stored transposed in memory
/// (stored as KxN instead of NxK). Set other parameters as if the matrix was not
/// transposed in memory. Not all matrix element types or matrix layouts support
/// transpose. Only OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16 is supported. Only
/// OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL and
/// OPTIX_COOP_VEC_MATRIX_LAYOUT_TRAINING_OPTIMAL are supported.
///
/// The bias pointer is assumed to not be null and may be dereferenced. If you wish to do
/// the matrix multiply without a bias then use the overloaded version of this function
/// that does not take the bias.
///
/// For row and column ordered matrix layouts, the stride will assume tight packing when
/// rowColumnStrideInBytes is a constant immediate 0 (computed values or loaded from
/// memory will not work). Ignored for other matrix layouts. Value must be 16 byte
/// aligned.
///
/// \tparam VecTOut             Type must match biasElementType and size must match N
/// \tparam VecTIn              Type must be i32, f16 or f32 type and size must match K
/// \tparam inputInterpretation Must match matrixLayout
/// \tparam matrixLayout        The layout of the matrix in memory
/// \tparam transpose           Whether the data in memory for matrix is transposed from the specified layout
/// \tparam N                   Must match VecTOut::size
/// \tparam K                   Must match VecTIn::size
/// \tparam matrixElementType   Type of elements stored in memory
/// \tparam biasElementType     Type of elements stored in memory, must also match VecTOut::elementType
///
/// \param[in] inputVector
/// \param[in] matrix                 pointer to global memory. Array of NxK elements. 64 byte aligned. Must not be modified during use.
/// \param[in] matrixOffsetInBytes    offset to start of matrix data. Using the same value for matrix with different offsets for all layers yields more effecient execution. 64 byte aligned.
/// \param[in] bias                   pointer to global memory. Array of N elements. 16 byte aligned. Must not be modified during use.
/// \param[in] biasOffsetInBytes      offset to start of bias data. Using the same value for bias with different offsets for all layers yields more effecient execution. 16 byte aligned.
/// \param[in] rowColumnStrideInBytes for row or column major matrix layouts, this identifies the stride between columns or rows.
///
/// Available in all OptiX program types
template <
    typename VecTOut,
    typename VecTIn,
    OptixCoopVecElemType inputInterpretation,
    OptixCoopVecMatrixLayout matrixLayout,
    bool transpose,
    unsigned int N,
    unsigned int K,
    OptixCoopVecElemType matrixElementType,
    OptixCoopVecElemType biasElementType>
static __forceinline__ __device__ VecTOut optixCoopVecMatMul( const VecTIn& inputVector,
                                                              CUdeviceptr matrix,  // 64 byte aligned, Array of KxN elements
                                                              unsigned    matrixOffsetInBytes,  // 64 byte aligned
                                                              CUdeviceptr bias,  // 16 byte aligned, Array of N elements
                                                              unsigned    biasOffsetInBytes,  // 16 byte aligned
                                                              unsigned    rowColumnStrideInBytes = 0 );

/// Same as #optixCoopVecMatMul, but without the bias parameters.
template <typename VecTOut, typename VecTIn, OptixCoopVecElemType inputInterpretation, OptixCoopVecMatrixLayout matrixLayout, bool transpose, unsigned int N, unsigned int K, OptixCoopVecElemType matrixElementType>
static __forceinline__ __device__ VecTOut optixCoopVecMatMul( const VecTIn& inputVector,
                                                              CUdeviceptr matrix,  // 64 byte aligned, Array of KxN elements
                                                              unsigned matrixOffsetInBytes,  // 64 byte aligned
                                                              unsigned rowColumnStrideInBytes = 0 );

/// Performs a component-wise atomic add reduction of the vector into global memory
/// starting at \a offsetInBytes bytes after \a outputVector.
///
/// VecTIn::elementType must be of type OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16 or
/// OPTIX_COOP_VEC_ELEM_TYPE_FLOAT32 The memory backed by \a outputVector + \a offsetInBytes
/// must be large enough to accomodate VecTIn::size elements.  The type of data in
/// \a outputVector must match VecTIn::elementType. No type conversion is performed.
/// \a outputVector + \a offsetInBytes must be 4 byte aligned.
///
/// \tparam VecTIn Type of inputVector
///
/// \param[in] inputVector
/// \param[in] outputVector  pointer to global memory on the device, sum with \a offsetInBytes must be a multiple of 4
/// \param[in] offsetInBytes offset in bytes from \a outputVector, sum with \a outputVector must be a multiple of 4
///
/// Available in all OptiX program types
template <typename VecTIn>
static __forceinline__ __device__ void optixCoopVecReduceSumAccumulate( const VecTIn& inputVector,
                                                                        CUdeviceptr   outputVector,
                                                                        unsigned      offsetInBytes );

/// Produces a matrix outer product of the input vecA and vecB ( vecA * transpose(vecB) )
/// and does a component-wise atomic add reduction of the result into global memory
/// starting \a offsetInBytes bytes after \a outputMatrix. The dimentions of the matrix are
/// [VecTA::size, VecTB::size]. VecTA::elementType, VecTB::elementType and the element
/// type of the matrix must be the same, no type conversion is performed. The element type
/// must be OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16.
///
/// outputMatrix + offsetInBytes must be 4B aligned, but performance may be better with
/// 128 byte alignments.
///
/// The output matrix will be in matrixLayout layout, though currently only
/// OPTIX_COOP_VEC_MATRIX_LAYOUT_TRAINING_OPTIMAL layout is supported.
///
/// \tparam VecTA        Type of vecA
/// \tparam VecTB        Type of vecB
/// \tparam matrixLayout Layout of matrix stored in outputMatrix
///
/// \param [in] vecA
/// \param [in] vecB
/// \param [in] outputMatrix           pointer to global memory on the device, sum with \a offsetInBytes must be a multiple of 4
/// \param [in] offsetInBytes          offset in bytes from \a outputMatrix, sum with \a outputMatrix must be a multiple of 4
/// \param [in] rowColumnStrideInBytes stride between rows or columns, zero takes natural stride, ignored for optimal layouts
///
/// Available in all OptiX program types
template <typename VecTA, typename VecTB, OptixCoopVecMatrixLayout matrixLayout = OPTIX_COOP_VEC_MATRIX_LAYOUT_TRAINING_OPTIMAL>
static __forceinline__ __device__ void optixCoopVecOuterProductAccumulate( const VecTA& vecA,
                                                                           const VecTB& vecB,
                                                                           CUdeviceptr  outputMatrix,
                                                                           unsigned     offsetInBytes,
                                                                           unsigned     rowColumnStrideInBytes = 0 );

/// This function is intended strictly for matrix layouts that must be computed through
/// the host API, #optixCoopVecMatrixComputeSize, but is needed on the device. For optimal
/// performance the offsets to each layer in a network should be constant, so this
/// function can be used to facilitate calculating the offset for subsequent layers in
/// shader code. It can also be used for calculating the size of row and column major
/// matrices, but the rowColumnStrideInBytes template parameter must be specified, so that
/// it can be calculated during compilation.
///
/// For row and column ordered matrix layouts, when rowColumnStrideInBytes is 0, the
/// stride will assume tight packing.
///
/// Results will be rounded to the next multiple of 64 to make it easy to pack the
/// matrices in memory and have the correct alignment.
///
/// Results are in number of bytes, and should match the output of the host function
/// #optixCoopVecMatrixComputeSize.
///
/// \tparam N, K        dimensions of the matrix
/// \tparam elementType Type of the matrix elements
/// \tparam layout      Layout of the matrix
///
/// Available anywhere
template <unsigned int N, unsigned int K, OptixCoopVecElemType elementType, OptixCoopVecMatrixLayout layout = OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL, unsigned int rowColumnStrideInBytes = 0>
static __forceinline__ __device__ unsigned int optixCoopVecGetMatrixSize();

/// The API does not require the use of this class specifically, but it must define a
/// certain interface as spelled out by the public members of the class. Note that not all
/// types of T are supported. Only 8 and 32 bit signed and unsigned integral types along
/// with 16 and 32 bit floating point values.
template <typename T, unsigned int N>
class OptixCoopVec
{
  public:
    static const unsigned int size = N;
    using value_type               = T;

    __forceinline__ __device__ OptixCoopVec() {}
    __forceinline__ __device__ OptixCoopVec( const value_type& val )
    {
        for( unsigned int i = 0; i < size; ++i )
            m_data[i]       = val;
    }
    __forceinline__ __device__ const value_type& operator[]( unsigned int index ) const { return m_data[index]; }
    __forceinline__ __device__ value_type& operator[]( unsigned int index ) { return m_data[index]; }

    __forceinline__ __device__ const value_type* data() const { return m_data; }
    __forceinline__ __device__ value_type* data() { return m_data; }

  protected:
    value_type m_data[size];
};

/**@}*/  // end group optix_device_api

#include "internal/optix_device_impl_coop_vec.h"

#endif //  OPTIX_INCLUDE_COOPERATIVE_VECTOR

#ifdef OPTIX_INCLUDE_COOPERATIVE_VECTOR_UNSET
#  undef OPTIX_INCLUDE_COOPERATIVE_VECTOR
#  undef OPTIX_INCLUDE_COOPERATIVE_VECTOR_UNSET
#endif


#endif  // OPTIX_OPTIX_DEVICE_H
