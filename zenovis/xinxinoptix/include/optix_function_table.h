/* 
* SPDX-FileCopyrightText: Copyright (c) 2019 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
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

#ifndef OPTIX_OPTIX_FUNCTION_TABLE_H
#define OPTIX_OPTIX_FUNCTION_TABLE_H

/// The OptiX ABI version.
#define OPTIX_ABI_VERSION 93

#ifndef OPTIX_DEFINE_ABI_VERSION_ONLY

#include "optix_types.h"

#if !defined( OPTIX_DONT_INCLUDE_CUDA )
// If OPTIX_DONT_INCLUDE_CUDA is defined, cuda driver types must be defined through other
// means before including optix headers.
#include <cuda.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/// \defgroup optix_function_table Function Table
/// \brief OptiX Function Table

/** \addtogroup optix_function_table
@{
*/

/// The function table containing all API functions.
///
/// See #optixInit() and #optixInitWithHandle().
typedef struct OptixFunctionTable
{
    /// \name Error handling
    //@ {

    /// See ::optixGetErrorName().
    const char* ( *optixGetErrorName )( OptixResult result );

    /// See ::optixGetErrorString().
    const char* ( *optixGetErrorString )( OptixResult result );

    //@ }
    /// \name Device context
    //@ {

    /// See ::optixDeviceContextCreate().
    OptixResult ( *optixDeviceContextCreate )( CUcontext fromContext, const OptixDeviceContextOptions* options, OptixDeviceContext* context );

    /// See ::optixDeviceContextDestroy().
    OptixResult ( *optixDeviceContextDestroy )( OptixDeviceContext context );

    /// See ::optixDeviceContextGetProperty().
    OptixResult ( *optixDeviceContextGetProperty )( OptixDeviceContext context, OptixDeviceProperty property, void* value, size_t sizeInBytes );

    /// See ::optixDeviceContextSetLogCallback().
    OptixResult ( *optixDeviceContextSetLogCallback )( OptixDeviceContext context,
                                                       OptixLogCallback   callbackFunction,
                                                       void*              callbackData,
                                                       unsigned int       callbackLevel );

    /// See ::optixDeviceContextSetCacheEnabled().
    OptixResult ( *optixDeviceContextSetCacheEnabled )( OptixDeviceContext context, int enabled );

    /// See ::optixDeviceContextSetCacheLocation().
    OptixResult ( *optixDeviceContextSetCacheLocation )( OptixDeviceContext context, const char* location );

    /// See ::optixDeviceContextSetCacheDatabaseSizes().
    OptixResult ( *optixDeviceContextSetCacheDatabaseSizes )( OptixDeviceContext context, size_t lowWaterMark, size_t highWaterMark );

    /// See ::optixDeviceContextGetCacheEnabled().
    OptixResult ( *optixDeviceContextGetCacheEnabled )( OptixDeviceContext context, int* enabled );

    /// See ::optixDeviceContextGetCacheLocation().
    OptixResult ( *optixDeviceContextGetCacheLocation )( OptixDeviceContext context, char* location, size_t locationSize );

    /// See ::optixDeviceContextGetCacheDatabaseSizes().
    OptixResult ( *optixDeviceContextGetCacheDatabaseSizes )( OptixDeviceContext context, size_t* lowWaterMark, size_t* highWaterMark );

    //@ }
    /// \name Modules
    //@ {

    /// See ::optixModuleCreate().
    OptixResult ( *optixModuleCreate )( OptixDeviceContext                 context,
                                        const OptixModuleCompileOptions*   moduleCompileOptions,
                                        const OptixPipelineCompileOptions* pipelineCompileOptions,
                                        const char*                        input,
                                        size_t                             inputSize,
                                        char*                              logString,
                                        size_t*                            logStringSize,
                                        OptixModule*                       module );

    /// See ::optixModuleCreateWithTasks().
    OptixResult ( *optixModuleCreateWithTasks )( OptixDeviceContext                 context,
                                                 const OptixModuleCompileOptions*   moduleCompileOptions,
                                                 const OptixPipelineCompileOptions* pipelineCompileOptions,
                                                 const char*                        input,
                                                 size_t                             inputSize,
                                                 char*                              logString,
                                                 size_t*                            logStringSize,
                                                 OptixModule*                       module,
                                                 OptixTask*                         firstTask );

    /// See ::optixModuleGetCompilationState().
    OptixResult ( *optixModuleGetCompilationState )( OptixModule module, OptixModuleCompileState* state );

    /// See ::optixModuleDestroy().
    OptixResult ( *optixModuleDestroy )( OptixModule module );

    /// See ::optixBuiltinISModuleGet().
    OptixResult( *optixBuiltinISModuleGet )( OptixDeviceContext                 context,
                                             const OptixModuleCompileOptions*   moduleCompileOptions,
                                             const OptixPipelineCompileOptions* pipelineCompileOptions,
                                             const OptixBuiltinISOptions*       builtinISOptions,
                                             OptixModule*                       builtinModule);

    //@ }
    /// \name Tasks
    //@ {

    /// See ::optixTaskExecute().
    OptixResult ( *optixTaskExecute )( OptixTask     task,
                                       OptixTask*    additionalTasks,
                                       unsigned int  maxNumAdditionalTasks,
                                       unsigned int* numAdditionalTasksCreated );
    //@ }
    /// \name Program groups
    //@ {

    /// See ::optixProgramGroupCreate().
    OptixResult ( *optixProgramGroupCreate )( OptixDeviceContext              context,
                                              const OptixProgramGroupDesc*    programDescriptions,
                                              unsigned int                    numProgramGroups,
                                              const OptixProgramGroupOptions* options,
                                              char*                           logString,
                                              size_t*                         logStringSize,
                                              OptixProgramGroup*              programGroups );

    /// See ::optixProgramGroupDestroy().
    OptixResult ( *optixProgramGroupDestroy )( OptixProgramGroup programGroup );

    /// See ::optixProgramGroupGetStackSize().
    OptixResult ( *optixProgramGroupGetStackSize )( OptixProgramGroup programGroup, OptixStackSizes* stackSizes, OptixPipeline pipeline );

    //@ }
    /// \name Pipeline
    //@ {

    /// See ::optixPipelineCreate().
    OptixResult ( *optixPipelineCreate )( OptixDeviceContext                 context,
                                          const OptixPipelineCompileOptions* pipelineCompileOptions,
                                          const OptixPipelineLinkOptions*    pipelineLinkOptions,
                                          const OptixProgramGroup*           programGroups,
                                          unsigned int                       numProgramGroups,
                                          char*                              logString,
                                          size_t*                            logStringSize,
                                          OptixPipeline*                     pipeline );

    /// See ::optixPipelineDestroy().
    OptixResult ( *optixPipelineDestroy )( OptixPipeline pipeline );

    /// See ::optixPipelineSetStackSize().
    OptixResult ( *optixPipelineSetStackSize )( OptixPipeline pipeline,
                                                unsigned int  directCallableStackSizeFromTraversal,
                                                unsigned int  directCallableStackSizeFromState,
                                                unsigned int  continuationStackSize,
                                                unsigned int  maxTraversableGraphDepth );

    //@ }
    /// \name Acceleration structures
    //@ {

    /// See ::optixAccelComputeMemoryUsage().
    OptixResult ( *optixAccelComputeMemoryUsage )( OptixDeviceContext            context,
                                                   const OptixAccelBuildOptions* accelOptions,
                                                   const OptixBuildInput*        buildInputs,
                                                   unsigned int                  numBuildInputs,
                                                   OptixAccelBufferSizes*        bufferSizes );

    /// See ::optixAccelBuild().
    OptixResult ( *optixAccelBuild )( OptixDeviceContext            context,
                                      CUstream                      stream,
                                      const OptixAccelBuildOptions* accelOptions,
                                      const OptixBuildInput*        buildInputs,
                                      unsigned int                  numBuildInputs,
                                      CUdeviceptr                   tempBuffer,
                                      size_t                        tempBufferSizeInBytes,
                                      CUdeviceptr                   outputBuffer,
                                      size_t                        outputBufferSizeInBytes,
                                      OptixTraversableHandle*       outputHandle,
                                      const OptixAccelEmitDesc*     emittedProperties,
                                      unsigned int                  numEmittedProperties );

    /// See ::optixAccelGetRelocationInfo().
    OptixResult ( *optixAccelGetRelocationInfo )( OptixDeviceContext context, OptixTraversableHandle handle, OptixRelocationInfo* info );


    /// See ::optixCheckRelocationCompatibility().
    OptixResult ( *optixCheckRelocationCompatibility )( OptixDeviceContext         context,
                                                        const OptixRelocationInfo* info,
                                                        int*                       compatible );

    /// See ::optixAccelRelocate().
    OptixResult ( *optixAccelRelocate )( OptixDeviceContext         context,
                                         CUstream                   stream,
                                         const OptixRelocationInfo* info,
                                         const OptixRelocateInput*  relocateInputs,
                                         size_t                     numRelocateInputs,
                                         CUdeviceptr                targetAccel,
                                         size_t                     targetAccelSizeInBytes,
                                         OptixTraversableHandle*    targetHandle );


    /// See ::optixAccelCompact().
    OptixResult ( *optixAccelCompact )( OptixDeviceContext      context,
                                        CUstream                stream,
                                        OptixTraversableHandle  inputHandle,
                                        CUdeviceptr             outputBuffer,
                                        size_t                  outputBufferSizeInBytes,
                                        OptixTraversableHandle* outputHandle );

    OptixResult ( *optixAccelEmitProperty )( OptixDeviceContext        context,
                                             CUstream                  stream,
                                             OptixTraversableHandle    handle,
                                             const OptixAccelEmitDesc* emittedProperty );

    /// See ::optixConvertPointerToTraversableHandle().
    OptixResult ( *optixConvertPointerToTraversableHandle )( OptixDeviceContext      onDevice,
                                                             CUdeviceptr             pointer,
                                                             OptixTraversableType    traversableType,
                                                             OptixTraversableHandle* traversableHandle );

    /// See ::optixOpacityMicromapArrayComputeMemoryUsage().
    OptixResult ( *optixOpacityMicromapArrayComputeMemoryUsage )( OptixDeviceContext                         context,
                                                                  const OptixOpacityMicromapArrayBuildInput* buildInput,
                                                                  OptixMicromapBufferSizes*                 bufferSizes );

    /// See ::optixOpacityMicromapArrayBuild().
    OptixResult ( *optixOpacityMicromapArrayBuild )( OptixDeviceContext                         context,
                                                     CUstream                                   stream,
                                                     const OptixOpacityMicromapArrayBuildInput* buildInput,
                                                     const OptixMicromapBuffers*               buffers );

    /// See ::optixOpacityMicromapArrayGetRelocationInfo().
    OptixResult ( *optixOpacityMicromapArrayGetRelocationInfo )( OptixDeviceContext   context,
                                                                 CUdeviceptr          opacityMicromapArray,
                                                                 OptixRelocationInfo* info );

    /// See ::optixOpacityMicromapArrayRelocate().
    OptixResult ( *optixOpacityMicromapArrayRelocate )( OptixDeviceContext         context,
                                                        CUstream                   stream,
                                                        const OptixRelocationInfo* info,
                                                        CUdeviceptr                targetOpacityMicromapArray,
                                                        size_t                     targetOpacityMicromapArraySizeInBytes );

    /// See ::optixDisplacementMicromapArrayComputeMemoryUsage().
    OptixResult ( *optixDisplacementMicromapArrayComputeMemoryUsage )( OptixDeviceContext context,
                                                                       const OptixDisplacementMicromapArrayBuildInput* buildInput,
                                                                       OptixMicromapBufferSizes* bufferSizes );

    /// See ::optixDisplacementMicromapArrayBuild().
    OptixResult ( *optixDisplacementMicromapArrayBuild )( OptixDeviceContext                              context,
                                                          CUstream                                        stream,
                                                          const OptixDisplacementMicromapArrayBuildInput* buildInput,
                                                          const OptixMicromapBuffers*                     buffers );

    //@ }
    /// \name Launch
    //@ {

    /// See ::optixConvertPointerToTraversableHandle().
    OptixResult ( *optixSbtRecordPackHeader )( OptixProgramGroup programGroup, void* sbtRecordHeaderHostPointer );

    /// See ::optixConvertPointerToTraversableHandle().
    OptixResult ( *optixLaunch )( OptixPipeline                  pipeline,
                                  CUstream                       stream,
                                  CUdeviceptr                    pipelineParams,
                                  size_t                         pipelineParamsSize,
                                  const OptixShaderBindingTable* sbt,
                                  unsigned int                   width,
                                  unsigned int                   height,
                                  unsigned int                   depth );

    OptixResult ( *optixPlaceholder001 )( OptixDeviceContext context );
    OptixResult ( *optixPlaceholder002 )( OptixDeviceContext context );

    //@ }
    /// \name Denoiser
    //@ {

    /// See ::optixDenoiserCreate().
    OptixResult ( *optixDenoiserCreate )( OptixDeviceContext context, OptixDenoiserModelKind modelKind, const OptixDenoiserOptions* options, OptixDenoiser* returnHandle );

    /// See ::optixDenoiserDestroy().
    OptixResult ( *optixDenoiserDestroy )( OptixDenoiser handle );

    /// See ::optixDenoiserComputeMemoryResources().
    OptixResult ( *optixDenoiserComputeMemoryResources )( const OptixDenoiser handle,
                                                          unsigned int        maximumInputWidth,
                                                          unsigned int        maximumInputHeight,
                                                          OptixDenoiserSizes* returnSizes );

    /// See ::optixDenoiserSetup().
    OptixResult ( *optixDenoiserSetup )( OptixDenoiser denoiser,
                                         CUstream      stream,
                                         unsigned int  inputWidth,
                                         unsigned int  inputHeight,
                                         CUdeviceptr   state,
                                         size_t        stateSizeInBytes,
                                         CUdeviceptr   scratch,
                                         size_t        scratchSizeInBytes );

    /// See ::optixDenoiserInvoke().
    OptixResult ( *optixDenoiserInvoke )( OptixDenoiser                   denoiser,
                                          CUstream                        stream,
                                          const OptixDenoiserParams*      params,
                                          CUdeviceptr                     denoiserState,
                                          size_t                          denoiserStateSizeInBytes,
                                          const OptixDenoiserGuideLayer * guideLayer,
                                          const OptixDenoiserLayer *      layers,
                                          unsigned int                    numLayers,
                                          unsigned int                    inputOffsetX,
                                          unsigned int                    inputOffsetY,
                                          CUdeviceptr                     scratch,
                                          size_t                          scratchSizeInBytes );

    /// See ::optixDenoiserComputeIntensity().
    OptixResult ( *optixDenoiserComputeIntensity )( OptixDenoiser       handle,
                                                    CUstream            stream,
                                                    const OptixImage2D* inputImage,
                                                    CUdeviceptr         outputIntensity,
                                                    CUdeviceptr         scratch,
                                                    size_t              scratchSizeInBytes );

    /// See ::optixDenoiserComputeAverageColor().
    OptixResult ( *optixDenoiserComputeAverageColor )( OptixDenoiser       handle,
                                                       CUstream            stream,
                                                       const OptixImage2D* inputImage,
                                                       CUdeviceptr         outputAverageColor,
                                                       CUdeviceptr         scratch,
                                                       size_t              scratchSizeInBytes );

    /// See ::optixDenoiserCreateWithUserModel().
    OptixResult ( *optixDenoiserCreateWithUserModel )( OptixDeviceContext context, const void * data, size_t dataSizeInBytes, OptixDenoiser* returnHandle );
    //@ }

} OptixFunctionTable;

// define global function table variable with ABI specific name.
#define OPTIX_CONCATENATE_ABI_VERSION(prefix, macro) OPTIX_CONCATENATE_ABI_VERSION_IMPL(prefix, macro)
#define OPTIX_CONCATENATE_ABI_VERSION_IMPL(prefix, macro) prefix ## _ ## macro
#define OPTIX_FUNCTION_TABLE_SYMBOL OPTIX_CONCATENATE_ABI_VERSION(g_optixFunctionTable, OPTIX_ABI_VERSION)

/**@}*/  // end group optix_function_table

#ifdef __cplusplus
}
#endif

#endif /* OPTIX_DEFINE_ABI_VERSION_ONLY */

#endif /* OPTIX_OPTIX_FUNCTION_TABLE_H */
