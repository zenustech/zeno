/*
 * SPDX-FileCopyrightText: Copyright (c) 2019 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/// @file
/// @author NVIDIA Corporation
/// @brief  OptiX public API header

#ifndef OPTIX_OPTIX_STUBS_H
#define OPTIX_OPTIX_STUBS_H

#include "optix_function_table.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#include <windows.h>
// The cfgmgr32 header is necessary for interrogating driver information in the registry.
// For convenience the library is also linked in automatically using the #pragma command.
#include <cfgmgr32.h>
#pragma comment( lib, "Cfgmgr32.lib" )
#include <string.h>
#else
#include <dlfcn.h>
#endif

/// Mixing multiple SDKs in a single application will result in symbol collisions.
/// To enable different compilation units to use different SDKs, use OPTIX_ENABLE_SDK_MIXING.
#ifndef OPTIXAPI
# ifdef OPTIX_ENABLE_SDK_MIXING
#   define OPTIXAPI static
# else  // OPTIX_ENABLE_SDK_MIXING
#   ifdef __cplusplus
#     define OPTIXAPI extern "C"
#   else  // __cplusplus
#     define OPTIXAPI
#   endif  // __cplusplus
# endif  // OPTIX_ENABLE_SDK_MIXING
#endif  // OPTIXAPI

#ifdef __cplusplus
extern "C" {
#endif

// The function table needs to be defined in exactly one translation unit. This can be
// achieved by including optix_function_table_definition.h in that translation unit.
extern OptixFunctionTable OPTIX_FUNCTION_TABLE_SYMBOL;

#ifdef __cplusplus
}
#endif

#ifdef _WIN32
#if defined( _MSC_VER )
// Visual Studio produces warnings suggesting strcpy and friends being replaced with _s
// variants. All the string lengths and allocation sizes have been calculated and should
// be safe, so we are disabling this warning to increase compatibility.
#pragma warning( push )
#pragma warning( disable : 4996 )
#endif
static void* optixLoadWindowsDllFromName( const char* optixDllName )
{
    void* handle = NULL;


    // Get the size of the path first, then allocate
    unsigned int size = GetSystemDirectoryA( NULL, 0 );
    if( size == 0 )
    {
        // Couldn't get the system path size, so bail
        return NULL;
    }
    size_t pathSize   = size + 1 + strlen( optixDllName );
    char*  systemPath = (char*)malloc( pathSize );
    if( systemPath == NULL )
        return NULL;
    if( GetSystemDirectoryA( systemPath, size ) != size - 1 )
    {
        // Something went wrong
        free( systemPath );
        return NULL;
    }
    strcat( systemPath, "\\" );
    strcat( systemPath, optixDllName );
    handle = LoadLibraryA( systemPath );
    free( systemPath );
    if( handle )
        return handle;

    // If we didn't find it, go looking in the register store.  Since nvoptix.dll doesn't
    // have its own registry entry, we are going to look for the opengl driver which lives
    // next to nvoptix.dll.  0 (null) will be returned if any errors occured.

    static const char* deviceInstanceIdentifiersGUID = "{4d36e968-e325-11ce-bfc1-08002be10318}";
    const ULONG        flags                         = CM_GETIDLIST_FILTER_CLASS | CM_GETIDLIST_FILTER_PRESENT;
    ULONG              deviceListSize                = 0;
    if( CM_Get_Device_ID_List_SizeA( &deviceListSize, deviceInstanceIdentifiersGUID, flags ) != CR_SUCCESS )
    {
        return NULL;
    }
    char* deviceNames = (char*)malloc( deviceListSize );
    if( deviceNames == NULL )
        return NULL;
    if( CM_Get_Device_ID_ListA( deviceInstanceIdentifiersGUID, deviceNames, deviceListSize, flags ) )
    {
        free( deviceNames );
        return NULL;
    }
    DEVINST devID   = 0;
    char*   dllPath = NULL;

    // Continue to the next device if errors are encountered.
    for( char* deviceName = deviceNames; *deviceName; deviceName += strlen( deviceName ) + 1 )
    {
        if( CM_Locate_DevNodeA( &devID, deviceName, CM_LOCATE_DEVNODE_NORMAL ) != CR_SUCCESS )
        {
            continue;
        }
        HKEY regKey = 0;
        if( CM_Open_DevNode_Key( devID, KEY_QUERY_VALUE, 0, RegDisposition_OpenExisting, &regKey, CM_REGISTRY_SOFTWARE ) != CR_SUCCESS )
        {
            continue;
        }
        const char* valueName = "OpenGLDriverName";
        DWORD       valueSize = 0;
        LSTATUS     ret       = RegQueryValueExA( regKey, valueName, NULL, NULL, NULL, &valueSize );
        if( ret != ERROR_SUCCESS )
        {
            RegCloseKey( regKey );
            continue;
        }
        char* regValue = (char*)malloc( valueSize );
        if( regValue == NULL )
        {
            RegCloseKey( regKey );
            continue;
        }
        ret = RegQueryValueExA( regKey, valueName, NULL, NULL, (LPBYTE)regValue, &valueSize );
        if( ret != ERROR_SUCCESS )
        {
            free( regValue );
            RegCloseKey( regKey );
            continue;
        }
        // Strip the opengl driver dll name from the string then create a new string with
        // the path and the nvoptix.dll name
        for( int i = (int)valueSize - 1; i >= 0 && regValue[i] != '\\'; --i )
            regValue[i] = '\0';
        size_t newPathSize = strlen( regValue ) + strlen( optixDllName ) + 1;
        dllPath            = (char*)malloc( newPathSize );
        if( dllPath == NULL )
        {
            free( regValue );
            RegCloseKey( regKey );
            continue;
        }
        strcpy( dllPath, regValue );
        strcat( dllPath, optixDllName );
        free( regValue );
        RegCloseKey( regKey );
        handle = LoadLibraryA( (LPCSTR)dllPath );
        free( dllPath );
        if( handle )
            break;
    }
    free( deviceNames );
    return handle;
}
#if defined( _MSC_VER )
#pragma warning( pop )
#endif

static void* optixLoadWindowsDll()
{
    return optixLoadWindowsDllFromName( "nvoptix.dll" );
}
#endif

/// \defgroup optix_utilities Utilities
/// \brief OptiX Utilities

/** \addtogroup optix_utilities
@{
*/

/// Loads the OptiX library and initializes the function table used by the stubs below.
///
/// If handlePtr is not nullptr, an OS-specific handle to the library will be returned in *handlePtr.
///
/// \see #optixUninitWithHandle
OPTIXAPI inline OptixResult optixInitWithHandle( void** handlePtr )
{
    // Make sure these functions get initialized to zero in case the DLL and function
    // table can't be loaded
    OPTIX_FUNCTION_TABLE_SYMBOL.optixGetErrorName   = 0;
    OPTIX_FUNCTION_TABLE_SYMBOL.optixGetErrorString = 0;

    if( !handlePtr )
        return OPTIX_ERROR_INVALID_VALUE;

#ifdef _WIN32
    *handlePtr = optixLoadWindowsDll();
    if( !*handlePtr )
        return OPTIX_ERROR_LIBRARY_NOT_FOUND;

    void* symbol = (void*)GetProcAddress( (HMODULE)*handlePtr, "optixQueryFunctionTable" );
    if( !symbol )
        return OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND;
#else
    *handlePtr = dlopen( "libnvoptix.so.1", RTLD_NOW );
    if( !*handlePtr )
        return OPTIX_ERROR_LIBRARY_NOT_FOUND;

    void* symbol = dlsym( *handlePtr, "optixQueryFunctionTable" );
    if( !symbol )
        return OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND;
#endif

    OptixQueryFunctionTable_t* optixQueryFunctionTable = (OptixQueryFunctionTable_t*)symbol;

    return optixQueryFunctionTable( OPTIX_ABI_VERSION, 0, 0, 0, &OPTIX_FUNCTION_TABLE_SYMBOL, sizeof( OPTIX_FUNCTION_TABLE_SYMBOL ) );
}

/// Loads the OptiX library and initializes the function table used by the stubs below.
///
/// A variant of #optixInitWithHandle() that does not make the handle to the loaded library available.
OPTIXAPI inline OptixResult optixInit( void )
{
    void* handle;
    return optixInitWithHandle( &handle );
}

/// Unloads the OptiX library and zeros the function table used by the stubs below.  Takes the
/// handle returned by optixInitWithHandle.  All OptixDeviceContext objects must be destroyed
/// before calling this function, or the behavior is undefined.
///
/// \see #optixInitWithHandle
OPTIXAPI inline OptixResult optixUninitWithHandle( void* handle )
{
    if( !handle )
        return OPTIX_ERROR_INVALID_VALUE;
#ifdef _WIN32
    if( !FreeLibrary( (HMODULE)handle ) )
        return OPTIX_ERROR_LIBRARY_UNLOAD_FAILURE;
#else
    if( dlclose( handle ) )
        return OPTIX_ERROR_LIBRARY_UNLOAD_FAILURE;
#endif
    OptixFunctionTable empty
#ifdef __cplusplus
      {}
#else
        = { 0 }
#endif
        ;
    OPTIX_FUNCTION_TABLE_SYMBOL = empty;
    return OPTIX_SUCCESS;
}


/**@}*/  // end group optix_utilities

#ifndef OPTIX_DOXYGEN_SHOULD_SKIP_THIS

// Stub functions that forward calls to the corresponding function pointer in the function table.

OPTIXAPI inline const char* optixGetErrorName( OptixResult result )
{
    if( OPTIX_FUNCTION_TABLE_SYMBOL.optixGetErrorName )
        return OPTIX_FUNCTION_TABLE_SYMBOL.optixGetErrorName( result );

    // If the DLL and symbol table couldn't be loaded, provide a set of error strings
    // suitable for processing errors related to the DLL loading.
    switch( result )
    {
        case OPTIX_SUCCESS:
            return "OPTIX_SUCCESS";
        case OPTIX_ERROR_INVALID_VALUE:
            return "OPTIX_ERROR_INVALID_VALUE";
        case OPTIX_ERROR_UNSUPPORTED_ABI_VERSION:
            return "OPTIX_ERROR_UNSUPPORTED_ABI_VERSION";
        case OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH:
            return "OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH";
        case OPTIX_ERROR_INVALID_ENTRY_FUNCTION_OPTIONS:
            return "OPTIX_ERROR_INVALID_ENTRY_FUNCTION_OPTIONS";
        case OPTIX_ERROR_LIBRARY_NOT_FOUND:
            return "OPTIX_ERROR_LIBRARY_NOT_FOUND";
        case OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND:
            return "OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND";
        case OPTIX_ERROR_LIBRARY_UNLOAD_FAILURE:
            return "OPTIX_ERROR_LIBRARY_UNLOAD_FAILURE";
        default:
            return "Unknown OptixResult code";
    }
}

OPTIXAPI inline const char* optixGetErrorString( OptixResult result )
{
    if( OPTIX_FUNCTION_TABLE_SYMBOL.optixGetErrorString )
        return OPTIX_FUNCTION_TABLE_SYMBOL.optixGetErrorString( result );

    // If the DLL and symbol table couldn't be loaded, provide a set of error strings
    // suitable for processing errors related to the DLL loading.
    switch( result )
    {
        case OPTIX_SUCCESS:
            return "Success";
        case OPTIX_ERROR_INVALID_VALUE:
            return "Invalid value";
        case OPTIX_ERROR_UNSUPPORTED_ABI_VERSION:
            return "Unsupported ABI version";
        case OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH:
            return "Function table size mismatch";
        case OPTIX_ERROR_INVALID_ENTRY_FUNCTION_OPTIONS:
            return "Invalid options to entry function";
        case OPTIX_ERROR_LIBRARY_NOT_FOUND:
            return "Library not found";
        case OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND:
            return "Entry symbol not found";
        case OPTIX_ERROR_LIBRARY_UNLOAD_FAILURE:
            return "Library could not be unloaded";
        default:
            return "Unknown OptixResult code";
    }
}

OPTIXAPI inline OptixResult optixDeviceContextCreate( CUcontext fromContext, const OptixDeviceContextOptions* options, OptixDeviceContext* context )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixDeviceContextCreate( fromContext, options, context );
}

OPTIXAPI inline OptixResult optixDeviceContextDestroy( OptixDeviceContext context )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixDeviceContextDestroy( context );
}

OPTIXAPI inline OptixResult optixDeviceContextGetProperty( OptixDeviceContext context, OptixDeviceProperty property, void* value, size_t sizeInBytes )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixDeviceContextGetProperty( context, property, value, sizeInBytes );
}

OPTIXAPI inline OptixResult optixDeviceContextSetLogCallback( OptixDeviceContext context,
                                                              OptixLogCallback   callbackFunction,
                                                              void*              callbackData,
                                                              unsigned int       callbackLevel )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixDeviceContextSetLogCallback( context, callbackFunction, callbackData, callbackLevel );
}

OPTIXAPI inline OptixResult optixDeviceContextSetCacheEnabled( OptixDeviceContext context, int enabled )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixDeviceContextSetCacheEnabled( context, enabled );
}

OPTIXAPI inline OptixResult optixDeviceContextSetCacheLocation( OptixDeviceContext context, const char* location )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixDeviceContextSetCacheLocation( context, location );
}

OPTIXAPI inline OptixResult optixDeviceContextSetCacheDatabaseSizes( OptixDeviceContext context, size_t lowWaterMark, size_t highWaterMark )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixDeviceContextSetCacheDatabaseSizes( context, lowWaterMark, highWaterMark );
}

OPTIXAPI inline OptixResult optixDeviceContextGetCacheEnabled( OptixDeviceContext context, int* enabled )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixDeviceContextGetCacheEnabled( context, enabled );
}

OPTIXAPI inline OptixResult optixDeviceContextGetCacheLocation( OptixDeviceContext context, char* location, size_t locationSize )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixDeviceContextGetCacheLocation( context, location, locationSize );
}

OPTIXAPI inline OptixResult optixDeviceContextGetCacheDatabaseSizes( OptixDeviceContext context, size_t* lowWaterMark, size_t* highWaterMark )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixDeviceContextGetCacheDatabaseSizes( context, lowWaterMark, highWaterMark );
}

OPTIXAPI inline OptixResult optixModuleCreate( OptixDeviceContext                 context,
                                               const OptixModuleCompileOptions*   moduleCompileOptions,
                                               const OptixPipelineCompileOptions* pipelineCompileOptions,
                                               const char*                        input,
                                               size_t                             inputSize,
                                               char*                              logString,
                                               size_t*                            logStringSize,
                                               OptixModule*                       module )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixModuleCreate( context, moduleCompileOptions, pipelineCompileOptions, input,
                                                          inputSize, logString, logStringSize, module );
}

OPTIXAPI inline OptixResult optixModuleCreateWithTasks( OptixDeviceContext                 context,
                                                        const OptixModuleCompileOptions*   moduleCompileOptions,
                                                        const OptixPipelineCompileOptions* pipelineCompileOptions,
                                                        const char*                        input,
                                                        size_t                             inputSize,
                                                        char*                              logString,
                                                        size_t*                            logStringSize,
                                                        OptixModule*                       module,
                                                        OptixTask*                         firstTask )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixModuleCreateWithTasks( context, moduleCompileOptions, pipelineCompileOptions, input,
                                                                   inputSize, logString, logStringSize, module, firstTask );
}

OPTIXAPI inline OptixResult optixModuleGetCompilationState( OptixModule module, OptixModuleCompileState* state )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixModuleGetCompilationState( module, state );
}

OPTIXAPI inline OptixResult optixModuleDestroy( OptixModule module )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixModuleDestroy( module );
}

OPTIXAPI inline OptixResult optixBuiltinISModuleGet( OptixDeviceContext                 context,
                                                     const OptixModuleCompileOptions*   moduleCompileOptions,
                                                     const OptixPipelineCompileOptions* pipelineCompileOptions,
                                                     const OptixBuiltinISOptions*       builtinISOptions,
                                                     OptixModule*                       builtinModule )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixBuiltinISModuleGet( context, moduleCompileOptions, pipelineCompileOptions,
                                                                builtinISOptions, builtinModule );
}

OPTIXAPI inline OptixResult optixTaskExecute( OptixTask     task,
                                              OptixTask*    additionalTasks,
                                              unsigned int  maxNumAdditionalTasks,
                                              unsigned int* numAdditionalTasksCreated )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixTaskExecute( task, additionalTasks, maxNumAdditionalTasks, numAdditionalTasksCreated );
}

OPTIXAPI inline OptixResult optixProgramGroupCreate( OptixDeviceContext              context,
                                                     const OptixProgramGroupDesc*    programDescriptions,
                                                     unsigned int                    numProgramGroups,
                                                     const OptixProgramGroupOptions* options,
                                                     char*                           logString,
                                                     size_t*                         logStringSize,
                                                     OptixProgramGroup*              programGroups )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixProgramGroupCreate( context, programDescriptions, numProgramGroups, options,
                                                                logString, logStringSize, programGroups );
}

OPTIXAPI inline OptixResult optixProgramGroupDestroy( OptixProgramGroup programGroup )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixProgramGroupDestroy( programGroup );
}

OPTIXAPI inline OptixResult optixProgramGroupGetStackSize( OptixProgramGroup programGroup, OptixStackSizes* stackSizes, OptixPipeline pipeline )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixProgramGroupGetStackSize( programGroup, stackSizes, pipeline );
}

OPTIXAPI inline OptixResult optixPipelineCreate( OptixDeviceContext                 context,
                                                 const OptixPipelineCompileOptions* pipelineCompileOptions,
                                                 const OptixPipelineLinkOptions*    pipelineLinkOptions,
                                                 const OptixProgramGroup*           programGroups,
                                                 unsigned int                       numProgramGroups,
                                                 char*                              logString,
                                                 size_t*                            logStringSize,
                                                 OptixPipeline*                     pipeline )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixPipelineCreate( context, pipelineCompileOptions, pipelineLinkOptions, programGroups,
                                                            numProgramGroups, logString, logStringSize, pipeline );
}

OPTIXAPI inline OptixResult optixPipelineDestroy( OptixPipeline pipeline )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixPipelineDestroy( pipeline );
}

OPTIXAPI inline OptixResult optixPipelineSetStackSize( OptixPipeline pipeline,
                                                       unsigned int  directCallableStackSizeFromTraversal,
                                                       unsigned int  directCallableStackSizeFromState,
                                                       unsigned int  continuationStackSize,
                                                       unsigned int  maxTraversableGraphDepth )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixPipelineSetStackSize( pipeline, directCallableStackSizeFromTraversal,
                                                                  directCallableStackSizeFromState,
                                                                  continuationStackSize, maxTraversableGraphDepth );
}

OPTIXAPI inline OptixResult optixAccelComputeMemoryUsage( OptixDeviceContext            context,
                                                          const OptixAccelBuildOptions* accelOptions,
                                                          const OptixBuildInput*        buildInputs,
                                                          unsigned int                  numBuildInputs,
                                                          OptixAccelBufferSizes*        bufferSizes )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixAccelComputeMemoryUsage( context, accelOptions, buildInputs, numBuildInputs, bufferSizes );
}

OPTIXAPI inline OptixResult optixAccelBuild( OptixDeviceContext            context,
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
                                             unsigned int                  numEmittedProperties )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixAccelBuild( context, stream, accelOptions, buildInputs, numBuildInputs, tempBuffer,
                                                        tempBufferSizeInBytes, outputBuffer, outputBufferSizeInBytes,
                                                        outputHandle, emittedProperties, numEmittedProperties );
}


OPTIXAPI inline OptixResult optixAccelGetRelocationInfo( OptixDeviceContext context, OptixTraversableHandle handle, OptixRelocationInfo* info )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixAccelGetRelocationInfo( context, handle, info );
}


OPTIXAPI inline OptixResult optixCheckRelocationCompatibility( OptixDeviceContext context, const OptixRelocationInfo* info, int* compatible )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixCheckRelocationCompatibility( context, info, compatible );
}

OPTIXAPI inline OptixResult optixAccelRelocate( OptixDeviceContext         context,
                                                CUstream                   stream,
                                                const OptixRelocationInfo* info,
                                                const OptixRelocateInput*  relocateInputs,
                                                size_t                     numRelocateInputs,
                                                CUdeviceptr                targetAccel,
                                                size_t                     targetAccelSizeInBytes,
                                                OptixTraversableHandle*    targetHandle )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixAccelRelocate( context, stream, info, relocateInputs, numRelocateInputs,
                                                           targetAccel, targetAccelSizeInBytes, targetHandle );
}

OPTIXAPI inline OptixResult optixAccelCompact( OptixDeviceContext      context,
                                               CUstream                stream,
                                               OptixTraversableHandle  inputHandle,
                                               CUdeviceptr             outputBuffer,
                                               size_t                  outputBufferSizeInBytes,
                                               OptixTraversableHandle* outputHandle )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixAccelCompact( context, stream, inputHandle, outputBuffer,
                                                          outputBufferSizeInBytes, outputHandle );
}

OPTIXAPI inline OptixResult optixAccelEmitProperty( OptixDeviceContext        context,
                                                    CUstream                  stream,
                                                    OptixTraversableHandle    handle,
                                                    const OptixAccelEmitDesc* emittedProperty )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixAccelEmitProperty( context, stream, handle, emittedProperty );
}

OPTIXAPI inline OptixResult optixConvertPointerToTraversableHandle( OptixDeviceContext      onDevice,
                                                                    CUdeviceptr             pointer,
                                                                    OptixTraversableType    traversableType,
                                                                    OptixTraversableHandle* traversableHandle )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixConvertPointerToTraversableHandle( onDevice, pointer, traversableType, traversableHandle );
}

OPTIXAPI inline OptixResult optixOpacityMicromapArrayComputeMemoryUsage( OptixDeviceContext context,
                                                                         const OptixOpacityMicromapArrayBuildInput* buildInput,
                                                                         OptixMicromapBufferSizes* bufferSizes )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixOpacityMicromapArrayComputeMemoryUsage( context, buildInput, bufferSizes );
}

OPTIXAPI inline OptixResult optixOpacityMicromapArrayBuild( OptixDeviceContext                         context,
                                                            CUstream                                   stream,
                                                            const OptixOpacityMicromapArrayBuildInput* buildInput,
                                                            const OptixMicromapBuffers*                buffers )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixOpacityMicromapArrayBuild( context, stream, buildInput, buffers );
}

OPTIXAPI inline OptixResult optixOpacityMicromapArrayGetRelocationInfo( OptixDeviceContext   context,
                                                                        CUdeviceptr          opacityMicromapArray,
                                                                        OptixRelocationInfo* info )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixOpacityMicromapArrayGetRelocationInfo( context, opacityMicromapArray, info );
}

OPTIXAPI inline OptixResult optixOpacityMicromapArrayRelocate( OptixDeviceContext         context,
                                                               CUstream                   stream,
                                                               const OptixRelocationInfo* info,
                                                               CUdeviceptr                targetOpacityMicromapArray,
                                                               size_t targetOpacityMicromapArraySizeInBytes )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixOpacityMicromapArrayRelocate( context, stream, info, targetOpacityMicromapArray,
                                                                          targetOpacityMicromapArraySizeInBytes );
}

OPTIXAPI inline OptixResult optixDisplacementMicromapArrayComputeMemoryUsage( OptixDeviceContext context,
                                                                              const OptixDisplacementMicromapArrayBuildInput* buildInput,
                                                                              OptixMicromapBufferSizes* bufferSizes )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixDisplacementMicromapArrayComputeMemoryUsage( context, buildInput, bufferSizes );
}

OPTIXAPI inline OptixResult optixDisplacementMicromapArrayBuild( OptixDeviceContext context,
                                                                 CUstream           stream,
                                                                 const OptixDisplacementMicromapArrayBuildInput* buildInput,
                                                                 const OptixMicromapBuffers* buffers )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixDisplacementMicromapArrayBuild( context, stream, buildInput, buffers );
}

OPTIXAPI inline OptixResult optixSbtRecordPackHeader( OptixProgramGroup programGroup, void* sbtRecordHeaderHostPointer )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixSbtRecordPackHeader( programGroup, sbtRecordHeaderHostPointer );
}

OPTIXAPI inline OptixResult optixLaunch( OptixPipeline                  pipeline,
                                         CUstream                       stream,
                                         CUdeviceptr                    pipelineParams,
                                         size_t                         pipelineParamsSize,
                                         const OptixShaderBindingTable* sbt,
                                         unsigned int                   width,
                                         unsigned int                   height,
                                         unsigned int                   depth )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixLaunch( pipeline, stream, pipelineParams, pipelineParamsSize, sbt, width, height, depth );
}

OPTIXAPI inline OptixResult optixDenoiserCreate( OptixDeviceContext          context,
                                                 OptixDenoiserModelKind      modelKind,
                                                 const OptixDenoiserOptions* options,
                                                 OptixDenoiser*              returnHandle )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixDenoiserCreate( context, modelKind, options, returnHandle );
}

OPTIXAPI inline OptixResult optixDenoiserCreateWithUserModel( OptixDeviceContext context,
                                                              const void*        data,
                                                              size_t             dataSizeInBytes,
                                                              OptixDenoiser*     returnHandle )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixDenoiserCreateWithUserModel( context, data, dataSizeInBytes, returnHandle );
}

OPTIXAPI inline OptixResult optixDenoiserDestroy( OptixDenoiser handle )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixDenoiserDestroy( handle );
}

OPTIXAPI inline OptixResult optixDenoiserComputeMemoryResources( const OptixDenoiser handle,
                                                                 unsigned int        maximumInputWidth,
                                                                 unsigned int        maximumInputHeight,
                                                                 OptixDenoiserSizes* returnSizes )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixDenoiserComputeMemoryResources( handle, maximumInputWidth, maximumInputHeight, returnSizes );
}

OPTIXAPI inline OptixResult optixDenoiserSetup( OptixDenoiser denoiser,
                                                CUstream      stream,
                                                unsigned int  inputWidth,
                                                unsigned int  inputHeight,
                                                CUdeviceptr   denoiserState,
                                                size_t        denoiserStateSizeInBytes,
                                                CUdeviceptr   scratch,
                                                size_t        scratchSizeInBytes )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixDenoiserSetup( denoiser, stream, inputWidth, inputHeight, denoiserState,
                                                           denoiserStateSizeInBytes, scratch, scratchSizeInBytes );
}

OPTIXAPI inline OptixResult optixDenoiserInvoke( OptixDenoiser                  handle,
                                                 CUstream                       stream,
                                                 const OptixDenoiserParams*     params,
                                                 CUdeviceptr                    denoiserData,
                                                 size_t                         denoiserDataSize,
                                                 const OptixDenoiserGuideLayer* guideLayer,
                                                 const OptixDenoiserLayer*      layers,
                                                 unsigned int                   numLayers,
                                                 unsigned int                   inputOffsetX,
                                                 unsigned int                   inputOffsetY,
                                                 CUdeviceptr                    scratch,
                                                 size_t                         scratchSizeInBytes )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixDenoiserInvoke( handle, stream, params, denoiserData, denoiserDataSize,
                                                            guideLayer, layers, numLayers, inputOffsetX, inputOffsetY,
                                                            scratch, scratchSizeInBytes );
}

OPTIXAPI inline OptixResult optixDenoiserComputeIntensity( OptixDenoiser       handle,
                                                           CUstream            stream,
                                                           const OptixImage2D* inputImage,
                                                           CUdeviceptr         outputIntensity,
                                                           CUdeviceptr         scratch,
                                                           size_t              scratchSizeInBytes )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixDenoiserComputeIntensity( handle, stream, inputImage, outputIntensity,
                                                                      scratch, scratchSizeInBytes );
}

OPTIXAPI inline OptixResult optixDenoiserComputeAverageColor( OptixDenoiser       handle,
                                                              CUstream            stream,
                                                              const OptixImage2D* inputImage,
                                                              CUdeviceptr         outputAverageColor,
                                                              CUdeviceptr         scratch,
                                                              size_t              scratchSizeInBytes )
{
    return OPTIX_FUNCTION_TABLE_SYMBOL.optixDenoiserComputeAverageColor( handle, stream, inputImage, outputAverageColor,
                                                                         scratch, scratchSizeInBytes );
}

#endif  // OPTIX_DOXYGEN_SHOULD_SKIP_THIS

#endif  // OPTIX_OPTIX_STUBS_H
