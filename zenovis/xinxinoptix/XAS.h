#pragma once 

#include <vector>
#include "optix.h"
#include "raiicuda.h"

#include <sstream>
#include <iostream>
#include <functional>

#define RETURN_IF_CUDA_ERROR( call )                                           \
        cudaError_t error = call;                                              \
        if( error != cudaSuccess )                                             \
        {                                                                      \
            printf("CUDA call ( \" %s \" ) failed with error: %s (%s: %d) \n", \
                    #call, cudaGetErrorString( error ), __FILE__, __LINE__);   \
            cudaGetLastError();                                                \
            return;                                                            \
        }                                                                      \
        //(error);                                                             \
                                                                               \
                                                                               
namespace xinxinoptix {

    inline void buildXAS(const OptixDeviceContext& context, OptixAccelBuildOptions& accel_options, OptixBuildInput& build_input,
                         CUdeviceptr& _bufferXAS_, OptixTraversableHandle& _handleXAS_, bool verbose=false)
    {

        _handleXAS_ = 0llu;

        size_t temp_buffer_size {};  
        size_t output_buffer_size {};
        {
            OptixAccelBufferSizes xas_buffer_sizes;
            OPTIX_CHECK( optixAccelComputeMemoryUsage(context,
                        &accel_options,
                        &build_input,
                        1, // num build inputs
                        &xas_buffer_sizes
                        ) );

            temp_buffer_size = roundUp<size_t>(xas_buffer_sizes.tempSizeInBytes, 128u);
            output_buffer_size = roundUp<size_t>( xas_buffer_sizes.outputSizeInBytes, 128u );

            if (verbose) {
                float temp_mb   = (float)temp_buffer_size   / (1024 * 1024);
                float output_mb = (float)output_buffer_size / (1024 * 1024);
                printf("Requires %f MB temp buffer and %f MB output buffer \n", temp_mb, output_mb);
            }
        }

        raii<CUdeviceptr> bufferTemp{};
        CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &bufferTemp.handle ), temp_buffer_size ) );

        const bool COMPACTION = accel_options.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION;

        if (!COMPACTION) {

            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &_bufferXAS_ ), output_buffer_size ) );

            OPTIX_CHECK( optixAccelBuild(   context,
                                            nullptr,  // CUDA stream
                                            &accel_options, &build_input,
                                            1,  // num build inputs
                                            bufferTemp, 
                                            temp_buffer_size,
                                            _bufferXAS_,
                                            output_buffer_size,
                                            &_handleXAS_,
                                            nullptr,
                                            0 ) );
        } else {

            CUdeviceptr output_buffer_xas {};
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &output_buffer_xas ), output_buffer_size + sizeof(size_t)) );

            OptixAccelEmitDesc emitProperty {};
            emitProperty.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            emitProperty.result = ( CUdeviceptr )( (char*)(CUdeviceptr)output_buffer_xas + output_buffer_size );

            OPTIX_CHECK( optixAccelBuild( context,
                                            0,  // CUDA stream
                                            &accel_options, &build_input,
                                            1,  // num build inputs
                                            bufferTemp, 
                                            temp_buffer_size,
                                            output_buffer_xas, 
                                            output_buffer_size, 
                                            &_handleXAS_,
                                            &emitProperty,  // emitted property list
                                            1               // num emitted properties
                                            ) );

            bufferTemp.reset();
            size_t compacted_size{};
            CUDA_CHECK( cudaMemcpy( &compacted_size, (void*)emitProperty.result, sizeof( size_t ), cudaMemcpyDeviceToHost ) );

            if( compacted_size < output_buffer_size )
            {
                CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &_bufferXAS_ ), compacted_size ) );
                OPTIX_CHECK( optixAccelCompact( context, 0, _handleXAS_, _bufferXAS_, compacted_size, &_handleXAS_ ) );

                cudaFree((void*)output_buffer_xas);
            }
            else
            {
                _bufferXAS_ = std::move(output_buffer_xas);
            }
        } // COMPACTION
    }

    inline void buildXAS(const OptixDeviceContext& context, OptixAccelBuildOptions& accel_options, OptixBuildInput& build_input,
                         raii<CUdeviceptr>& _bufferXAS_, OptixTraversableHandle& _handleXAS_, bool verbose=false) {
                                               
        buildXAS(context, accel_options, build_input, _bufferXAS_.reset(), _handleXAS_, verbose);
    }

    inline void buildIAS(OptixDeviceContext& context, OptixAccelBuildOptions& accel_options, std::vector<OptixInstance>& instances, 
                         raii<CUdeviceptr>& bufferIAS, OptixTraversableHandle& handleIAS) 
    {

        if (instances.empty()) {
            bufferIAS.reset();
            handleIAS = 0llu;
            return;
        }

        raii<CUdeviceptr>  d_instances;
        const size_t size_in_bytes = sizeof( OptixInstance ) * instances.size();
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_instances.reset() ), size_in_bytes ) );
        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>( (CUdeviceptr)d_instances ),
                    instances.data(),
                    size_in_bytes,
                    cudaMemcpyHostToDevice
                    ) );

        OptixBuildInput instance_input{};
        instance_input.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        instance_input.instanceArray.instances    = d_instances;
        instance_input.instanceArray.numInstances = static_cast<unsigned int>( instances.size() );

        // OptixAccelBuildOptions accel_options{};
        // accel_options.buildFlags                  = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS;
        // accel_options.operation                   = OPTIX_BUILD_OPERATION_BUILD;

        buildXAS(context, accel_options, instance_input, bufferIAS, handleIAS);
    } 
}