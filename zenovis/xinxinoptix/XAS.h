#pragma once 


#include "optix.h"
#include "raiicuda.h"

#include <map>
#include <vector>
#include <sstream>
#include <iostream>
#include <functional>

#ifndef uint
using uint = unsigned int; 
#endif

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
                         CUdeviceptr& _bufferXAS_, OptixTraversableHandle& _handleXAS_, size_t aux_size=0, bool verbose=false)
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

            temp_buffer_size = roundUp<size_t>(xas_buffer_sizes.tempSizeInBytes, 8u);
            output_buffer_size = roundUp<size_t>( xas_buffer_sizes.outputSizeInBytes, 8u );

            if (verbose) {
                float temp_mb   = (float)temp_buffer_size   / (1024 * 1024);
                float output_mb = (float)output_buffer_size / (1024 * 1024);
                printf("Requires %f MB temp buffer and %f MB output buffer \n", temp_mb, output_mb);
            }
        }

        aux_size = roundUp<size_t>(aux_size, 128u);

        raii<CUdeviceptr> bufferTemp{};
        CUDA_CHECK( cudaMallocAsync(reinterpret_cast<void**>( &bufferTemp.handle ), temp_buffer_size, 0 ) );

        const bool COMPACTION = accel_options.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION;

        if (!COMPACTION) {

            CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &_bufferXAS_ ), output_buffer_size + aux_size, 0) );

            OPTIX_CHECK( optixAccelBuild(   context,
                                            0,  // CUDA stream
                                            &accel_options, &build_input,
                                            1,  // num build inputs
                                            bufferTemp, 
                                            temp_buffer_size,
                                            (CUdeviceptr)( (char*)_bufferXAS_ + aux_size),
                                            output_buffer_size,
                                            &_handleXAS_,
                                            nullptr,
                                            0 ) );
        } else {

            CUdeviceptr output_buffer_xas {};
            CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &output_buffer_xas ), output_buffer_size + sizeof(size_t) + aux_size, 0) );

            OptixAccelEmitDesc emitProperty {};
            emitProperty.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            emitProperty.result = ( CUdeviceptr )( (char*)(CUdeviceptr)output_buffer_xas + output_buffer_size + aux_size );

            OPTIX_CHECK( optixAccelBuild( context,
                                            0,  // CUDA stream
                                            &accel_options, &build_input,
                                            1,  // num build inputs
                                            bufferTemp, 
                                            temp_buffer_size,
                                            (CUdeviceptr)( (char*)output_buffer_xas + aux_size),
                                            output_buffer_size, 
                                            &_handleXAS_,
                                            &emitProperty,  // emitted property list
                                            1               // num emitted properties
                                            ) );

            //bufferTemp.reset();
            size_t compacted_size{};
            CUDA_CHECK( cudaMemcpy( &compacted_size, (void*)emitProperty.result, sizeof( size_t ), cudaMemcpyDeviceToHost ) );

            if( compacted_size < output_buffer_size )
            {
                CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &_bufferXAS_ ), compacted_size + aux_size, 0) );
                OPTIX_CHECK( optixAccelCompact( context, 0, _handleXAS_, 
                (CUdeviceptr)( (char*)_bufferXAS_ + aux_size), compacted_size, &_handleXAS_ ) );

                cudaFreeAsync((void*)output_buffer_xas, 0);
            }
            else
            {
                _bufferXAS_ = std::move(output_buffer_xas);
            }
        } // COMPACTION
    }

    inline void buildXAS(const OptixDeviceContext& context, OptixAccelBuildOptions& accel_options, OptixBuildInput& build_input,
                         raii<CUdeviceptr>& _bufferXAS_, OptixTraversableHandle& _handleXAS_, size_t aux_size=0, bool verbose=false) {
                                               
        buildXAS(context, accel_options, build_input, _bufferXAS_.reset(), _handleXAS_, aux_size, verbose);
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
        CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &d_instances.reset() ), size_in_bytes, 0) );
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

    inline void buildIAS(OptixDeviceContext& context, std::vector<OptixInstance>& instances, raii<CUdeviceptr>& bufferIAS, OptixTraversableHandle& handleIAS) 
    {
        OptixAccelBuildOptions accel_options{};
        accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS;

        buildIAS(context, accel_options, instances, bufferIAS, handleIAS);
    } 

    inline void buildMeshGAS(const OptixDeviceContext& context, std::vector<float3>& vertices, std::vector<uint3>& indices, std::vector<uint16_t>& mat_idx, const std::map<std::string, uint16_t>& mtlidlut,
                                raii<CUdeviceptr>& _bufferXAS_, OptixTraversableHandle& _handleXAS_, size_t extra_size)
    {
        if (vertices.empty()) { return; }

        raii<CUdeviceptr> dverts {};
        raii<CUdeviceptr> dmats {};
        raii<CUdeviceptr> didx {};

        {
            const size_t size_in_byte = vertices.size() * sizeof( vertices[0] );
            CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &dverts ), size_in_byte, 0 ) );
            CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( (CUdeviceptr&)dverts ), vertices.data(), size_in_byte, cudaMemcpyHostToDevice) );
        }

        if (mat_idx.size()>1)
        {
            const size_t size_in_byte = mat_idx.size() * sizeof( mat_idx[0] );
            CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &dmats ), size_in_byte, 0 ) );
            CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( (CUdeviceptr)dmats ), mat_idx.data(), size_in_byte, cudaMemcpyHostToDevice ) );
        }

        if (!indices.empty())
        {
            const size_t size_in_byte = indices.size() * sizeof(uint3);
            CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &didx ), size_in_byte, 0) );
            CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( (CUdeviceptr)didx ), indices.data(), size_in_byte, cudaMemcpyHostToDevice
                        ) );
        }
        // // Build triangle GAS // // One per SBT record for this build input
        const auto numSbtRecords = (mtlidlut.empty() || mat_idx.size()<=1) ? 1 : mtlidlut.size();
        std::vector<uint32_t> triangle_input_flags( numSbtRecords, OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL );

        OptixBuildInput triangle_input                           = {};
        triangle_input.type                                      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triangle_input.triangleArray.vertexFormat                = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangle_input.triangleArray.vertexStrideInBytes         = sizeof( float3 );
        triangle_input.triangleArray.numVertices                 = static_cast<uint32_t>( vertices.size() );
        triangle_input.triangleArray.vertexBuffers               = vertices.empty() ? nullptr : &dverts;
        triangle_input.triangleArray.flags                       = triangle_input_flags.data();
        triangle_input.triangleArray.numSbtRecords               = numSbtRecords;
        triangle_input.triangleArray.sbtIndexOffsetBuffer        = dmats;
        triangle_input.triangleArray.sbtIndexOffsetSizeInBytes   = sizeof( uint16_t );
        triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof( uint16_t );
        
        triangle_input.triangleArray.indexBuffer                 = didx;
        triangle_input.triangleArray.indexFormat                 = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangle_input.triangleArray.indexStrideInBytes          = sizeof(uint)*3;
        triangle_input.triangleArray.numIndexTriplets            = indices.size();

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS;
        accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

        buildXAS(context, accel_options, triangle_input, _bufferXAS_, _handleXAS_, extra_size);
    }
}