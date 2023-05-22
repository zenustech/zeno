#include <tuple>

#include <optix.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>
#include <sutil/Exception.h>

#include "raiicuda.h"
#include "xinxinoptixapi.h"

namespace xinxinoptix 
{

inline void updateSphereGAS(const OptixDeviceContext& context, OptixBuildInput& sphere_input, OptixAccelBuildOptions& accel_options, 
                     raii<CUdeviceptr>& sphere_gas_buffer, OptixTraversableHandle& sphere_gas_handle) 
{
    OptixAccelBufferSizes gas_buffer_sizes{};
    OPTIX_CHECK( optixAccelComputeMemoryUsage( context, &accel_options, &sphere_input, 1, &gas_buffer_sizes ) );

    size_t temp_buffer_size = roundUp<size_t>(gas_buffer_sizes.tempSizeInBytes, 8u);
    size_t output_buffer_size = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8u );
    
    raii<CUdeviceptr> d_temp_buffer_gas {};
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_temp_buffer_gas ), temp_buffer_size));

    printf("Requires %lu MB for tmp and %lu MB output for sphere GAS \n", temp_buffer_size / (1024 * 1024), output_buffer_size / (1024 * 1024));

    const bool COMPACTION = accel_options.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION;

    if (!COMPACTION) {

        auto& output_buffer_gas = sphere_gas_buffer.reset();
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &output_buffer_gas ), output_buffer_size) );

        OPTIX_CHECK( optixAccelBuild(   context,
                                        nullptr,  // CUDA stream
                                        &accel_options, &sphere_input,
                                        1,  // num build inputs
                                        d_temp_buffer_gas, 
                                        gas_buffer_sizes.tempSizeInBytes,
                                        output_buffer_gas,
                                        gas_buffer_sizes.outputSizeInBytes,
                                        &sphere_gas_handle,
                                        nullptr,
                                        0 ) );
    } else {

        sphere_gas_buffer.reset();

        raii<CUdeviceptr> output_buffer_gas {};
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &output_buffer_gas ), output_buffer_size + sizeof(size_t)) );

        OptixAccelEmitDesc emitProperty {};
        emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitProperty.result = ( CUdeviceptr )( (char*)(CUdeviceptr)output_buffer_gas.handle + output_buffer_size );

        OPTIX_CHECK( optixAccelBuild( context,
                                        0,  // CUDA stream
                                        &accel_options, &sphere_input,
                                        1,  // num build inputs
                                        d_temp_buffer_gas, 
                                        gas_buffer_sizes.tempSizeInBytes,
                                        output_buffer_gas, 
                                        gas_buffer_sizes.outputSizeInBytes, 
                                        &sphere_gas_handle,
                                        &emitProperty,  // emitted property list
                                        1               // num emitted properties
                                        ) );

        size_t compacted_gas_size{};
        CUDA_CHECK( cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof( size_t ), cudaMemcpyDeviceToHost ) );

        if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
        {
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &sphere_gas_buffer ), compacted_gas_size ) );
            // use handle as input and output
            OPTIX_CHECK( optixAccelCompact( context, 0, 
                sphere_gas_handle, 
                sphere_gas_buffer, 
            compacted_gas_size, &sphere_gas_handle ) );
        }
        else
        {
            sphere_gas_buffer = std::move(output_buffer_gas);
        }
    }
}

inline void makeUniformSphereGAS(const OptixDeviceContext& context,  OptixTraversableHandle& gas_handle, raii<CUdeviceptr>& d_gas_output_buffer) 
{
    OptixAccelBuildOptions accel_options{};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

    float3 sphereVertex = make_float3( 0.f, 0.f, 0.f );
    float  sphereRadius = 1.0f;

    CUdeviceptr d_vertex_buffer{};
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_vertex_buffer ), sizeof( float3 ) ) );
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_vertex_buffer ), &sphereVertex,
                            sizeof( float3 ), cudaMemcpyHostToDevice ) );

    CUdeviceptr d_radius_buffer{};
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_radius_buffer ), sizeof( float ) ) );
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_radius_buffer ), &sphereRadius, sizeof( float ),
                            cudaMemcpyHostToDevice ) );

    OptixBuildInput sphere_input{};

    sphere_input.type                      = OPTIX_BUILD_INPUT_TYPE_SPHERES;
    sphere_input.sphereArray.numVertices   = 1;
    sphere_input.sphereArray.vertexBuffers = &d_vertex_buffer;
    sphere_input.sphereArray.radiusBuffers = &d_radius_buffer;

    sphere_input.sphereArray.primitiveIndexOffset = 0; 

    uint32_t sphere_input_flags[1]         = {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
    sphere_input.sphereArray.flags         = sphere_input_flags;
    sphere_input.sphereArray.numSbtRecords = 1;

    updateSphereGAS(context, sphere_input, accel_options, d_gas_output_buffer, gas_handle);

    CUDA_CHECK( cudaFree( (void*)d_vertex_buffer ) );
    CUDA_CHECK( cudaFree( (void*)d_radius_buffer ) );
}

} // NAMESPACE END