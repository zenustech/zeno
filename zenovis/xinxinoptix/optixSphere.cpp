#include "optixSphere.h"

void buildUnitSphereGAS(const OptixDeviceContext& context,  OptixTraversableHandle& gas_handle, xinxinoptix::raii<CUdeviceptr>& d_gas_output_buffer) 
{
    OptixAccelBuildOptions accel_options{};
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
                            OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS |
                            OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS;

    float4 sphereVertex = make_float4(0, 0, 0, 1);

    xinxinoptix::raii<CUdeviceptr> d_vertex_buffer {};
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_vertex_buffer.handle ), sizeof( float4) ) );
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_vertex_buffer.handle ), &sphereVertex,
                            sizeof( float4 ), cudaMemcpyHostToDevice ) );

    CUdeviceptr d_radius_buffer = (CUdeviceptr) ((char*)d_vertex_buffer.handle + sizeof(float3));

    OptixBuildInput sphere_input{};

    sphere_input.type                      = OPTIX_BUILD_INPUT_TYPE_SPHERES;
    sphere_input.sphereArray.numVertices   = 1;
    sphere_input.sphereArray.vertexBuffers = &d_vertex_buffer;
    sphere_input.sphereArray.radiusBuffers = &d_radius_buffer;

    sphere_input.sphereArray.primitiveIndexOffset = 0; 

    uint32_t sphere_input_flags[1]         = {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
    sphere_input.sphereArray.flags         = sphere_input_flags;
    sphere_input.sphereArray.numSbtRecords = 1;

    buildXAS(context, accel_options, sphere_input, d_gas_output_buffer, gas_handle, 8);
    cudaMemset((char*)d_gas_output_buffer.handle+128-8, 0, 8);
}

void buildSphereGroupGAS(const OptixDeviceContext &context, SphereGroup& group) {

    OptixAccelBuildOptions accel_options{};
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
                                OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS |
                                OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS;

    using namespace xinxinoptix;

    raii<CUdeviceptr> _vertex_buffer;
    raii<CUdeviceptr> _radius_buffer;

    const auto sphere_count = group.centerV.size();
    if (sphere_count == 0) return;
    {
        auto data_length = sizeof( zeno::vec3f ) * sphere_count;

        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &_vertex_buffer ), data_length) );
        CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( (CUdeviceptr)_vertex_buffer ), group.centerV.data(),
                                data_length, cudaMemcpyHostToDevice ) );
    }
    
    {
        auto data_length = sizeof( float ) * sphere_count;

        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &_radius_buffer ), data_length) );
        CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( (CUdeviceptr)_radius_buffer ), group.radiusV.data(), 
                                data_length, cudaMemcpyHostToDevice ) );
    }

    OptixBuildInput sphere_input{};

    sphere_input.type                      = OPTIX_BUILD_INPUT_TYPE_SPHERES;
    sphere_input.sphereArray.numVertices   = sphere_count;
    sphere_input.sphereArray.vertexBuffers = &_vertex_buffer;
    sphere_input.sphereArray.radiusBuffers = &_radius_buffer;
    //sphere_input.sphereArray.singleRadius = false;
    //sphere_input.sphereArray.vertexStrideInBytes = 12;
    //sphere_input.sphereArray.radiusStrideInBytes = 4;

    uint32_t sphere_input_flags[1] = {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
    
    sphere_input.sphereArray.flags         = sphere_input_flags;
    sphere_input.sphereArray.numSbtRecords = 1;
    sphere_input.sphereArray.sbtIndexOffsetBuffer = 0;
    sphere_input.sphereArray.sbtIndexOffsetSizeInBytes = 0;
    sphere_input.sphereArray.sbtIndexOffsetStrideInBytes = 0;

    std::cout << "sphere_count: " << sphere_count << std::endl;
    buildXAS(context, accel_options, sphere_input, group.node->buffer, group.node->handle, 8);

    _vertex_buffer.reset();
    _radius_buffer.reset();

    auto& _color_buffer = group.color_buffer;
    {
        auto data_length = sizeof( zeno::vec3f ) * sphere_count;

        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &_color_buffer ), data_length) );
        CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( (CUdeviceptr)_color_buffer ), group.colorV.data(), 
                                data_length, cudaMemcpyHostToDevice ) );
    }

    auto color_offset = group.node->buffer + 128u - 8u;
    {
        cudaMemcpy((void*)color_offset, &_color_buffer.handle, 8, cudaMemcpyHostToDevice);
    }
}