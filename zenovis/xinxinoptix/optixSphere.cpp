#include "optixSphere.h"

namespace xinxinoptix 
{

void preload_sphere_transformed(std::string const &key, std::string const &mtlid, const std::string &instID, const glm::mat4& transform) 
{
    InfoSphereTransformed dsphere;
    dsphere.materialID = mtlid;
    dsphere.instanceID = instID;
    dsphere.optix_transform = glm::transpose(transform);

    SphereTransformedTable[key] = dsphere;
    sphere_unique_mats.insert(mtlid);
}

void preload_sphere_instanced(std::string const &key, std::string const &mtlid, const std::string &instID, const float &radius, const zeno::vec3f &center) 
{
    SphereInstanceGroupBase base;
    base.instanceID = instID;
    base.materialID = mtlid;
    base.key = key;

    base.radius = radius;
    base.center = center;

    SpheresInstanceGroupMap[instID] = base;

    sphere_unique_mats.insert(mtlid);
}

void buildUniformedSphereGAS(const OptixDeviceContext& context,  OptixTraversableHandle& gas_handle, raii<CUdeviceptr>& d_gas_output_buffer) 
{
    OptixAccelBuildOptions accel_options{};
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
                            OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS |
                            OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS;
                            
    float3 sphereVertex = make_float3( 0.f, 0.f, 0.f );
    float  sphereRadius = 0.5f;

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

    buildXAS(context, accel_options, sphere_input, d_gas_output_buffer, gas_handle);

    CUDA_CHECK( cudaFree( (void*)d_vertex_buffer ) );
    CUDA_CHECK( cudaFree( (void*)d_radius_buffer ) );
}

void buildInstancedSpheresGAS(const OptixDeviceContext &context, std::vector<std::shared_ptr<SphereInstanceAgent>>& agentList) {

    OptixAccelBuildOptions accel_options{};
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
                                OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS |
                                OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS;

    raii<CUdeviceptr> _vertex_buffer;
    raii<CUdeviceptr> _radius_buffer;

    for (auto &sphereAgent : agentList) {

        const auto sphere_count = sphereAgent->center_list.size();
        if (sphere_count == 0) continue; 

        {
            auto data_length = sizeof( zeno::vec3f ) * sphere_count;

            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &_vertex_buffer ), data_length) );
            CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( (CUdeviceptr)_vertex_buffer ), sphereAgent->center_list.data(),
                                    data_length, cudaMemcpyHostToDevice ) );
        }
        
        {
            auto data_length = sizeof( float ) * sphere_count;

            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &_radius_buffer ), data_length) );
            CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( (CUdeviceptr)_radius_buffer ), sphereAgent->radius_list.data(), 
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
        buildXAS(context, accel_options, sphere_input, sphereAgent->inst_sphere_gas_buffer, sphereAgent->inst_sphere_gas_handle, true);

        _vertex_buffer.reset();
        _radius_buffer.reset();

        sphereAgent->center_list.clear();
        sphereAgent->radius_list.clear();
    }
}

} // namespace end