#pragma once 
#include "optix.h"
#include "raiicuda.h"

#include <algorithm>
#include <chrono>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef uint
using uint = unsigned int;
#endif

#ifndef ushort
using ushort = unsigned short;
#endif

struct IASBuildWorkspace {
    xinxinoptix::raii<CUdeviceptr> instances;
    xinxinoptix::raii<CUdeviceptr> temp;
    OptixAccelBufferSizes bufferSizes{};
    unsigned int cacheBuildFlags{};
    unsigned int cacheInstanceCount{};
    bool bufferSizesValid{};
};

template <typename T = double, typename = std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double>>>
struct CppTimer {
    using Clock = std::chrono::steady_clock;
    using Duration = std::chrono::duration<T, std::milli>;

    void tick() { start = Clock::now(); }

    T elapsed() const noexcept {
        auto elapsed_ns = std::chrono::nanoseconds(Clock::now() - start).count();
        return elapsed_ns / 1'000'000.0;
    }

    T tock(std::optional<std::string_view> tag = std::nullopt) {
        durat = elapsed();
        if (tag && !tag->empty()) {
            printf("%s: %f ms\n", tag->data(), durat);
        } else {
            return durat;
        }
    }

private:
    Clock::time_point start;
    T durat;
};

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

    struct NoUploadFence {
        void record(cudaStream_t) {}
    };

    template <class Allocator, class = void>
    struct IsPinnedHostAllocator : std::false_type {};

    template <class Allocator>
    struct IsPinnedHostAllocator<Allocator, std::void_t<typename Allocator::is_pinned_host_allocator>>
        : Allocator::is_pinned_host_allocator {};

    inline void reserveDeviceBuffer(raii<CUdeviceptr>& buffer, size_t required_size)
    {
        if (required_size == 0) return;
        if (buffer.handle != 0 && buffer.capacity >= required_size) {
            buffer.size = required_size;
            return;
        }
        buffer.resize(required_size);
    }

    inline void buildXASImpl(const OptixDeviceContext& context, OptixAccelBuildOptions& accel_options, OptixBuildInput& build_input,
                             CUdeviceptr& _bufferXAS_, size_t& retained_buffer_size,
                             OptixTraversableHandle& _handleXAS_, size_t aux_size=0, bool verbose=false,
                             bool reuse_output=false, raii<CUdeviceptr>* reusable_temp=nullptr,
                             const OptixAccelBufferSizes* cached_buffer_sizes=nullptr)
    {
        const bool update = OPTIX_BUILD_OPERATION_UPDATE == accel_options.operation;
        if (!update)
            _handleXAS_ = 0llu;

        size_t temp_buffer_size {};  
        size_t output_buffer_size {};
        {
            OptixAccelBufferSizes xas_buffer_sizes{};
            if (cached_buffer_sizes == nullptr) {
                OPTIX_CHECK( optixAccelComputeMemoryUsage(context,
                            &accel_options,
                            &build_input,
                            1, // num build inputs
                            &xas_buffer_sizes
                            ) );
            } else {
                xas_buffer_sizes = *cached_buffer_sizes;
            }
            const auto required_temp_size = update
                ? xas_buffer_sizes.tempUpdateSizeInBytes
                : xas_buffer_sizes.tempSizeInBytes;
            temp_buffer_size = roundUp<size_t>(required_temp_size, 8u);
            output_buffer_size = roundUp<size_t>( xas_buffer_sizes.outputSizeInBytes, 8u );

            if (verbose) {
                float temp_mb   = (float)temp_buffer_size   / (1024 * 1024);
                float output_mb = (float)output_buffer_size / (1024 * 1024);
                printf("Requires %f MB temp buffer and %f MB output buffer \n", temp_mb, output_mb);
            }
        }

        aux_size = roundUp<size_t>(aux_size, 128u);

        if (update && (_bufferXAS_ == 0 || _handleXAS_ == 0 || retained_buffer_size <= aux_size)) {
            throw std::runtime_error("Cannot update acceleration structure without a retained output buffer");
        }

        raii<CUdeviceptr> local_buffer_temp{};
        raii<CUdeviceptr>* buffer_temp_ptr = (reusable_temp == nullptr)? std::addressof(local_buffer_temp) : reusable_temp;
        auto& bufferTemp = *buffer_temp_ptr;
        if (reusable_temp == nullptr) {
            bufferTemp.resize(temp_buffer_size);
        } else {
            reserveDeviceBuffer(bufferTemp, temp_buffer_size);
        }

        // Do not synchronize here: report a completed failure from earlier
        // work on stream 0 before attributing it to optixAccelBuild. A busy
        // stream is expected and remains ordered with the build below.
        const cudaError_t stream_status = cudaStreamQuery(0);
        if (stream_status != cudaErrorNotReady)
            CUDA_CHECK(stream_status);

        const bool COMPACTION = !update && (accel_options.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION);

        if (!COMPACTION) {

            if (!update) {
                const auto required_output_size = output_buffer_size + aux_size;
                if (!reuse_output || _bufferXAS_ == 0 || retained_buffer_size < required_output_size) {
                    if (_bufferXAS_ != 0) {
                        CUDA_CHECK(cudaFreeAsync((void*)(_bufferXAS_), 0));
                        _bufferXAS_ = 0;
                        retained_buffer_size = 0;
                    }
                    CUDA_CHECK(cudaMallocAsync((void**)(&_bufferXAS_), required_output_size, 0));
                    retained_buffer_size = required_output_size;
                }
            }

            const auto output_buffer_size = retained_buffer_size - aux_size;
            const auto build_result = optixAccelBuild(
                                            context,
                                            0,  // CUDA stream
                                            &accel_options, &build_input,
                                            1,  // num build inputs
                                            bufferTemp,
                                            temp_buffer_size,
                                            (CUdeviceptr)( (char*)_bufferXAS_ + aux_size),
                                            output_buffer_size,
                                            &_handleXAS_,
                                            nullptr,
                                            0);
                                            
            if (build_result != OPTIX_SUCCESS) {
                unsigned int instance_count = 0;
                if (build_input.type == OPTIX_BUILD_INPUT_TYPE_INSTANCES)
                    instance_count = build_input.instanceArray.numInstances;

                std::fprintf(
                    stderr,
                    "optixAccelBuild failed: result=%d operation=%u flags=0x%x "
                    "inputType=%u instances=%u temp=%zu output=%zu retained=%zu "
                    "buffer=0x%llx handle=0x%llx priorStream=%s\n",
                    static_cast<int>(build_result),
                    static_cast<unsigned int>(accel_options.operation),
                    static_cast<unsigned int>(accel_options.buildFlags),
                    static_cast<unsigned int>(build_input.type),
                    instance_count,
                    temp_buffer_size,
                    output_buffer_size,
                    retained_buffer_size,
                    static_cast<unsigned long long>(_bufferXAS_),
                    static_cast<unsigned long long>(_handleXAS_),
                    cudaGetErrorString(stream_status));
            }
            OPTIX_CHECK(build_result);
        } else {

            CUdeviceptr output_buffer_xas {};
            CUDA_CHECK(cudaMallocAsync( reinterpret_cast<void**>( &output_buffer_xas ), output_buffer_size + sizeof(size_t) + aux_size, 0) );

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

                CUDA_CHECK(cudaFreeAsync((void*)output_buffer_xas, 0));
                retained_buffer_size = compacted_size + aux_size;
            }
            else
            {
                _bufferXAS_ = std::move(output_buffer_xas);
                retained_buffer_size = output_buffer_size + aux_size;
            }
        } // COMPACTION
    }

    inline void buildXAS(const OptixDeviceContext& context, OptixAccelBuildOptions& accel_options, OptixBuildInput& build_input,
                         raii<CUdeviceptr>& _bufferXAS_, OptixTraversableHandle& _handleXAS_, size_t aux_size=0, bool verbose=false,
                         bool reuse_output=false, raii<CUdeviceptr>* reusable_temp=nullptr,
                         const OptixAccelBufferSizes* cached_buffer_sizes=nullptr) {
        
        if (OPTIX_BUILD_OPERATION_BUILD == accel_options.operation && !reuse_output)
            _bufferXAS_.reset();

        buildXASImpl(context, accel_options, build_input, _bufferXAS_.handle, _bufferXAS_.capacity,
                     _handleXAS_, aux_size, verbose, reuse_output, reusable_temp, cached_buffer_sizes);
        _bufferXAS_.size = _bufferXAS_.capacity;
    }

    template <template <class> class ALLOC, class UploadFence = NoUploadFence>
    inline void buildIAS(OptixDeviceContext& context, std::vector<OptixInstance, ALLOC<OptixInstance>>& instances, 
                         raii<CUdeviceptr>& bufferIAS, OptixTraversableHandle& handleIAS, bool update=false,
                         IASBuildWorkspace* workspace=nullptr, bool prefer_fast_build=false, bool compact=false,
                         UploadFence* host_upload=nullptr)
    {

        if (instances.empty()) {
            bufferIAS.reset();
            handleIAS = 0llu;
            return;
        }

        raii<CUdeviceptr> local_temp;
        raii<CUdeviceptr> local_instances;
        raii<CUdeviceptr>* temp_ptr = (workspace == nullptr)? std::addressof(local_temp) : std::addressof(workspace->temp);
        raii<CUdeviceptr>* instances_ptr = (workspace==nullptr)? std::addressof(local_instances) : std::addressof(workspace->instances);
        auto& build_temp = *temp_ptr;
        auto& d_instances = *instances_ptr;
        const auto old_temp_capacity = build_temp.capacity;
        const auto old_instance_capacity = d_instances.capacity;
        const auto old_output_capacity = bufferIAS.capacity;
        const size_t size_in_bytes = sizeof( OptixInstance ) * instances.size();

        reserveDeviceBuffer(d_instances, size_in_bytes);
        CUDA_CHECK( cudaMemcpyAsync( (void*)d_instances.handle, instances.data(), size_in_bytes, cudaMemcpyHostToDevice, 0) );
        if (host_upload != nullptr) host_upload->record(0);

        OptixBuildInput instance_input{};
        instance_input.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        instance_input.instanceArray.instances    = d_instances;
        instance_input.instanceArray.numInstances = static_cast<unsigned int>( instances.size() );

        OptixAccelBuildOptions accel_options{};
        accel_options.operation = update? OPTIX_BUILD_OPERATION_UPDATE : OPTIX_BUILD_OPERATION_BUILD;
        // Dynamic IASes are rebuilt/refitted frequently, where compaction's
        // blocking size readback and second build are a poor trade. Static
        // hierarchy nodes keep compaction because they remain cached.
        if (compact) {
            accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        } else {
            accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
            if (prefer_fast_build) {
                accel_options.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
            }
        }

        const OptixAccelBufferSizes* cached_buffer_sizes = nullptr;
        if (workspace != nullptr) {
            const bool cache_matches = workspace->bufferSizesValid
                && workspace->cacheBuildFlags == accel_options.buildFlags
                && workspace->cacheInstanceCount == instance_input.instanceArray.numInstances;
            if (!cache_matches) {
                OPTIX_CHECK( optixAccelComputeMemoryUsage(context,
                            &accel_options,
                            &instance_input,
                            1,
                            &workspace->bufferSizes) );

                workspace->cacheBuildFlags = accel_options.buildFlags;
                workspace->cacheInstanceCount = instance_input.instanceArray.numInstances;
                workspace->bufferSizesValid = true;
            }
            cached_buffer_sizes = std::addressof(workspace->bufferSizes);
        }

        buildXAS(context, accel_options, instance_input, bufferIAS, handleIAS, 0, false, !compact,
                 temp_ptr, cached_buffer_sizes);
    }

    template <template <class> class ALLOC, class Node>
    inline void buildIAS(OptixDeviceContext& context, std::vector<OptixInstance, ALLOC<OptixInstance>>& instances,
                         Node& node, bool allow_update, bool prefer_fast_build=false, bool compact=false)
    {
        const auto requested_build_flags = [compact, allow_update, prefer_fast_build] () {
            uint flags = prefer_fast_build? OPTIX_BUILD_FLAG_PREFER_FAST_BUILD : OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
            if (compact)
                flags |= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
            if (allow_update)
                flags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE;
            return flags;
        } ();

        const bool update = allow_update
                   && node.count == instances.size()
                   && node.buffer.handle != 0
                   && node.buffer.capacity > 0
                   && node.handle != 0
                   && node.buildFlags == requested_build_flags;

        if constexpr (IsPinnedHostAllocator<ALLOC<OptixInstance>>::value) {
            buildIAS(context, instances, node.buffer, node.handle, update,
                     std::addressof(node.buildWorkspace), prefer_fast_build, compact,
                     std::addressof(node.hostInstancesUpload));
        } else {
            buildIAS(context, instances, node.buffer, node.handle, update,
                     std::addressof(node.buildWorkspace), prefer_fast_build, compact);
        }

        node.count = static_cast<uint32_t>(instances.size());
        node.buildFlags = requested_build_flags;
    }

    template <template <class> class ALLOC>
    inline void buildMeshGAS(bool update, const OptixDeviceContext& context, 
            std::vector<float3, ALLOC<float3> >& vertices, std::vector<uint3, ALLOC<uint3> >& indices, std::vector<uint16_t, ALLOC<uint16_t> >& mat_idx, 
            uint16_t sbt_count, raii<CUdeviceptr>& _bufferXAS_, OptixTraversableHandle& _handleXAS_, size_t extra_size, OptixBuildInputOpacityMicromap* inputOMM=nullptr)
    {
        if (vertices.empty()) { return; }

        OptixBuildOperation operation = OPTIX_BUILD_OPERATION_BUILD;
        if (update && _bufferXAS_.handle>0 && _bufferXAS_.capacity>0 && _handleXAS_>0)
            operation = OPTIX_BUILD_OPERATION_UPDATE;

        raii<CUdeviceptr> dverts {};
        raii<CUdeviceptr> dmats {};
        raii<CUdeviceptr> didx {};

        {
            const size_t size_in_byte = vertices.size() * sizeof( vertices[0] );
            CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &dverts ), size_in_byte, 0 ) );
            CUDA_CHECK( cudaMemcpyAsync( reinterpret_cast<void*>( (CUdeviceptr&)dverts ), vertices.data(), size_in_byte, cudaMemcpyHostToDevice) );
        }

        if (sbt_count > 1 && mat_idx.size()>1)
        {
            const size_t size_in_byte = mat_idx.size() * sizeof( mat_idx[0] );
            CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &dmats ), size_in_byte, 0 ) );
            CUDA_CHECK( cudaMemcpyAsync( reinterpret_cast<void*>( (CUdeviceptr)dmats ), mat_idx.data(), size_in_byte, cudaMemcpyHostToDevice ) );
        }
        
        if (!indices.empty())
        {
            const size_t size_in_byte = indices.size() * sizeof(uint3);
            CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &didx ), size_in_byte, 0) );
            CUDA_CHECK( cudaMemcpyAsync( reinterpret_cast<void*>( (CUdeviceptr)didx ), indices.data(), size_in_byte, cudaMemcpyHostToDevice) );
        }
        // // Build triangle GAS // // One per SBT record for this build input
        const auto numSbtRecords = (sbt_count<=1 || mat_idx.size()<=1) ? 1 : sbt_count;
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
        if (inputOMM != nullptr) {
            triangle_input.triangleArray.opacityMicromap = *inputOMM;
        }
        triangle_input.triangleArray.indexBuffer                 = didx;
        triangle_input.triangleArray.indexFormat                 = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangle_input.triangleArray.indexStrideInBytes          = sizeof(uint)*3;
        triangle_input.triangleArray.numIndexTriplets            = indices.size();

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accel_options.buildFlags            |= OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS;
        accel_options.buildFlags            |= OPTIX_BUILD_FLAG_ALLOW_OPACITY_MICROMAP_UPDATE | OPTIX_BUILD_FLAG_ALLOW_DISABLE_OPACITY_MICROMAPS;
        accel_options.operation              = operation;

        buildXAS(context, accel_options, triangle_input, _bufferXAS_, _handleXAS_, extra_size);
    }
}