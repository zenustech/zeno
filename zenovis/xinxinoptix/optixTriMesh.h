#pragma once 

#include "optixCommon.h"

#include <map>
#include <vector>
#include "OptiXStuff.h"

//#include <OptiXToolkit/CuOmmBaking/CuBuffer.h>
#include <OptiXToolkit/CuOmmBaking/CuOmmBaking.h>

// Check status returned by a CUDA call.
inline void check( cudaError_t status )
{
    if( status != cudaSuccess )
        throw std::runtime_error( cudaGetErrorString( status ) );
}

// Check status returned by a OptiX call.
inline void check( OptixResult status )
{
    if( status != OPTIX_SUCCESS )
        throw std::runtime_error( "OptiX failure." );
}

// Check status returned by a CuOmmBaking call.
inline void check( cuOmmBaking::Result status )
{
    if( status != cuOmmBaking::Result::SUCCESS )
        throw std::runtime_error( "Omm baking failure." );
}

struct MeshDat {
    bool dirty = true;

    std::vector<std::string> mtlidList;
    std::string mtlid;
    std::vector<float> verts;
    std::vector<uint> tris;
    std::vector<int> triMats;

    std::vector<float> dummy{};
    std::map<std::string, std::vector<float>> vertattrs;
    auto const &getAttr(std::string const &s) const
    {
        auto find = vertattrs.find(s);
        if(find!=vertattrs.end())
            return find->second;
        else
            return dummy;
    }
};

struct MeshObject {
    bool dirty = true;
    std::string matid;

    std::vector<float3> vertices{};
    std::vector<uint3>  indices {};
    std::vector<uint16_t> mat_idx{};

    std::vector<float2> g_uv;
    std::vector<ushort3> g_clr, g_nrm, g_tan;

template<class T, class E=uint8_t>
using raii = xinxinoptix::raii<T, E>;

    raii<CUdeviceptr> d_uv, d_clr, d_nrm, d_tan;
    raii<CUdeviceptr> d_idx;
    raii<CUdeviceptr> d_mat;

    std::shared_ptr<SceneNode> node = std::make_shared<SceneNode>();
    
    MeshObject() = default;

    void resize(size_t tri_num, size_t vert_num, bool has_clr, bool has_uv) {

        indices.resize(tri_num);
        //mat_idx.resize(tri_num);
        vertices.resize(vert_num);
        g_nrm.resize(vert_num);

        if (has_clr) {
            g_clr.resize(vert_num);
        } else {
            g_clr.resize(0);
            g_clr.shrink_to_fit();
        }
        if (has_uv) {
            g_tan.resize(vert_num);
            g_uv.resize(vert_num);
        } else {
            g_tan.resize(0);
            g_tan.shrink_to_fit();
            g_uv.resize(0);
            g_uv.shrink_to_fit();
        }
    }

    template<class T>
    static void upload(std::vector<T>& vector, raii<CUdeviceptr>& buffer) {

        if (!vector.empty()) {

            auto byte_size = sizeof(vector[0]) * vector.size();
            buffer.resize(byte_size);
            cudaMemcpy((void*)buffer.handle, vector.data(), byte_size, cudaMemcpyHostToDevice);

        } else { buffer.reset(); }
    }

    void upload(size_t extra_size) {

        upload(g_uv, d_uv);
        upload(g_clr, d_clr);
        upload(g_nrm, d_nrm);
        upload(g_tan, d_tan);
        upload(indices, d_idx);

        auto offset = roundUp<size_t>(extra_size, 128u);
        auto gas_ptr = node->buffer.handle + offset;

        auto buffers = this->aux();
        if (0) {
            upload(mat_idx, d_mat);
            buffers.push_back(d_mat.handle);
        } else {
            d_mat.reset();
            buffers.push_back(0);
        }
        std::reverse(buffers.begin(), buffers.end());

        auto byte_size = sizeof(buffers[0]) * buffers.size();
        cudaMemcpy((void*)(gas_ptr-byte_size), buffers.data(), byte_size, cudaMemcpyHostToDevice);
    }

    std::vector<CUdeviceptr> aux() {
        return std::vector { d_idx.handle, d_uv.handle, d_clr.handle, d_nrm.handle, d_tan.handle };
    }

    void buildGas(OptixDeviceContext context, uint16_t sbt_count,
        const std::map<uint16_t, zeno::OpacityMicroMapConfig>& binding_cfg_map={}) {

        auto buffers = this->aux();
        std::reverse(buffers.begin(), buffers.end());
        auto extra_size = sizeof(buffers[0]) * buffers.size();

        if (binding_cfg_map.size()>0) { 
            inputOMM = bakeOMM(context, binding_cfg_map);
        } else {
            inputOMM = nullptr;
            check(d_ommArray.free());
            check(d_ommIndices.free());
        }
        xinxinoptix::buildMeshGAS(context, vertices, indices, mat_idx, sbt_count, node->buffer, node->handle, extra_size, inputOMM.get());
        upload(extra_size);
    }

    using ommc = zeno::OpacityMicroMapConfig;
    static inline std::map<ommc::AlphaMode, cuOmmBaking::CudaTextureAlphaMode> amlut
    {
        {ommc::AlphaMode::Auto, cuOmmBaking::CudaTextureAlphaMode::DEFAULT},
        {ommc::AlphaMode::Max,  cuOmmBaking::CudaTextureAlphaMode::MAX_NUM},
        {ommc::AlphaMode::RGB,  cuOmmBaking::CudaTextureAlphaMode::RGB_INTENSITY},

        {ommc::AlphaMode::X,    cuOmmBaking::CudaTextureAlphaMode::CHANNEL_X},
        {ommc::AlphaMode::Y,    cuOmmBaking::CudaTextureAlphaMode::CHANNEL_Y},
        {ommc::AlphaMode::Z,    cuOmmBaking::CudaTextureAlphaMode::CHANNEL_Z},
        {ommc::AlphaMode::W,    cuOmmBaking::CudaTextureAlphaMode::CHANNEL_W},
    };

    std::shared_ptr<OptixBuildInputOpacityMicromap> bakeOMM(OptixDeviceContext context,  
        const std::map<uint16_t, zeno::OpacityMicroMapConfig>& binding_cfg_map) 
    {
        // Upload geometry data.
        std::vector<uint8_t> tri_tex_idx;
        tri_tex_idx.reserve(indices.size());

        auto fallback_cfg = zeno::OpacityMicroMapConfig();
        std::vector<zeno::OpacityMicroMapConfig> cfg_array;
        cfg_array.reserve(binding_cfg_map.size()+1);
        cfg_array.push_back(fallback_cfg);

        auto fallback_tex = OptixUtil::makeFallbackAlphaTexture();
        std::vector<cudaTextureObject_t> tex_array;
        tex_array.reserve(binding_cfg_map.size()+1);
        tex_array.push_back(fallback_tex->texture);

        std::map<uint16_t, uint8_t> binding_to_tex_array_idx;
        
        for (auto& [binding, cfg] : binding_cfg_map) 
        {
            binding_to_tex_array_idx[binding] = tex_array.size();
            tex_array.push_back(cfg.reference);
            cfg_array.push_back(cfg);
        }

        if (mat_idx.size() == 1) {
            auto binding = mat_idx[0];
            if (binding_cfg_map.count(binding)==0)
                return nullptr;

            auto tex_idx = binding_to_tex_array_idx[binding];
            tri_tex_idx.resize(indices.size(), tex_idx);
        }
        else {
            for (uint i = 0; i < indices.size(); ++i) {
                auto binding = mat_idx[i];
                if (binding_cfg_map.count(binding) == 0)
                    tri_tex_idx.push_back(0);
                else {
                    auto idx = binding_to_tex_array_idx[binding];
                    tri_tex_idx.push_back(idx);
                }
            }
        }

        raii<CUdeviceptr, uint3> d_geoIndices;
        d_geoIndices.allocAndUpload(indices);

        raii<CUdeviceptr, float2> d_geoTexCoords;
        d_geoTexCoords.allocAndUpload(g_uv);

        raii<CUdeviceptr, uint8_t> textureIndexBuffer;
        textureIndexBuffer.allocAndUpload(tri_tex_idx);

        // Bake the Opacity Micromap data.
        cuOmmBaking::BakeOptions ommOptions = {};
        std::vector<cuOmmBaking::TextureDesc> textures(tex_array.size());

        for(int idx=0; idx<tex_array.size(); ++idx)
        {
            auto& cfg = cfg_array[idx];
            auto& texture = textures[idx];
            // Use the cuda texture directly.
            texture.type = cuOmmBaking::TextureType::CUDA;
            texture.cuda.texObject = tex_array[idx];
            texture.cuda.transparencyCutoff = cfg.transparencyCutoff;
            texture.cuda.opacityCutoff = cfg.opacityCutoff;
            texture.cuda.filterKernelWidthInTexels = 1.0f;

            if ( amlut.count(cfg.alphaMode) == 0)
                texture.cuda.alphaMode = cuOmmBaking::CudaTextureAlphaMode::DEFAULT;
            else 
                texture.cuda.alphaMode = amlut.at(cfg.alphaMode);
        }

        // Prepare for baking by query the pre baking info.
        cuOmmBaking::BakeInputDesc bakeInput {};
        cuOmmBaking::BakeBuffers bakeBuffers {};
        cuOmmBaking::BakeInputBuffers bakeInputBuffers;
        {
            bakeInput.indexFormat = cuOmmBaking::IndexFormat::I32_UINT;
            bakeInput.indexBuffer = d_geoIndices.get();
            bakeInput.numIndexTriplets = indices.size();

            bakeInput.texCoordFormat = cuOmmBaking::TexCoordFormat::UV32_FLOAT2;
            bakeInput.texCoordBuffer = d_geoTexCoords.get();

            bakeInput.numTextures = tex_array.size();
            bakeInput.textures = textures.data();
            bakeInput.textureIndexBuffer = textureIndexBuffer.get();
            bakeInput.textureIndexFormat = cuOmmBaking::IndexFormat::I8_UINT;

            check( cuOmmBaking::GetPreBakeInfo( &ommOptions, 1, &bakeInput, &bakeInputBuffers, &bakeBuffers ) );
        }

        // Allocate baking output buffers.
        raii<CUdeviceptr>                            d_temp;
        raii<CUdeviceptr>                            d_ommOutput;
        raii<CUdeviceptr, OptixOpacityMicromapDesc>           d_ommDescs;
        raii<CUdeviceptr, OptixOpacityMicromapUsageCount>     d_usageCounts;
        raii<CUdeviceptr, OptixOpacityMicromapHistogramEntry> d_histogramEntries;

        check( d_ommIndices.alloc( bakeInputBuffers.indexBufferSizeInBytes ) );
        check( d_usageCounts.alloc( bakeInputBuffers.numMicromapUsageCounts ) );
        check( d_ommOutput.alloc( bakeBuffers.outputBufferSizeInBytes ) );
        check( d_ommDescs.alloc( bakeBuffers.numMicromapDescs ) );
        check( d_histogramEntries.alloc( bakeBuffers.numMicromapHistogramEntries ) );
        check( d_temp.alloc( bakeBuffers.tempBufferSizeInBytes ) );

        bakeInputBuffers.indexBuffer = d_ommIndices.get();
        bakeInputBuffers.micromapUsageCountsBuffer = d_usageCounts.get();
        bakeBuffers.outputBuffer = d_ommOutput.get();
        bakeBuffers.perMicromapDescBuffer = d_ommDescs.get();
        bakeBuffers.micromapHistogramEntriesBuffer = d_histogramEntries.get();
        bakeBuffers.tempBuffer = d_temp.get();

        // Execute the baking.
        check( cuOmmBaking::BakeOpacityMicromaps( &ommOptions, 1, &bakeInput, &bakeInputBuffers, &bakeBuffers, 0 ) );

        std::vector<OptixOpacityMicromapHistogramEntry> h_histogram;
        // Download data that is needed on the host to build the OptiX Opacity Micromap Array.
        usageOMM.resize( bakeInputBuffers.numMicromapUsageCounts );
        h_histogram.resize( bakeBuffers.numMicromapHistogramEntries );

        d_usageCounts.download( usageOMM );
        d_histogramEntries.download( h_histogram );

        // Free buffers that were inputs to the baker, but not needed for OptiX Opacity Micromap Array and GAS builds.
        check( d_temp.free() );
        check( d_usageCounts.free() );
        check( d_histogramEntries.free() );

        // Build OptiX Opacity Micromap Array.
        OptixMicromapBufferSizes            ommArraySizes = {};
        OptixOpacityMicromapArrayBuildInput ommArrayInput = {};

        ommArrayInput.micromapHistogramEntries = h_histogram.data();
        ommArrayInput.numMicromapHistogramEntries = ( uint32_t )h_histogram.size();
        ommArrayInput.perMicromapDescStrideInBytes = sizeof( OptixOpacityMicromapDesc );
        check( optixOpacityMicromapArrayComputeMemoryUsage( context, &ommArrayInput, &ommArraySizes ) );

        OptixMicromapBuffers ommArrayBuffers = {};
        ommArrayBuffers.outputSizeInBytes = ommArraySizes.outputSizeInBytes;
        ommArrayBuffers.tempSizeInBytes = ommArraySizes.tempSizeInBytes;
        ommArrayInput.perMicromapDescBuffer = bakeBuffers.perMicromapDescBuffer;
        ommArrayInput.inputBuffer = bakeBuffers.outputBuffer;

        check( d_ommArray.alloc( ommArrayBuffers.outputSizeInBytes ) );
        check( d_temp.alloc( ommArrayBuffers.tempSizeInBytes ) );

        ommArrayBuffers.output = d_ommArray.get();
        ommArrayBuffers.temp = d_temp.get();

        check( optixOpacityMicromapArrayBuild( context, 0, &ommArrayInput, &ommArrayBuffers ) );

        // Free the input buffers to the  OptiX Opacity Micromap Array build.
        check( d_ommOutput.free() );
        check( d_ommDescs.free() );
        check( d_temp.free() );

        // Build OptiX OpacityMicromap.
        auto opacityMicromap = std::make_shared<OptixBuildInputOpacityMicromap>();
        opacityMicromap->indexingMode = OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_INDEXED;
        opacityMicromap->indexBuffer = bakeInputBuffers.indexBuffer;
        opacityMicromap->indexSizeInBytes =
            ( bakeBuffers.indexFormat == cuOmmBaking::IndexFormat::I32_UINT ) ? sizeof( uint32_t ) : sizeof( uint16_t );
        opacityMicromap->micromapUsageCounts = usageOMM.data();
        opacityMicromap->numMicromapUsageCounts = usageOMM.size();
        opacityMicromap->opacityMicromapArray = ommArrayBuffers.output;

        return opacityMicromap;
    }

    raii<CUdeviceptr> d_ommArray;
    raii<CUdeviceptr> d_ommIndices;

    std::vector<OptixOpacityMicromapUsageCount> usageOMM;
    std::shared_ptr<OptixBuildInputOpacityMicromap> inputOMM;

    ~MeshObject() {
        node = nullptr;
        // Free the OptiX Opacity Micromap Array.
        check( d_ommArray.free() );
        check( d_ommIndices.free() );
    }
};