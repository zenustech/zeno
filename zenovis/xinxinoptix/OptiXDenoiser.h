/*

 * SPDX-FileCopyrightText: Copyright (c) 2020 - 2024  NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_denoiser_tiling.h>

#include <sutil/Exception.h>

#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <iomanip>
#include <vector>


// Create four channel float OptixImage2D with given dimension. Allocate memory on device and
// Copy data from host memory given in hmem to device if hmem is nonzero.
static OptixImage2D createOptixImage2D( unsigned int width, unsigned int height, const float * hmem = nullptr ) 
{
    OptixImage2D oi;

    const uint64_t frame_byte_size = width * height * sizeof(float4);
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &oi.data ), frame_byte_size ) );
    if( hmem )
    {
        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>( oi.data ),
                    hmem,
                    frame_byte_size,
                    cudaMemcpyHostToDevice
                    ) );
    }
    oi.width              = width;
    oi.height             = height;
    oi.rowStrideInBytes   = width*sizeof(float4);
    oi.pixelStrideInBytes = sizeof(float4);
    oi.format             = OPTIX_PIXEL_FORMAT_FLOAT4;
    return oi;
}

// Copy OptixImage2D from src to dest.
static void copyOptixImage2D( OptixImage2D& dest, const OptixImage2D& src )
{
    CUDA_CHECK( cudaMemcpy( (void*)dest.data, (void*)src.data, src.width * src.height * sizeof( float4 ), cudaMemcpyDeviceToDevice ) );
}

class OptiXDenoiser
{
public:
    struct Data
    {
        uint32_t  width     = 0;
        uint32_t  height    = 0;
        float*    color     = nullptr;
        float*    albedo    = nullptr;
        float*    normal    = nullptr;
        float*    flow      = nullptr;
        float*    flowtrust = nullptr;
        std::vector< float* > aovs;     // input AOVs
        std::vector< float* > outputs;  // denoised beauty, followed by denoised AOVs
    };

    // Initialize the API and push all data to the GPU -- normaly done only once per session.
    // tileWidth, tileHeight: If nonzero, enable tiling with given dimension.
    // kpMode: If enabled, use kernel prediction model even if no AOVs are given.
    // temporalMode: If enabled, use a model for denoising sequences of images.
    // applyFlowMode: Apply flow vectors from current frame to previous image (no denoising).
    void init( const Data&  data,
               unsigned int tileWidth     = 0,
               unsigned int tileHeight    = 0,
               bool         kpMode        = false,
               bool         temporalMode  = false,
               bool         applyFlowMode = false,
               bool         upscale2xMode = false,
               OptixDenoiserAlphaMode alphaMode = OPTIX_DENOISER_ALPHA_MODE_COPY,
               bool         specularMode  = false );

    // Execute the denoiser. In interactive sessions, this would be done once per frame/subframe.
    void exec();

    // Update denoiser input data on GPU from host memory.
    void update( const Data& data );

    // Copy results from GPU to host memory.
    void getResults();

    // Return internal guide layer data for temporal models, if available. Returned memory must be freed.
    void getInternalGuideLayerData( unsigned char** data, size_t* sizeInBytes );

    // Cleanup state, deallocate memory -- normally done only once per render session.
    void finish();

private:
    // --- Test flow vectors: Flow is applied to noisy input image and written back to result.
    // --- No denoising.
    void applyFlow();

private:
    OptixDeviceContext    m_context      = nullptr;
    OptixDenoiser         m_denoiser     = nullptr;
    OptixDenoiserParams   m_params       = {};

    bool                  m_temporalMode;
    bool                  m_applyFlowMode;

    CUdeviceptr           m_intensity    = 0;
    CUdeviceptr           m_avgColor     = 0;
    CUdeviceptr           m_scratch      = 0;
    uint32_t              m_scratch_size = 0;
    CUdeviceptr           m_state        = 0;
    uint32_t              m_state_size   = 0;

    unsigned int          m_tileWidth    = 0;
    unsigned int          m_tileHeight   = 0;
    unsigned int          m_overlap      = 0;

    OptixDenoiserGuideLayer           m_guideLayer = {};
    std::vector< OptixDenoiserLayer > m_layers;
    std::vector< float* >             m_host_outputs;
};

void OptiXDenoiser::init( const Data&  data,
                          unsigned int tileWidth,
                          unsigned int tileHeight,
                          bool         kpMode,
                          bool         temporalMode,
                          bool         applyFlowMode,
                          bool         upscale2xMode,
                          OptixDenoiserAlphaMode alphaMode,
                          bool         specularMode )
{
    SUTIL_ASSERT( data.color  );
    SUTIL_ASSERT( data.outputs.size() >= 1 );
    SUTIL_ASSERT( data.width  );
    SUTIL_ASSERT( data.height );
    SUTIL_ASSERT_MSG( !data.normal || data.albedo, "Currently albedo is required if normal input is given" );
    SUTIL_ASSERT_MSG( ( tileWidth == 0 && tileHeight == 0 ) || ( tileWidth > 0 && tileHeight > 0 ), "tile size must be > 0 for width and height" );

    unsigned int outScale = 1;
    if( upscale2xMode )
    {
        kpMode = true;
        outScale = 2;
    }

    m_host_outputs = data.outputs;
    m_temporalMode = temporalMode;
    m_applyFlowMode = applyFlowMode;

    m_tileWidth  = tileWidth > 0 ? tileWidth : data.width;
    m_tileHeight = tileHeight > 0 ? tileHeight : data.height;

    //
    // Initialize CUDA and create OptiX context
    //
    {
        // Initialize CUDA
        CUDA_CHECK( cudaFree( nullptr ) );

        CUcontext cu_ctx = nullptr;  // zero means take the current context
        OPTIX_CHECK( optixInit() );
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction       = &context_log_cb;
        options.logCallbackLevel          = 4;
        OPTIX_CHECK( optixDeviceContextCreate( cu_ctx, &options, &m_context ) );
    }

    //
    // Create denoiser
    //
    {
        /*****
        // Load user provided model if model.bin is present in the currrent directory,
        // configuration of filename not done here.
        std::ifstream file( "model.bin" );
        if ( file.good() ) {
            std::stringstream source_buffer;
            source_buffer << file.rdbuf();
            OPTIX_CHECK( optixDenoiserCreateWithUserModel( m_context, (void*)source_buffer.str().c_str(), source_buffer.str().size(), &m_denoiser ) );
        }
        else
        *****/
        {
            OptixDenoiserOptions options = {};
            options.guideAlbedo  = data.albedo ? 1 : 0;
            options.guideNormal  = data.normal ? 1 : 0;
            options.denoiseAlpha = alphaMode;

            OptixDenoiserModelKind modelKind;
            if( upscale2xMode )
                modelKind = temporalMode ? OPTIX_DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X : OPTIX_DENOISER_MODEL_KIND_UPSCALE2X;
            else if( kpMode || data.aovs.size() > 0 )
                modelKind = temporalMode ? OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV : OPTIX_DENOISER_MODEL_KIND_AOV;
            else
                modelKind = temporalMode ? OPTIX_DENOISER_MODEL_KIND_TEMPORAL : OPTIX_DENOISER_MODEL_KIND_HDR;
            OPTIX_CHECK( optixDenoiserCreate( m_context, modelKind, &options, &m_denoiser ) );
        }
    }


    //
    // Allocate device memory for denoiser
    //
    {
        OptixDenoiserSizes denoiser_sizes;

        OPTIX_CHECK( optixDenoiserComputeMemoryResources(
                    m_denoiser,
                    m_tileWidth,
                    m_tileHeight,
                    &denoiser_sizes
                    ) );

        if( tileWidth == 0 )
        {
            m_scratch_size = static_cast<uint32_t>( denoiser_sizes.withoutOverlapScratchSizeInBytes );
            m_overlap = 0;
        }
        else
        {
            m_scratch_size = static_cast<uint32_t>( denoiser_sizes.withOverlapScratchSizeInBytes );
            m_overlap = denoiser_sizes.overlapWindowSizeInPixels;
        }

        if( data.aovs.size() == 0 && kpMode == false )
        {
            CUDA_CHECK( cudaMalloc(
                        reinterpret_cast<void**>( &m_intensity ),
                        sizeof( float )
                        ) );
        }
        else
        {
            CUDA_CHECK( cudaMalloc(
                        reinterpret_cast<void**>( &m_avgColor ),
                        3 * sizeof( float )
                        ) );
        }

        CUDA_CHECK( cudaMalloc(
                    reinterpret_cast<void**>( &m_scratch ),
                    m_scratch_size 
                    ) );

        CUDA_CHECK( cudaMalloc(
                    reinterpret_cast<void**>( &m_state ),
                    denoiser_sizes.stateSizeInBytes
                    ) );

        m_state_size = static_cast<uint32_t>( denoiser_sizes.stateSizeInBytes );

        OptixDenoiserLayer layer = {};
        layer.input  = createOptixImage2D( data.width, data.height, data.color );
        layer.output = createOptixImage2D( outScale * data.width, outScale * data.height );

        if( m_temporalMode )
        {
            layer.previousOutput = createOptixImage2D( outScale * data.width, outScale * data.height );

            // This is the first frame, create zero motion vector image.
            void* flowmem;
            CUDA_CHECK( cudaMalloc( &flowmem, data.width * data.height * sizeof( float4 ) ) );
            CUDA_CHECK( cudaMemset( flowmem, 0, data.width * data.height * sizeof(float4) ) );
            m_guideLayer.flow = {(CUdeviceptr)flowmem, data.width, data.height, (unsigned int)(data.width * sizeof( float4 )), (unsigned int)sizeof( float4 ), OPTIX_PIXEL_FORMAT_FLOAT4 };

            // Set first frame previous output to noisy input image from first frame
            if( !upscale2xMode )
                copyOptixImage2D( layer.previousOutput, layer.input );

            // Internal guide layer memory set to zero for first frame.
            void* internalMemIn  = 0;
            void* internalMemOut = 0;
            size_t internalSize = outScale * data.width * outScale * data.height * denoiser_sizes.internalGuideLayerPixelSizeInBytes;
            CUDA_CHECK( cudaMalloc( &internalMemIn, internalSize ) );
            CUDA_CHECK( cudaMalloc( &internalMemOut, internalSize ) );
            CUDA_CHECK( cudaMemset( internalMemIn, 0, internalSize ) );

            m_guideLayer.previousOutputInternalGuideLayer.data   = (CUdeviceptr)internalMemIn;
            m_guideLayer.previousOutputInternalGuideLayer.width  = outScale * data.width;
            m_guideLayer.previousOutputInternalGuideLayer.height = outScale * data.height;
            m_guideLayer.previousOutputInternalGuideLayer.pixelStrideInBytes = unsigned( denoiser_sizes.internalGuideLayerPixelSizeInBytes );
            m_guideLayer.previousOutputInternalGuideLayer.rowStrideInBytes = m_guideLayer.previousOutputInternalGuideLayer.width * m_guideLayer.previousOutputInternalGuideLayer.pixelStrideInBytes;
            m_guideLayer.previousOutputInternalGuideLayer.format = OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER;

            m_guideLayer.outputInternalGuideLayer = m_guideLayer.previousOutputInternalGuideLayer;
            m_guideLayer.outputInternalGuideLayer.data = (CUdeviceptr)internalMemOut;

            if( data.flowtrust )
            {
                void* ftmem;
                CUDA_CHECK( cudaMalloc( &ftmem, data.width * data.height * sizeof( float4 ) ) );
                CUDA_CHECK( cudaMemset( ftmem, 0, data.width * data.height * sizeof( float4 ) ) );
                m_guideLayer.flowTrustworthiness = {(CUdeviceptr)ftmem, data.width, data.height, (unsigned int)(data.width * sizeof( float4 )), (unsigned int)sizeof( float4 ), OPTIX_PIXEL_FORMAT_FLOAT4 };
            }
        }
        m_layers.push_back( layer );

        if( data.albedo )
            m_guideLayer.albedo = createOptixImage2D( data.width, data.height, data.albedo );
        if( data.normal )
            m_guideLayer.normal = createOptixImage2D( data.width, data.height, data.normal );

        for( size_t i=0; i < data.aovs.size(); i++ )
        {
            layer = {};
            layer.input  = createOptixImage2D( data.width, data.height, data.aovs[i] );
            layer.output = createOptixImage2D( outScale * data.width, outScale * data.height );
            if( m_temporalMode )
            {
                // First frame initializaton.
                layer.previousOutput = createOptixImage2D( outScale * data.width, outScale * data.height );
                if( !upscale2xMode )
                    copyOptixImage2D( layer.previousOutput, layer.input );
                if( specularMode )
                    layer.type = OPTIX_DENOISER_AOV_TYPE_SPECULAR;
            }
            m_layers.push_back( layer );
        }
    }

    //
    // Setup denoiser
    //
    {
        OPTIX_CHECK( optixDenoiserSetup(
                    m_denoiser,
                    nullptr,  // CUDA stream
                    m_tileWidth + 2 * m_overlap,
                    m_tileHeight + 2 * m_overlap,
                    m_state,
                    m_state_size,
                    m_scratch,
                    m_scratch_size
                    ) );


        m_params.hdrIntensity    = m_intensity;
        m_params.hdrAverageColor = m_avgColor;
        m_params.blendFactor     = 0.0f;
        m_params.temporalModeUsePreviousLayers = 0;
    }
}

void OptiXDenoiser::update( const Data& data )
{
    SUTIL_ASSERT( data.color  );
    SUTIL_ASSERT( data.outputs.size() >= 1 );
    SUTIL_ASSERT( data.width  );
    SUTIL_ASSERT( data.height );
    SUTIL_ASSERT_MSG( !data.normal || data.albedo, "Currently albedo is required if normal input is given" );

    m_host_outputs = data.outputs;

    CUDA_CHECK( cudaMemcpy( (void*)m_layers[0].input.data, data.color, data.width * data.height * sizeof( float4 ), cudaMemcpyHostToDevice ) );

    if( m_temporalMode )
        CUDA_CHECK( cudaMemcpy( (void*)m_guideLayer.flow.data, data.flow, data.width * data.height * sizeof( float4 ), cudaMemcpyHostToDevice ) );

    if( data.albedo )
        CUDA_CHECK( cudaMemcpy( (void*)m_guideLayer.albedo.data, data.albedo, data.width * data.height * sizeof( float4 ), cudaMemcpyHostToDevice ) );

    if( data.normal )
        CUDA_CHECK( cudaMemcpy( (void*)m_guideLayer.normal.data, data.normal, data.width * data.height * sizeof( float4 ), cudaMemcpyHostToDevice ) );

    if( data.flowtrust )
        CUDA_CHECK( cudaMemcpy( (void*)m_guideLayer.flowTrustworthiness.data, data.flowtrust, data.width * data.height * sizeof( float4 ), cudaMemcpyHostToDevice ) );

    for( size_t i=0; i < data.aovs.size(); i++ )
        CUDA_CHECK( cudaMemcpy( (void*)m_layers[1+i].input.data, data.aovs[i], data.width * data.height * sizeof( float4 ), cudaMemcpyHostToDevice ) );

    if( m_temporalMode )
    {
        OptixImage2D temp = m_guideLayer.previousOutputInternalGuideLayer;
        m_guideLayer.previousOutputInternalGuideLayer = m_guideLayer.outputInternalGuideLayer;
        m_guideLayer.outputInternalGuideLayer = temp;

        for( size_t i=0; i < m_layers.size(); i++ )
        {
            temp = m_layers[i].previousOutput;
            m_layers[i].previousOutput = m_layers[i].output;
            m_layers[i].output = temp;
        }
    }
    m_params.temporalModeUsePreviousLayers = 1;
}

void OptiXDenoiser::exec()
{
    if( m_intensity )
    {
        OPTIX_CHECK( optixDenoiserComputeIntensity(
                    m_denoiser,
                    nullptr, // CUDA stream
                    &m_layers[0].input,
                    m_intensity,
                    m_scratch,
                    m_scratch_size
                    ) );
    }
    
    if( m_avgColor )
    {
        OPTIX_CHECK( optixDenoiserComputeAverageColor(
                    m_denoiser,
                    nullptr, // CUDA stream
                    &m_layers[0].input,
                    m_avgColor,
                    m_scratch,
                    m_scratch_size
                    ) );
    }

    if( m_applyFlowMode )
    {
        applyFlow();
    }
    else
    {
        // This sample is always using tiling mode. 
#if 0
        OPTIX_CHECK( optixDenoiserInvoke(
                    m_denoiser,
                    nullptr, // CUDA stream
                    &m_params,
                    m_state,
                    m_state_size,
                    &m_guideLayer,
                    m_layers.data(),
                    static_cast<unsigned int>( m_layers.size() ),
                    0, // input offset X
                    0, // input offset y
                    m_scratch,
                    m_scratch_size
                    ) );
#else
        OPTIX_CHECK( optixUtilDenoiserInvokeTiled(
                    m_denoiser,
                    nullptr, // CUDA stream
                    &m_params,
                    m_state,
                    m_state_size,
                    &m_guideLayer,
                    m_layers.data(),
                    static_cast<unsigned int>( m_layers.size() ),
                    m_scratch,
                    m_scratch_size,
                    m_overlap,
                    m_tileWidth,
                    m_tileHeight
                    ) );
#endif
    }
    CUDA_SYNC_CHECK();
}

inline float catmull_rom(
    float       p[4],
    float       t)
{
    return p[1] + 0.5f * t * ( p[2] - p[0] + t * ( 2.f * p[0] - 5.f * p[1] + 4.f * p[2] - p[3] + t * ( 3.f * ( p[1] - p[2]) + p[3] - p[0] ) ) );
}

// Apply flow to image at given pixel position (using bilinear interpolation), write back RGB result.
static void addFlow(
    float4*             result,
    const float4*       image,
    const float4*       flow,
    unsigned int        width,
    unsigned int        height,
    unsigned int        x,
    unsigned int        y )
{
    float dst_x = float( x ) - flow[x + y * width].x;
    float dst_y = float( y ) - flow[x + y * width].y;

    float x0 = dst_x - 1.f;
    float y0 = dst_y - 1.f;

    float r[4][4], g[4][4], b[4][4];
    for (int j=0; j < 4; j++)
    {
        for (int k=0; k < 4; k++)
        {
            int tx = static_cast<int>( x0 ) + k;
            if( tx < 0 )
                tx = 0;
            else if( tx >= (int)width )
                tx = width - 1;

            int ty = static_cast<int>( y0 ) + j;
            if( ty < 0 )
                ty = 0;
            else if( ty >= (int)height )
                ty = height - 1;

            r[j][k] = image[tx + ty * width].x;
            g[j][k] = image[tx + ty * width].y;
            b[j][k] = image[tx + ty * width].z;
        }
    }
    float tx = dst_x <= 0.f ? 0.f : dst_x - floorf( dst_x );

    r[0][0] = catmull_rom( r[0], tx );
    r[0][1] = catmull_rom( r[1], tx );
    r[0][2] = catmull_rom( r[2], tx );
    r[0][3] = catmull_rom( r[3], tx );

    g[0][0] = catmull_rom( g[0], tx );
    g[0][1] = catmull_rom( g[1], tx );
    g[0][2] = catmull_rom( g[2], tx );
    g[0][3] = catmull_rom( g[3], tx );

    b[0][0] = catmull_rom( b[0], tx );
    b[0][1] = catmull_rom( b[1], tx );
    b[0][2] = catmull_rom( b[2], tx );
    b[0][3] = catmull_rom( b[3], tx );

    float ty = dst_y <= 0.f ? 0.f : dst_y - floorf( dst_y );

    result[y * width + x].x = catmull_rom( r[0], ty );
    result[y * width + x].y = catmull_rom( g[0], ty );
    result[y * width + x].z = catmull_rom( b[0], ty );
}

// Apply flow from current frame to the previous noisy image. 
void OptiXDenoiser::applyFlow()
{
    if( m_layers.size() == 0 )
        return;

    const uint64_t frame_size = m_layers[0].output.width * m_layers[0].output.height;
    const uint64_t frame_byte_size = frame_size * sizeof(float4);

    const float4* device_flow = (float4*)m_guideLayer.flow.data;
    if( !device_flow )
        return;
    float4* flow = new float4[ frame_size ];
    CUDA_CHECK( cudaMemcpy( flow, device_flow, frame_byte_size, cudaMemcpyDeviceToHost ) );

    float4* image = new float4[ frame_size ];
    float4* result = new float4[frame_size];

    for( size_t i=0; i < m_layers.size(); i++ )
    {
        CUDA_CHECK( cudaMemcpy( image, (float4*)m_layers[i].previousOutput.data, frame_byte_size, cudaMemcpyDeviceToHost ) );
        for( unsigned int y=0; y < m_layers[i].previousOutput.height; y++ )
            for( unsigned int x=0; x < m_layers[i].previousOutput.width; x++ )
                addFlow( result, image, flow, m_layers[i].previousOutput.width, m_layers[i].previousOutput.height, x, y );

        CUDA_CHECK( cudaMemcpy( (void*)m_layers[i].output.data, result, frame_byte_size, cudaMemcpyHostToDevice ) );
    }
    delete[] result;
    delete[] image;
    delete[] flow;
}

void OptiXDenoiser::getResults()
{
    const uint64_t frame_byte_size = m_layers[0].output.width*m_layers[0].output.height*sizeof(float4);
    for( size_t i=0; i < m_layers.size(); i++ )
    {
        CUDA_CHECK( cudaMemcpy(
                    m_host_outputs[i],
                    reinterpret_cast<void*>( m_layers[i].output.data ),
                    frame_byte_size,
                    cudaMemcpyDeviceToHost
                    ) );

        // We start with a noisy image in this mode for each frame, otherwise the warped images would accumulate.
        if( m_applyFlowMode )
            CUDA_CHECK( cudaMemcpy( (void*)m_layers[i].output.data,
                                    reinterpret_cast<void*>( m_layers[i].input.data ),
                                    frame_byte_size, cudaMemcpyDeviceToHost ) );
    }
}

void OptiXDenoiser::getInternalGuideLayerData( unsigned char** data, size_t* sizeInBytes )
{
    *data        = 0;
    *sizeInBytes = 0;

    if( m_guideLayer.outputInternalGuideLayer.data )
    {
        *sizeInBytes = m_guideLayer.outputInternalGuideLayer.width * m_guideLayer.outputInternalGuideLayer.height * m_guideLayer.outputInternalGuideLayer.pixelStrideInBytes;
        *data = new unsigned char[ *sizeInBytes ];
        CUDA_CHECK( cudaMemcpy( *data, (void*)m_guideLayer.outputInternalGuideLayer.data, *sizeInBytes, cudaMemcpyDeviceToHost ) );
    }
}

void OptiXDenoiser::finish() 
{
    // Cleanup resources
    optixDenoiserDestroy( m_denoiser );
    optixDeviceContextDestroy( m_context );

    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_intensity)) );
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_avgColor)) );
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_scratch)) );
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_state)) );
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_guideLayer.albedo.data)) );
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_guideLayer.normal.data)) );
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_guideLayer.flow.data)) );
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_guideLayer.flowTrustworthiness.data)) );
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_guideLayer.previousOutputInternalGuideLayer.data)) );
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_guideLayer.outputInternalGuideLayer.data)) );
    for( size_t i=0; i < m_layers.size(); i++ )
    {
        CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_layers[i].input.data) ) );
        CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_layers[i].output.data) ) ); 
        CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_layers[i].previousOutput.data) ) );
    }
    m_layers.clear();
}
