//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//


#pragma once

#include "sutilapi.h"
#include "sampleConfig.h"

#include <cuda_runtime.h>
#include <vector_types.h>

#include <cstdlib>
#include <chrono>
#include <vector>

struct GLFWwindow;

// Some helper macros to stringify the sample's name that comes in as a define
#define OPTIX_STRINGIFY2(name) #name
#define OPTIX_STRINGIFY(name) OPTIX_STRINGIFY2(name)
#define OPTIX_SAMPLE_NAME OPTIX_STRINGIFY(OPTIX_SAMPLE_NAME_DEFINE)
#define OPTIX_SAMPLE_DIR OPTIX_STRINGIFY(OPTIX_SAMPLE_DIR_DEFINE)

namespace sutil
{

enum BufferImageFormat
{
    UNSIGNED_BYTE4,
    FLOAT4,
    FLOAT3
};

struct ImageBuffer
{
    void* data =      nullptr;
    unsigned int      width = 0;
    unsigned int      height = 0;
    BufferImageFormat pixel_format;
};

struct Texture
{
    cudaArray_t         array;
    cudaTextureObject_t texture;
};

// Return a path to a sample data file, or NULL if the file cannot be located.
// The pointer returned may point to a static array.
SUTILAPI const char* sampleDataFilePath( const char* relativeFilePath );

// Return a path to a sample file inside a sub directory, or NULL if the file cannot be located.
// The pointer returned may point to a static array.
SUTILAPI const char* sampleFilePath( const char* relativeSubDir, const char* relativePath );

SUTILAPI size_t pixelFormatSize( BufferImageFormat format );

// Create a cudaTextureObject_t for the given image file.  If the filename is
// empty or if loading the file fails, return 1x1 texture with default color.
SUTILAPI Texture loadTexture( const char* filename, float3 default_color, cudaTextureDesc* tex_desc = nullptr );

// Floating point image buffers (see BufferImageFormat above) are assumed to be
// linear and will be converted to sRGB when writing to a file format with 8
// bits per channel.  This can be skipped if disable_srgb is set to true.
// Image buffers with format UNSIGNED_BYTE4 are assumed to be in sRGB already
// and will be written like that.
SUTILAPI void        saveImage( const char* filename, const ImageBuffer& buffer, bool disable_srgb );
SUTILAPI ImageBuffer loadImage( const char* filename, int32_t force_components = 0 );

SUTILAPI void displayBufferWindow( const char* argv, const ImageBuffer& buffer );


SUTILAPI void        initGL();
SUTILAPI void        initGLFW();
SUTILAPI GLFWwindow* initGLFW( const char* window_title, int width, int height );
SUTILAPI void        initImGui( GLFWwindow* window );
SUTILAPI GLFWwindow* initUI( const char* window_title, int width, int height );
SUTILAPI void        cleanupUI( GLFWwindow* window );

SUTILAPI void        beginFrameImGui();
SUTILAPI void        endFrameImGui();

// Display frames per second, where the OpenGL context
// is managed by the caller.
SUTILAPI void displayFPS( unsigned total_frame_count );

SUTILAPI void displayStats( std::chrono::duration<double>& state_update_time,
                            std::chrono::duration<double>& render_time,
                            std::chrono::duration<double>& display_time );

// Display a short string starting at x,y.
SUTILAPI void displayText( const char* text, float x, float y );

// Blocking sleep call
SUTILAPI void sleep(
        int seconds );                      // Number of seconds to sleep


// Parse the string of the form <width>x<height> and return numeric values.
SUTILAPI void parseDimensions(
        const char* arg,                    // String of form <width>x<height>
        int& width,                         // [out] width
        int& height );                      // [in]  height


SUTILAPI void calculateCameraVariables(
        float3 eye,
        float3 lookat,
        float3 up,
        float  fov,
        float  aspect_ratio,
        float3& U,
        float3& V,
        float3& W,
        bool fov_is_vertical );

// Get current time in seconds for benchmarking/timing purposes.
double SUTILAPI currentTime();

// Get input data, either pre-compiled with NVCC or JIT compiled by NVRTC.
SUTILAPI const char* getInputData( const char* sampleName,  // Name of the sample, used to locate the input file. NULL = only search the common /cuda dir
                                   const char* sampleDir,  // Directory name for the sample (typically the same as the sample name).
                                   const char* filename,      // Cuda C input file name
                                   size_t&     dataSize, 
                                   const char** log = NULL,    // (Optional) pointer to compiler log string. If *log == NULL there is no output. Only valid until the next getInputData call
                                   const std::vector<const char*>& compilerOptions = {CUDA_NVRTC_OPTIONS} );  // Optional vector of compiler options.



// Ensures that width and height have the minimum size to prevent launch errors.
SUTILAPI void ensureMinimumSize(
    int& width,                             // Will be assigned the minimum suitable width if too small.
    int& height);                           // Will be assigned the minimum suitable height if too small.

// Ensures that width and height have the minimum size to prevent launch errors.
SUTILAPI void ensureMinimumSize(
    unsigned& width,                        // Will be assigned the minimum suitable width if too small.
    unsigned& height);                      // Will be assigned the minimum suitable height if too small.

SUTILAPI void reportErrorMessage( const char* message );

} // end namespace sutil

