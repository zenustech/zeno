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


#include <Exception.h>
#include <PPMLoader.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>


//-----------------------------------------------------------------------------
//
//  PPMLoader class definition
//
//-----------------------------------------------------------------------------


PPMLoader::PPMLoader( const std::string& filename, const bool vflip )
    : m_nx( 0u )
    , m_ny( 0u )
    , m_max_val( 0u )
    , m_raster( 0 )
    , m_is_ascii( false )
{
    if( filename.empty() )
        return;

    size_t      pos;
    std::string extension;
    if( ( pos = filename.find_last_of( '.' ) ) != std::string::npos )
        extension = filename.substr( pos );
    if( !( extension == ".ppm" || extension == ".PPM" ) )
    {
        std::cerr << "PPMLoader( '" << filename << "' ) non-ppm file extension given '" << extension << "'" << std::endl;
        return;
    }

    // Open file
    try
    {
        std::ifstream file_in( filename.c_str(), std::ifstream::in | std::ifstream::binary );
        if( !file_in )
        {
            std::cerr << "PPMLoader( '" << filename << "' ) failed to open file." << std::endl;
            return;
        }

        // Check magic number to make sure we have an ascii or binary PPM
        std::string line, magic_number;
        getLine( file_in, line );
        std::istringstream iss1( line );
        iss1 >> magic_number;
        if( magic_number != "P6" && magic_number != "P3" )
        {
            std::cerr << "PPMLoader( '" << filename << "' ) unknown magic number: " << magic_number
                      << ".  Only P3 and P6 supported." << std::endl;
            return;
        }
        if( magic_number == "P3" )
        {
            m_is_ascii = true;
        }

        // width, height
        getLine( file_in, line );
        std::istringstream iss2( line );
        iss2 >> m_nx >> m_ny;

        // max channel value
        getLine( file_in, line );
        std::istringstream iss3( line );
        iss3 >> m_max_val;

        m_raster = new( std::nothrow ) unsigned char[m_nx * m_ny * 3];
        if( m_is_ascii )
        {
            unsigned int num_elements = m_nx * m_ny * 3;
            unsigned int count        = 0;

            while( count < num_elements )
            {
                getLine( file_in, line );
                std::istringstream iss( line );

                while( iss.good() )
                {
                    unsigned int c;
                    iss >> c;
                    m_raster[count++] = static_cast<unsigned char>( c );
                }
            }
        }
        else
        {
            file_in.read( reinterpret_cast<char*>( m_raster ), m_nx * m_ny * 3 );
        }

        if( vflip )
        {
            for( unsigned int y2 = m_ny - 1, y = 0; y < y2; y2--, y++ )
            {
                for( unsigned int x = 0; x < m_nx * 3; x++ )
                {
                    unsigned char temp          = m_raster[y * m_nx * 3 + x];
                    m_raster[y * m_nx * 3 + x]  = m_raster[y2 * m_nx * 3 + x];
                    m_raster[y2 * m_nx * 3 + x] = temp;
                }
            }
        }
    }
    catch( ... )
    {
        std::cerr << "PPMLoader( '" << filename << "' ) failed to load" << std::endl;
        m_raster = 0;
    }
}


PPMLoader::~PPMLoader()
{
    if( m_raster )
        delete[] m_raster;
}


bool PPMLoader::failed() const
{
    return m_raster == 0;
}


unsigned int PPMLoader::width() const
{
    return m_nx;
}


unsigned int PPMLoader::height() const
{
    return m_ny;
}


unsigned char* PPMLoader::raster() const
{
    return m_raster;
}


void PPMLoader::getLine( std::ifstream& file_in, std::string& s )
{
    for( ;; )
    {
        if( !std::getline( file_in, s ) )
            return;
        std::string::size_type index = s.find_first_not_of( "\n\r\t " );
        if( index != std::string::npos && s[index] != '#' )
            break;
    }
}


//-----------------------------------------------------------------------------
//
//  Utility functions
//
//-----------------------------------------------------------------------------
float clamp( float f, float a, float b )
{
    return std::max( a, std::min( f, b ) );
}

sutil::Texture PPMLoader::loadTexture( const float3& default_color, cudaTextureDesc* tex_desc )
{
    std::vector<unsigned char> buffer;
    const unsigned int         nx = width();
    const unsigned int         ny = height();
    if( failed() )
    {
        buffer.resize( 4 );

        // Create buffer with single texel set to default_color
        constexpr float gamma = 2.2f;
        // multiplying by 255.5 and rounding down is a good trade-off when compressing a float to [0,255].
        buffer[0] = static_cast<unsigned char>( (int)( powf( clamp( default_color.x, 0.0f, 1.0f ), 1.0f / gamma ) * 255.5f ) );
        buffer[1] = static_cast<unsigned char>( (int)( powf( clamp( default_color.y, 0.0f, 1.0f ), 1.0f / gamma ) * 255.5f ) );
        buffer[2] = static_cast<unsigned char>( (int)( powf( clamp( default_color.z, 0.0f, 1.0f ), 1.0f / gamma ) * 255.5f ) );
        buffer[3] = 255;
    }
    else
    {
        buffer.resize( 4 * nx * ny );

        for( unsigned int i = 0; i < nx; ++i )
        {
            for( unsigned int j = 0; j < ny; ++j )
            {

                unsigned int ppm_index = ( ( ny - j - 1 ) * nx + i ) * 3;
                unsigned int buf_index = ( (j)*nx + i ) * 4;

                buffer[buf_index + 0] = raster()[ppm_index + 0];
                buffer[buf_index + 1] = raster()[ppm_index + 1];
                buffer[buf_index + 2] = raster()[ppm_index + 2];
                buffer[buf_index + 3] = 255;
            }
        }
    }

    // Allocate CUDA array in device memory
    int32_t               pitch        = nx * 4 * sizeof( unsigned char );
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar4>();

    cudaArray_t cuda_array = nullptr;
    CUDA_CHECK( cudaMallocArray( &cuda_array, &channel_desc, nx, ny ) );
    CUDA_CHECK( cudaMemcpy2DToArray( cuda_array, 0, 0, buffer.data(), pitch, pitch, ny, cudaMemcpyHostToDevice ) );

    // Create texture object
    cudaResourceDesc res_desc = {};
    res_desc.resType          = cudaResourceTypeArray;
    res_desc.res.array.array  = cuda_array;

    cudaTextureDesc default_tex_desc = {};
    if( tex_desc == nullptr )
    {
        default_tex_desc.addressMode[0]      = cudaAddressModeWrap;
        default_tex_desc.addressMode[1]      = cudaAddressModeWrap;
        default_tex_desc.filterMode          = cudaFilterModeLinear;
        default_tex_desc.readMode            = cudaReadModeNormalizedFloat;
        default_tex_desc.normalizedCoords    = 1;
        default_tex_desc.maxAnisotropy       = 1;
        default_tex_desc.maxMipmapLevelClamp = 99;
        default_tex_desc.minMipmapLevelClamp = 0;
        default_tex_desc.mipmapFilterMode    = cudaFilterModePoint;
        default_tex_desc.borderColor[0]      = 1.0f;
        default_tex_desc.sRGB                = 1;  // ppm files are in sRGB space according to specification

        tex_desc = &default_tex_desc;
    }

    // Create texture object
    cudaTextureObject_t cuda_tex = 0;
    CUDA_CHECK( cudaCreateTextureObject( &cuda_tex, &res_desc, tex_desc, nullptr ) );

    sutil::Texture ppm_texture = {cuda_array, cuda_tex};
    return ppm_texture;
}


//-----------------------------------------------------------------------------
//
//  Utility functions
//
//-----------------------------------------------------------------------------

sutil::Texture loadPPMTexture( const std::string& filename, const float3& default_color, cudaTextureDesc* tex_desc )
{
    PPMLoader ppm( filename );
    return ppm.loadTexture( default_color, tex_desc );
}

