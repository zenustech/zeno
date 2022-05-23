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

#include <sutil/Preprocessor.h>

#include <stdint.h>

class StaticWorkDistribution
{
public:
    SUTIL_INLINE SUTIL_HOSTDEVICE void setRasterSize( int width, int height )
    {
        m_width = width;
        m_height = height;
    }


    SUTIL_INLINE SUTIL_HOSTDEVICE void setNumGPUs( int32_t num_gpus )
    {
        m_num_gpus = num_gpus;
    }


    SUTIL_INLINE SUTIL_HOSTDEVICE int32_t numSamples( int32_t gpu_idx )
    {
        const int tile_strip_width  = TILE_WIDTH*m_num_gpus;
        const int tile_strip_height = TILE_HEIGHT;
        const int num_tile_strip_cols = m_width /tile_strip_width  + ( m_width %tile_strip_width  == 0 ? 0 : 1 );
        const int num_tile_strip_rows = m_height/tile_strip_height + ( m_height%tile_strip_height == 0 ? 0 : 1 );
        return num_tile_strip_rows*num_tile_strip_cols*TILE_WIDTH*TILE_HEIGHT;
    }


    SUTIL_INLINE SUTIL_HOSTDEVICE int2 getSamplePixel( int32_t gpu_idx, int32_t sample_idx )
    {
        const int tile_strip_width  = TILE_WIDTH*m_num_gpus;
        const int tile_strip_height = TILE_HEIGHT;
        const int num_tile_strip_cols = m_width /tile_strip_width + ( m_width % tile_strip_width == 0 ? 0 : 1 );

        const int tile_strip_idx     = sample_idx / (TILE_WIDTH*TILE_HEIGHT );
        const int tile_strip_y       = tile_strip_idx / num_tile_strip_cols;
        const int tile_strip_x       = tile_strip_idx - tile_strip_y * num_tile_strip_cols;
        const int tile_strip_x_start = tile_strip_x * tile_strip_width;
        const int tile_strip_y_start = tile_strip_y * tile_strip_height;

        const int tile_pixel_idx     = sample_idx - ( tile_strip_idx * TILE_WIDTH*TILE_HEIGHT );
        const int tile_pixel_y       = tile_pixel_idx / TILE_WIDTH;
        const int tile_pixel_x       = tile_pixel_idx - tile_pixel_y * TILE_WIDTH;

        const int tile_offset_x      = ( gpu_idx + tile_strip_y % m_num_gpus ) % m_num_gpus * TILE_WIDTH;

        const int pixel_y = tile_strip_y_start + tile_pixel_y;
        const int pixel_x = tile_strip_x_start + tile_pixel_x + tile_offset_x ;
        return make_int2( pixel_x, pixel_y );
    }


private:
    int32_t m_num_gpus = 0;
    int32_t m_width    = 0;
    int32_t m_height   = 0;

    static const int32_t TILE_WIDTH  = 8;
    static const int32_t TILE_HEIGHT = 4;
};
