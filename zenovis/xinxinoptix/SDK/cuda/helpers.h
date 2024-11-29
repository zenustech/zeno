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

#include <vector_types.h>
#include <sutil/vec_math.h>


__forceinline__ __device__ float3 toSRGB( const float3& c, float gamma )
{
    float  invGamma = 1.0f / gamma;
    float3 powed    = make_float3( powf( c.x, invGamma ), powf( c.y, invGamma ), powf( c.z, invGamma ) );

    return powed;
}

__forceinline__ __device__ unsigned char quantizeUnsigned8Bits( float x )
{
    x = clamp( x, 0.0f, 1.0f );
    enum { N = (1 << 8) - 1, Np1 = (1 << 8) };
    return (unsigned char)min((unsigned int)(x * (float)Np1), (unsigned int)N);
}

__forceinline__ __device__ uchar4 makeSRGB( const float3& c, float gamma=2.2f, float dither=0.0f)
{
    // first apply gamma, then convert to unsigned char
    float3 srgb = toSRGB( c, gamma );
    //srgb += make_float3(dither);
    srgb += make_float3(dither, 1.0-dither, dither)/255;
    srgb = clamp( srgb, 0.0f, 1.0f );

    return make_uchar4( quantizeUnsigned8Bits( srgb.x ), quantizeUnsigned8Bits( srgb.y ), quantizeUnsigned8Bits( srgb.z ), 255u );
}
__forceinline__ __device__ uchar4 make_color( const float4& c )
{
    return makeSRGB( make_float3( c.x, c.y, c.z ) );
}
