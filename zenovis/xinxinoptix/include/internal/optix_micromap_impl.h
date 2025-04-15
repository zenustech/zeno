/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


/**
* @file   optix_micromap_impl.h
* @author NVIDIA Corporation
* @brief  OptiX micromap helper functions
*/

#ifndef OPTIX_OPTIX_MICROMAP_IMPL_H
#define OPTIX_OPTIX_MICROMAP_IMPL_H

#ifndef OPTIX_MICROMAP_FUNC
#ifdef __CUDACC__
#define OPTIX_MICROMAP_FUNC __device__
#else
#define OPTIX_MICROMAP_FUNC
#endif
#endif

namespace optix_impl {

/** \addtogroup optix_utilities
@{
*/

#define OPTIX_MICROMAP_INLINE_FUNC OPTIX_MICROMAP_FUNC inline

#ifdef __CUDACC__
// the device implementation of __uint_as_float is declared in cuda_runtime.h
#else
// the host implementation of __uint_as_float
OPTIX_MICROMAP_INLINE_FUNC float __uint_as_float( unsigned int x )
{
    union { float f; unsigned int i; } var;
    var.i = x;
    return var.f;
}
#endif

// Extract even bits
OPTIX_MICROMAP_INLINE_FUNC unsigned int extractEvenBits( unsigned int x )
{
    x &= 0x55555555;
    x = ( x | ( x >> 1 ) ) & 0x33333333;
    x = ( x | ( x >> 2 ) ) & 0x0f0f0f0f;
    x = ( x | ( x >> 4 ) ) & 0x00ff00ff;
    x = ( x | ( x >> 8 ) ) & 0x0000ffff;
    return x;
}


// Calculate exclusive prefix or (log(n) XOR's and SHF's)
OPTIX_MICROMAP_INLINE_FUNC unsigned int prefixEor( unsigned int x )
{
    x ^= x >> 1;
    x ^= x >> 2;
    x ^= x >> 4;
    x ^= x >> 8;
    return x;
}

// Convert distance along the curve to discrete barycentrics
OPTIX_MICROMAP_INLINE_FUNC void index2dbary( unsigned int index, unsigned int& u, unsigned int& v, unsigned int& w )
{
    unsigned int b0 = extractEvenBits( index );
    unsigned int b1 = extractEvenBits( index >> 1 );

    unsigned int fx = prefixEor( b0 );
    unsigned int fy = prefixEor( b0 & ~b1 );

    unsigned int t = fy ^ b1;

    u = ( fx & ~t ) | ( b0 & ~t ) | ( ~b0 & ~fx & t );
    v = fy ^ b0;
    w = ( ~fx & ~t ) | ( b0 & ~t ) | ( ~b0 & fx & t );
}

// Compute barycentrics of a sub or micro triangle wrt a base triangle.  The order of the returned
// bary0, bary1, bary2 matters and allows for using this function for sub triangles and the
// conversion from sub triangle to base triangle barycentric space
OPTIX_MICROMAP_INLINE_FUNC void micro2bary( unsigned int index, unsigned int subdivisionLevel, float2& bary0, float2& bary1, float2& bary2 )
{
    if( subdivisionLevel == 0 )
    {
        bary0 = { 0, 0 };
        bary1 = { 1, 0 };
        bary2 = { 0, 1 };
        return;
    }

    unsigned int iu, iv, iw;
    index2dbary( index, iu, iv, iw );

    // we need to only look at "level" bits
    iu = iu & ( ( 1 << subdivisionLevel ) - 1 );
    iv = iv & ( ( 1 << subdivisionLevel ) - 1 );
    iw = iw & ( ( 1 << subdivisionLevel ) - 1 );

    int yFlipped = ( iu & 1 ) ^ ( iv & 1 ) ^ ( iw & 1 ) ^ 1;

    int xFlipped = ( ( 0x8888888888888888ull ^ 0xf000f000f000f000ull ^ 0xffff000000000000ull ) >> index ) & 1;
    xFlipped    ^= ( ( 0x8888888888888888ull ^ 0xf000f000f000f000ull ^ 0xffff000000000000ull ) >> ( index >> 6 ) ) & 1;

    const float levelScale = __uint_as_float( ( 127u - subdivisionLevel ) << 23 );

    // scale the barycentic coordinate to the global space/scale
    float du = 1.f * levelScale;
    float dv = 1.f * levelScale;

    // scale the barycentic coordinate to the global space/scale
    float u = (float)iu * levelScale;
    float v = (float)iv * levelScale;

    //     c        d
    //      x-----x
    //     / \   /
    //    /   \ /
    //   x-----x
    //  a        b
    //
    // !xFlipped && !yFlipped: abc
    // !xFlipped &&  yFlipped: cdb
    //  xFlipped && !yFlipped: bac
    //  xFlipped &&  yFlipped: dcb

    bary0 = { u + xFlipped * du    , v + yFlipped * dv };
    bary1 = { u + (1-xFlipped) * du, v + yFlipped * dv };
    bary2 = { u + yFlipped * du    , v + (1-yFlipped) * dv };
}

// avoid any conflicts due to multiple definitions
#define OPTIX_MICROMAP_FLOAT2_SUB(a,b) { a.x - b.x, a.y - b.y }

// Compute barycentrics for micro triangle from base barycentrics
OPTIX_MICROMAP_INLINE_FUNC float2 base2micro( const float2& baseBarycentrics, const float2 microVertexBaseBarycentrics[3] )
{
    float2 baryV0P  = OPTIX_MICROMAP_FLOAT2_SUB( baseBarycentrics, microVertexBaseBarycentrics[0] );
    float2 baryV0V1 = OPTIX_MICROMAP_FLOAT2_SUB( microVertexBaseBarycentrics[1], microVertexBaseBarycentrics[0] );
    float2 baryV0V2 = OPTIX_MICROMAP_FLOAT2_SUB( microVertexBaseBarycentrics[2], microVertexBaseBarycentrics[0] );

    float  rdetA = 1.f / ( baryV0V1.x * baryV0V2.y - baryV0V1.y * baryV0V2.x );
    float4 A     = { baryV0V2.y, -baryV0V2.x, -baryV0V1.y, baryV0V1.x };

    float2 localUV;
    localUV.x = rdetA * ( baryV0P.x * A.x + baryV0P.y * A.y );
    localUV.y = rdetA * ( baryV0P.x * A.z + baryV0P.y * A.w );

    return localUV;
}
#undef OPTIX_MICROMAP_FLOAT2_SUB

/*@}*/  // end group optix_utilities

}  // namespace optix_impl

#endif  // OPTIX_OPTIX_MICROMAP_IMPL_H
