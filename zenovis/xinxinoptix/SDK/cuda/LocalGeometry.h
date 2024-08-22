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

#include <optix.h>

#include <sutil/Matrix.h>
#include <sutil/Preprocessor.h>
#include <sutil/vec_math.h>
#include <cuda/BufferView.h>
#include <cuda/GeometryData.h>
#include <cuda/util.h>


struct LocalGeometry
{
    float3 P;
    float3 N;
    float3 Ng;

    struct Texcoord
    {
        float2 UV;
        float3 dndu;
        float3 dndv;
        float3 dpdu;
        float3 dpdv;
    } texcoord[GeometryData::num_texcoords];

    float4 color;
};


__forceinline__ __device__ LocalGeometry getLocalGeometry( const GeometryData& geometry_data )
{
    LocalGeometry lgeom;
    switch( geometry_data.type )
    {
        case GeometryData::TRIANGLE_MESH:
        {
            const GeometryData::TriangleMesh& mesh_data = geometry_data.getTriangleMesh();

            const unsigned int prim_idx = optixGetPrimitiveIndex();
            const float2       barys    = optixGetTriangleBarycentrics();

            uint3 tri = make_uint3(0u, 0u, 0u);
            if( mesh_data.indices.elmt_byte_size == 4 )
            {
                const uint3* indices = reinterpret_cast<uint3*>( mesh_data.indices.data );
                tri = indices[ prim_idx ];
            }
            else if( mesh_data.indices.elmt_byte_size == 2 )
            {
                const unsigned short* indices = reinterpret_cast<unsigned short*>( mesh_data.indices.data );
                const unsigned short  idx0    = indices[prim_idx * 3 + 0];
                const unsigned short  idx1    = indices[prim_idx * 3 + 1];
                const unsigned short  idx2    = indices[prim_idx * 3 + 2];
                tri                           = make_uint3( idx0, idx1, idx2 );
            }
            else
            {
                const unsigned int base_idx = prim_idx * 3;
                tri = make_uint3( base_idx + 0, base_idx + 1, base_idx + 2 );
            }

            const float3 P0 = mesh_data.positions[ tri.x ];
            const float3 P1 = mesh_data.positions[ tri.y ];
            const float3 P2 = mesh_data.positions[ tri.z ];
            lgeom.P = ( 1.0f-barys.x-barys.y)*P0 + barys.x*P1 + barys.y*P2;
            lgeom.P = optixTransformPointFromObjectToWorldSpace( lgeom.P );

            if( mesh_data.colors )
            {
                const float4 COLOR0 = mesh_data.colors[tri.x];
                const float4 COLOR1 = mesh_data.colors[tri.y];
                const float4 COLOR2 = mesh_data.colors[tri.z];
                lgeom.color = ( 1.0f - barys.x - barys.y )*COLOR0 + barys.x*COLOR1 + barys.y*COLOR2;
            }
            else
            {
                lgeom.color = make_float4(1);
            }

            lgeom.Ng = cross( P1-P0, P2-P0 );
            lgeom.Ng = normalize( optixTransformNormalFromObjectToWorldSpace( lgeom.Ng ) );

            float3 N0, N1, N2;
            if( mesh_data.normals )
            {
                N0 = mesh_data.normals[ tri.x ];
                N1 = mesh_data.normals[ tri.y ];
                N2 = mesh_data.normals[ tri.z ];
                lgeom.N = ( 1.0f-barys.x-barys.y)*N0 + barys.x*N1 + barys.y*N2;
                lgeom.N = normalize( optixTransformNormalFromObjectToWorldSpace( lgeom.N ) );
            }
            else
            {
                lgeom.N = N0 = N1 = N2 = lgeom.Ng;
            }

            const float3 dp1 = P0 - P2;
            const float3 dp2 = P1 - P2;

            const float3 dn1 = N0 - N2;
            const float3 dn2 = N1 - N2;

            for( size_t j = 0; j < GeometryData::num_texcoords; j++ )
            {
                float2 UV0, UV1, UV2;
                if( mesh_data.texcoords[j] )
                {
                    UV0 = mesh_data.texcoords[j][tri.x];
                    UV1 = mesh_data.texcoords[j][tri.y];
                    UV2 = mesh_data.texcoords[j][tri.z];
                    lgeom.texcoord[j].UV = ( 1.0f - barys.x - barys.y )*UV0 + barys.x*UV1 + barys.y*UV2;

                    const float du1 = UV0.x - UV2.x;
                    const float du2 = UV1.x - UV2.x;
                    const float dv1 = UV0.y - UV2.y;
                    const float dv2 = UV1.y - UV2.y;

                    const float det = du1 * dv2 - dv1 * du2;

                    const float invdet = 1.f / det;
                    lgeom.texcoord[j].dpdu = (  dv2 * dp1 - dv1 * dp2 ) * invdet;
                    lgeom.texcoord[j].dpdv = ( -du2 * dp1 + du1 * dp2 ) * invdet;
                    lgeom.texcoord[j].dndu = (  dv2 * dn1 - dv1 * dn2 ) * invdet;
                    lgeom.texcoord[j].dndv = ( -du2 * dn1 + du1 * dn2 ) * invdet;
                }
                else
                {
                    lgeom.texcoord[j].UV   = barys;
                    lgeom.texcoord[j].dpdu = -dp1;
                    lgeom.texcoord[j].dpdv = -dp1 + dp2;
                    lgeom.texcoord[j].dndu = -dn1;
                    lgeom.texcoord[j].dndv = -dn1 + dn2;
                }
            }

            break;
        }
        case GeometryData::SPHERE:
        {
            break;
        }
        default: break;
    }


    return lgeom;
}


