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
#include <optix.h>

#include <cuda/LocalGeometry.h>
#include <cuda/helpers.h>
#include <cuda/random.h>
#include <sutil/vec_math.h>

#include "whitted_cuda.h"

//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

extern "C" __global__ void __raygen__pinhole()
{
    const uint3  launch_idx     = optixGetLaunchIndex();
    const uint3  launch_dims    = optixGetLaunchDimensions();
    const float3 eye            = whitted::params.eye;
    const float3 U              = whitted::params.U;
    const float3 V              = whitted::params.V;
    const float3 W              = whitted::params.W;
    const int    subframe_index = whitted::params.subframe_index;

    //
    // Generate camera ray
    //
    unsigned int seed = tea<4>( launch_idx.y * launch_dims.x + launch_idx.x, subframe_index );

    // The center of each pixel is at fraction (0.5,0.5)
    const float2 subpixel_jitter =
        subframe_index == 0 ? make_float2( 0.5f, 0.5f ) : make_float2( rnd( seed ), rnd( seed ) );

    const float2 d =
        2.0f
            * make_float2( ( static_cast<float>( launch_idx.x ) + subpixel_jitter.x ) / static_cast<float>( launch_dims.x ),
                           ( static_cast<float>( launch_idx.y ) + subpixel_jitter.y ) / static_cast<float>( launch_dims.y ) )
        - 1.0f;
    const float3 ray_direction = normalize( d.x * U + d.y * V + W );
    const float3 ray_origin    = eye;

    //
    // Trace camera ray
    //
    whitted::PayloadRadiance payload;
    payload.result     = make_float3( 0.0f );
    payload.importance = 1.0f;
    payload.depth      = 0.0f;

    traceRadiance( whitted::params.handle, ray_origin, ray_direction,
                   0.01f,  // tmin       // TODO: smarter offset
                   1e16f,  // tmax
                   &payload );

    //
    // Update results
    // TODO: timview mode
    //
    const unsigned int image_index = launch_idx.y * launch_dims.x + launch_idx.x;
    float3             accum_color = payload.result;

    if( subframe_index > 0 )
    {
        const float  a                = 1.0f / static_cast<float>( subframe_index + 1 );
        const float3 accum_color_prev = make_float3( whitted::params.accum_buffer[image_index] );
        accum_color                   = lerp( accum_color_prev, accum_color, a );
    }
    whitted::params.accum_buffer[image_index] = make_float4( accum_color, 1.0f );
    whitted::params.frame_buffer[image_index] = make_color( accum_color );
}


extern "C" __global__ void __miss__constant_radiance()
{
    whitted::setPayloadResult( whitted::params.miss_color );
}


extern "C" __global__ void __closesthit__occlusion()
{
    whitted::setPayloadOcclusion( true );
}


extern "C" __global__ void __closesthit__radiance()
{
    const whitted::HitGroupData* hit_group_data = reinterpret_cast<whitted::HitGroupData*>( optixGetSbtDataPointer() );
    const LocalGeometry          geom           = getLocalGeometry( hit_group_data->geometry_data );

    //
    // Retrieve material data
    //
    float3 base_color = make_float3( hit_group_data->material_data.pbr.base_color );
    if( hit_group_data->material_data.pbr.base_color_tex )
        base_color *= whitted::linearize(
            make_float3( tex2D<float4>( hit_group_data->material_data.pbr.base_color_tex, geom.UV.x, geom.UV.y ) ) );

    float  metallic  = hit_group_data->material_data.pbr.metallic;
    float  roughness = hit_group_data->material_data.pbr.roughness;
    float4 mr_tex    = make_float4( 1.0f );
    if( hit_group_data->material_data.pbr.metallic_roughness_tex )
        // MR tex is (occlusion, roughness, metallic )
        mr_tex = tex2D<float4>( hit_group_data->material_data.pbr.metallic_roughness_tex, geom.UV.x, geom.UV.y );
    roughness *= mr_tex.y;
    metallic *= mr_tex.z;

    //
    // Convert to material params
    //
    const float  F0         = 0.04f;
    const float3 diff_color = base_color * ( 1.0f - F0 ) * ( 1.0f - metallic );
    const float3 spec_color = lerp( make_float3( F0 ), base_color, metallic );
    const float  alpha      = roughness * roughness;

    //
    // compute direct lighting
    //

    float3 N = geom.N;
    if( hit_group_data->material_data.pbr.normal_tex )
    {
        const float4 NN =
            2.0f * tex2D<float4>( hit_group_data->material_data.pbr.normal_tex, geom.UV.x, geom.UV.y ) - make_float4( 1.0f );
        N = normalize( NN.x * normalize( geom.dpdu ) + NN.y * normalize( geom.dpdv ) + NN.z * geom.N );
    }

    float3 result = make_float3( 0.0f );

    for( int i = 0; i < whitted::params.lights.count; ++i )
    {
        Light light = whitted::params.lights[i];
        if( light.type == Light::Type::POINT )
        {
            // TODO: optimize
            const float  L_dist  = length( light.point.position - geom.P );
            const float3 L       = ( light.point.position - geom.P ) / L_dist;
            const float3 V       = -normalize( optixGetWorldRayDirection() );
            const float3 H       = normalize( L + V );
            const float  N_dot_L = dot( N, L );
            const float  N_dot_V = dot( N, V );
            const float  N_dot_H = dot( N, H );
            const float  V_dot_H = dot( V, H );

            if( N_dot_L > 0.0f && N_dot_V > 0.0f )
            {
                const float tmin     = 0.001f;           // TODO
                const float tmax     = L_dist - 0.001f;  // TODO
                const bool  occluded = whitted::traceOcclusion( whitted::params.handle, geom.P, L, tmin, tmax );
                if( !occluded )
                {
                    const float3 F     = whitted::schlick( spec_color, V_dot_H );
                    const float  G_vis = whitted::vis( N_dot_L, N_dot_V, alpha );
                    const float  D     = whitted::ggxNormal( N_dot_H, alpha );

                    const float3 diff = ( 1.0f - F ) * diff_color / M_PIf;
                    const float3 spec = F * G_vis * D;

                    result += light.point.color * light.point.intensity * N_dot_L * ( diff + spec );
                }
            }
        }
        else if( light.type == Light::Type::AMBIENT )
        {
            result += light.ambient.color * base_color;
        }
    }

    whitted::setPayloadResult( result );
}
