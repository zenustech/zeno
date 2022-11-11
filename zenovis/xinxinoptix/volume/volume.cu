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

#include "volume_cuda.h"

#include <optix.h>
#include <vector_types.h>
#include <cuda/random.h>
#include <cuda/helpers.h>

#include <nanovdb/util/Ray.h>
#include <nanovdb/util/HDDA.h>


__constant__ LaunchParams params;


// ----------------------------------------------------------------------------
// Raygen program
// ----------------------------------------------------------------------------
extern "C" __global__ void __raygen__pinhole()
{
    const uint3  launch_idx     = optixGetLaunchIndex();
    const uint3  launch_dims    = optixGetLaunchDimensions();
    const float3 eye            = params.eye;
    const float3 U              = params.U;
    const float3 V              = params.V;
    const float3 W              = params.W;
    const int    subframe_index = params.subframe_index;

    //
    // Generate camera ray
    //
    unsigned int seed = tea<4>( launch_idx.y * launch_dims.x + launch_idx.x, subframe_index );

    const float2 subpixel_jitter =
        subframe_index == 0 ? make_float2( 0.5f, 0.5f ) : make_float2( rnd( seed ), rnd( seed ) );

    const float2 d =
        2.0f * make_float2( ( static_cast<float>( launch_idx.x ) + subpixel_jitter.x ) / static_cast<float>( launch_dims.x ),
                            ( static_cast<float>( launch_idx.y ) + subpixel_jitter.y ) / static_cast<float>( launch_dims.y ) )
        - 1.0f;
    const float3 ray_direction = normalize( d.x * U + d.y * V + W );
    const float3 ray_origin    = eye;

    //
    // Trace camera ray
    //
    PayloadRadiance payload;
    payload.result     = make_float3( 0.0f );

	traceRadiance( params.handle, ray_origin, ray_direction,
		0.01f,  // tmin
		1e16f,  // tmax
		params.solid_objects | params.volume_object,
		&payload );

    //
    // Update results
    //
    const unsigned int image_index = launch_idx.y * launch_dims.x + launch_idx.x;
    const float4 accum_color = make_float4( payload.result, 1.0f );
    if( subframe_index )
        params.accum_buffer[image_index] +=  accum_color;
    else
        params.accum_buffer[image_index] = accum_color;
    const float scale = 1.0f / (subframe_index + 1);
    params.frame_buffer[image_index] = make_color( scale * params.accum_buffer[image_index] );
}

// ----------------------------------------------------------------------------
// Miss programs
// ----------------------------------------------------------------------------
extern "C" __global__ void __miss__radiance()
{
    optixSetPayload_0( __float_as_uint( params.miss_color.x ) );
    optixSetPayload_1( __float_as_uint( params.miss_color.y ) );
    optixSetPayload_2( __float_as_uint( params.miss_color.z ) );
    optixSetPayload_3( __float_as_uint( 1e16 ) ); // report depth (here "infinity")
}

extern "C" __global__ void __miss__occlusion()
{
    optixSetPayload_0( __float_as_uint( 1.0f ) ); // report transmittance
}

inline __device__ void transformNormalObjectToWorld(float3 &n)
{
    // because the sample only uses the translation part of the instance
    // transform, this can be a no-op
    // enforcing, that linear part of the affine transform is identity
    // matrix:
    for( unsigned int i = 0; i < optixGetTransformListSize(); ++i )
    {
        OptixTraversableHandle handle = optixGetTransformListHandle( i );
        switch( optixGetTransformTypeFromHandle( handle ) )
        {
            case OPTIX_TRANSFORM_TYPE_INSTANCE: {
                const float4* trns = optixGetInstanceInverseTransformFromHandle( handle );
                assert(trns[0].x == 1.0f && trns[0].y == 0.0f && trns[0].z == 0.0f);
                assert(trns[1].x == 0.0f && trns[1].y == 1.0f && trns[1].z == 0.0f);
                assert(trns[2].x == 0.0f && trns[2].y == 0.0f && trns[2].z == 1.0f);
            }
            break;
            default:
                assert(false); // there can only be a single instance transform
        }
    }
}

// ----------------------------------------------------------------------------
// Closest hit programs
// ----------------------------------------------------------------------------
extern "C" __global__ void __closesthit__radiance_mesh()
{
    const HitGroupData* hit_group_data = reinterpret_cast<HitGroupData*>( optixGetSbtDataPointer() );
    const float3 base_color = hit_group_data->material_data.lambert.base_color;
    const float ray_tmax = optixGetRayTmax();
    const float3 P = optixGetWorldRayOrigin() + ray_tmax * optixGetWorldRayDirection();

    //
    // compute direct lighting
    //

    const OptixTraversableHandle gas       = optixGetGASTraversableHandle();
    const unsigned int           gasSbtIdx = optixGetSbtGASIndex();
    const unsigned int           primIdx   = optixGetPrimitiveIndex();
    float3 vertices[3] = {};
    optixGetTriangleVertexData(
        gas,
        primIdx,
        gasSbtIdx,
        0,
        vertices );

    // compute normal from vertices (all objects have planar surfaces)
    float3 N = normalize(
        cross( vertices[1] - vertices[0], vertices[2] - vertices[0] )
    );
    transformNormalObjectToWorld(N);


    float3 result = make_float3( 0.0f );

    for( int i = 0; i < params.lights.count; ++i )
    {
        Light light = params.lights[i];
        if( light.type == Light::Type::POINT )
        {
            const float  L_dist  = length( light.point.position - P );
            const float3 L       = ( light.point.position - P ) / L_dist;
            const float  N_dot_L = dot( N, L );

            if( N_dot_L > 0.0f )
            {
                const float tmin    = 0.001f;
                const float tmax    = L_dist - 0.001f;
                float transmittance = 1.0f;
                traceOcclusion(
                    params.handle,
                    P,
                    L,
                    tmin,
                    tmax,
                    params.solid_objects | params.volume_object,
                    &transmittance
                );
                result += transmittance * base_color * light.point.color * light.point.intensity * N_dot_L;
            }
        }
    }
    optixSetPayload_0( __float_as_uint( result.x ) );
    optixSetPayload_1( __float_as_uint( result.y ) );
    optixSetPayload_2( __float_as_uint( result.z ) );
    optixSetPayload_3( __float_as_uint( ray_tmax ) ); // report depth
}

extern "C" __global__ void __closesthit__occlusion_mesh()
{
    optixSetPayload_0( __float_as_uint( 0.0f ) ); // report transmittance, i.e. plane is opaque
}


// ----------------------------------------------------------------------------
// Volume programs
// ----------------------------------------------------------------------------

inline __device__ void confine( const nanovdb::BBox<nanovdb::Coord> &bbox, nanovdb::Vec3f &iVec )
{
    // NanoVDB's voxels and tiles are formed from half-open intervals, i.e.
    // voxel[0, 0, 0] spans the set [0, 1) x [0, 1) x [0, 1). To find a point's voxel,
    // its coordinates are simply truncated to integer. Ray-box intersections yield
    // pairs of points that, because of numerical errors, fall randomly on either side
    // of the voxel boundaries.
    // This confine method, given a point and a (integer-based/Coord-based) bounding
    // box, moves points outside the bbox into it. That means coordinates at lower
    // boundaries are snapped to the integer boundary, and in case of the point being
    // close to an upper boundary, it is move one EPS below that bound and into the volume.

    // get the tighter box around active values
    auto iMin = nanovdb::Vec3f( bbox.min() );
    auto iMax = nanovdb::Vec3f( bbox.max() ) + nanovdb::Vec3f( 1.0f );

    // move the start and end points into the bbox
    float eps = 1e-7f;
    if( iVec[0] < iMin[0] ) iVec[0] = iMin[0];
    if( iVec[1] < iMin[1] ) iVec[1] = iMin[1];
    if( iVec[2] < iMin[2] ) iVec[2] = iMin[2];
    if( iVec[0] >= iMax[0] ) iVec[0] = iMax[0] - fmaxf( 1.0f, fabsf( iVec[0] ) ) * eps;
    if( iVec[1] >= iMax[1] ) iVec[1] = iMax[1] - fmaxf( 1.0f, fabsf( iVec[1] ) ) * eps;
    if( iVec[2] >= iMax[2] ) iVec[2] = iMax[2] - fmaxf( 1.0f, fabsf( iVec[2] ) ) * eps;
}

inline __hostdev__ void confine( const nanovdb::BBox<nanovdb::Coord> &bbox, nanovdb::Vec3f &iStart, nanovdb::Vec3f &iEnd )
{
    confine( bbox, iStart );
    confine( bbox, iEnd );
}

template<typename AccT>
inline __device__ float transmittanceHDDA(
    const nanovdb::Vec3f& start,
    const nanovdb::Vec3f& end,
    AccT& acc, const float opacity )
{

    // transmittance along a ray through the volume is computed by
    // taking the negative exponential of volume's density integrated
    // along the ray.
    float transmittance = 1.f;
    auto dir = end - start;
    auto len = dir.length();
    nanovdb::Ray<float> ray( start, dir / len, 0.0f, len );
    nanovdb::Coord ijk = nanovdb::RoundDown<nanovdb::Coord>( ray.start() ); // first hit of bbox

    // Use NanoVDB's HDDA line digitization for fast integration.
    // This algorithm (http://www.museth.org/Ken/Publications_files/Museth_SIG14.pdf)
    // can skip over sparse parts of the data structure.
    //
    nanovdb::HDDA<nanovdb::Ray<float> > hdda( ray, acc.getDim( ijk, ray ) );

    float t = 0.0f;
    float density = acc.getValue( ijk ) * opacity;
    while( hdda.step() )
    {
        float dt = hdda.time() - t; // compute length of ray-segment intersecting current voxel/tile
        transmittance *= expf( -density * dt );
        t = hdda.time();
        ijk = hdda.voxel();

        density = acc.getValue( ijk ) * opacity;
        hdda.update( ray, acc.getDim( ijk, ray ) ); // if necessary adjust DDA step size
    }

    return transmittance;
}


extern "C" __global__ void __intersection__volume()
{
    const auto* sbt_data = reinterpret_cast<const HitGroupData*>( optixGetSbtDataPointer() );
    const nanovdb::FloatGrid* grid = reinterpret_cast<const nanovdb::FloatGrid*>(
        sbt_data->geometry_data.volume.grid );
    assert( grid );

    // compute intersection points with the volume's bounds in index (object) space.
    const float3 ray_orig = optixGetObjectRayOrigin();
    const float3 ray_dir  = optixGetObjectRayDirection();

    auto bbox = grid->indexBBox();
    float t0 = optixGetRayTmin();
    float t1 = optixGetRayTmax();
    auto iRay = nanovdb::Ray<float>( reinterpret_cast<const nanovdb::Vec3f&>( ray_orig ),
        reinterpret_cast<const nanovdb::Vec3f&>( ray_dir ), t0, t1 );

    if( iRay.intersects( bbox, t0, t1 ) )
    {
        // report the exit point via payload
        optixSetPayload_0( __float_as_uint( t1 ) );
        // report the entry-point as hit-point
        optixReportIntersection( fmaxf( t0, optixGetRayTmin() ), 0 );
    }
}

extern "C" __global__ void __closesthit__radiance_volume()
{
    const HitGroupData* sbt_data = reinterpret_cast<HitGroupData*>( optixGetSbtDataPointer() );

    const auto* grid = reinterpret_cast<const nanovdb::FloatGrid*>(
        sbt_data->geometry_data.volume.grid );
    const auto& tree = grid->tree();
    auto        acc  = tree.getAccessor();

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();

    const float t0 = optixGetRayTmax();
    const float t1 = __uint_as_float( optixGetPayload_0() );

    // trace a continuation ray
    //
    // the continuation ray provides two things:
    //   - the radiance "entering the volume"
    //   - the "depth" to the next closest object intersected by the ray.
    // Note, that such an object might be inside the volume. In that case,
    // transmittance needs to be integrated through the volume along the ray
    // up to that closer hit-point.
    PayloadRadiance payload = {};
    traceRadiance(
        params.handle,
        ray_orig,
        ray_dir,
        0.0f,
        1e16f,
        params.solid_objects, // visibility mask - limit intersections to solid objects
        &payload
    );

    const auto ray = nanovdb::Ray<float>( reinterpret_cast<const nanovdb::Vec3f&>( ray_orig ),
		reinterpret_cast<const nanovdb::Vec3f&>( ray_dir ) );
    auto start = grid->worldToIndexF( ray( t0 ) );
    auto end   = grid->worldToIndexF( ray( fminf( payload.depth, t1 ) ) );

    auto bbox = grid->indexBBox();
    confine( bbox, start, end );

    // compute transmittance from the entry-point into the volume to either
    // the ray's exit point out of the volume, or the hit point found by the
    // continuation ray, if that is closer.
    const float opacity = sbt_data->material_data.volume.opacity;
    float  transmittance = transmittanceHDDA( start, end, acc, opacity );

    float3 result = payload.result * transmittance;

    optixSetPayload_0( __float_as_uint( result.x ) );
    optixSetPayload_1( __float_as_uint( result.y ) );
    optixSetPayload_2( __float_as_uint( result.z ) );
    optixSetPayload_3( __float_as_uint( 0.0f ) );
}

extern "C" __global__ void __closesthit__occlusion_volume()
{
    const HitGroupData* sbt_data = ( HitGroupData* )optixGetSbtDataPointer();

    const auto* grid = reinterpret_cast<const nanovdb::FloatGrid*>( sbt_data->geometry_data.volume.grid );
    auto        acc = grid->tree().getAccessor();

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();

    const float t0 = optixGetRayTmax();
    const float t1 = __uint_as_float( optixGetPayload_0() );

    float transmittance = 1.0f;

    // trace a continuation ray
    traceOcclusion(
        params.handle,
        ray_orig,
        ray_dir,
        0.01f,
        1e16f,
        params.solid_objects,
        &transmittance
    );

    // if the continuation ray didn't hit a solid, compute how much the volume
    // attenuates/shadows the light along the ray
    if( transmittance != 0.0f )
    {
        const auto ray = nanovdb::Ray<float>( reinterpret_cast<const nanovdb::Vec3f&>( ray_orig ),
                                              reinterpret_cast<const nanovdb::Vec3f&>( ray_dir ) );
        auto start = grid->worldToIndexF( ray( t0 ) );
        auto end   = grid->worldToIndexF( ray( t1 ) );

        auto bbox = grid->indexBBox();
        confine( bbox, start, end );

        const float opacity = sbt_data->material_data.volume.opacity;
        transmittance       *= transmittanceHDDA( start, end, acc, opacity );
    }
    optixSetPayload_0( __float_as_uint( transmittance ) );
}
