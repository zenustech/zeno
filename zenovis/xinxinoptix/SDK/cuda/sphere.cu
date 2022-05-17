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

#include <sutil/vec_math.h>

#include "sphere.h"

#define float3_as_ints( u ) float_as_int( u.x ), float_as_int( u.y ), float_as_int( u.z )

extern "C" __global__ void __intersection__sphere()
{
    const sphere::SphereHitGroupData* hit_group_data = reinterpret_cast<sphere::SphereHitGroupData*>( optixGetSbtDataPointer() );

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();
    const float  ray_tmin = optixGetRayTmin();
    const float  ray_tmax = optixGetRayTmax();

    const float3 O      = ray_orig - hit_group_data->sphere.center;
    const float  l      = 1.0f / length( ray_dir );
    const float3 D      = ray_dir * l;
    const float  radius = hit_group_data->sphere.radius;

    float b    = dot( O, D );
    float c    = dot( O, O ) - radius * radius;
    float disc = b * b - c;
    if( disc > 0.0f )
    {
        float sdisc        = sqrtf( disc );
        float root1        = ( -b - sdisc );
        float root11       = 0.0f;
        bool  check_second = true;

        const bool do_refine = fabsf( root1 ) > ( 10.0f * radius );

        if( do_refine )
        {
            // refine root1
            float3 O1 = O + root1 * D;
            b         = dot( O1, D );
            c         = dot( O1, O1 ) - radius * radius;
            disc      = b * b - c;

            if( disc > 0.0f )
            {
                sdisc  = sqrtf( disc );
                root11 = ( -b - sdisc );
            }
        }

        float  t;
        float3 normal;
        t = ( root1 + root11 ) * l;
        if( t > ray_tmin && t < ray_tmax )
        {
            normal = ( O + ( root1 + root11 ) * D ) / radius;
            if( optixReportIntersection( t, 0, float3_as_ints( normal ), float_as_int( radius ) ) )
                check_second = false;
        }

        if( check_second )
        {
            float root2 = ( -b + sdisc ) + ( do_refine ? root1 : 0 );
            t           = root2 * l;
            normal      = ( O + root2 * D ) / radius;
            if( t > ray_tmin && t < ray_tmax )
                optixReportIntersection( t, 0, float3_as_ints( normal ), float_as_int( radius ) );
        }
    }
}
