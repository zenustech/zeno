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
#include <sutil/vec_math.h>
#include <vector_types.h>

//
// First order polynomial interpolator
//
struct LinearBSplineSegment
{
    __device__ __forceinline__ LinearBSplineSegment() {}
    __device__ __forceinline__ LinearBSplineSegment( const float4* q ) { initialize( q ); }

    __device__ __forceinline__ void initialize( const float4* q )
    {
        p[0] = q[0];
        p[1] = q[1] - q[0];  // pre-transform p[] for fast evaluation
    }

    __device__ __forceinline__ float radius( const float& u ) const { return p[0].w + p[1].w * u; }

    __device__ __forceinline__ float3 position3( float u ) const { return (float3&)p[0] + u * (float3&)p[1]; }
    __device__ __forceinline__ float4 position4( float u ) const { return p[0] + u * p[1]; }

    __device__ __forceinline__ float min_radius( float u1, float u2 ) const
    {
        return fminf( radius( u1 ), radius( u2 ) );
    }

    __device__ __forceinline__ float max_radius( float u1, float u2 ) const
    {
        if( !p[1].w )
            return p[0].w;  // a quick bypass for constant width
        return fmaxf( radius( u1 ), radius( u2 ) );
    }

    __device__ __forceinline__ float3 velocity3( float u ) const { return (float3&)p[1]; }
    __device__ __forceinline__ float4 velocity4( float u ) const { return p[1]; }

    __device__ __forceinline__ float3 acceleration3( float u ) const { return make_float3( 0.f ); }
    __device__ __forceinline__ float4 acceleration4( float u ) const { return make_float4( 0.f ); }

    __device__ __forceinline__ float derivative_of_radius( float u ) const { return p[1].w; }

    float4 p[2];  // pre-transformed "control points" for fast evaluation
};


//
// Second order polynomial interpolator
//
struct QuadraticBSplineSegment
{
    __device__ __forceinline__ QuadraticBSplineSegment() {}
    __device__ __forceinline__ QuadraticBSplineSegment( const float4* q ) { initializeFromBSpline( q ); }

    __device__ __forceinline__ void initializeFromBSpline( const float4* q )
    {
        // pre-transform control-points for fast evaluation
        p[0] = q[1] / 2.0f + q[0] / 2.0f;
        p[1] = q[1] - q[0];
        p[2] = q[0] / 2.0f - q[1] + q[2] / 2.0f;
    }

    __device__ __forceinline__ void export2BSpline( float4 bs[3] ) const
    {
        bs[0] = p[0] - p[1] / 2;
        bs[1] = p[0] + p[1] / 2;
        bs[2] = p[0] + 1.5f * p[1] + 2 * p[2];
    }

    __device__ __forceinline__ float3 position3( float u ) const
    {
        return (float3&)p[0] + u * (float3&)p[1] + u * u * (float3&)p[2];
    }
    __device__ __forceinline__ float4 position4( float u ) const { return p[0] + u * p[1] + u * u * p[2]; }

    __device__ __forceinline__ float radius( float u ) const { return p[0].w + u * ( p[1].w + u * p[2].w ); }

    __device__ __forceinline__ float min_radius( float u1, float u2 ) const
    {
        float root1 = clamp( -0.5f * p[1].w / p[2].w, u1, u2 );
        return fminf( fminf( radius( u1 ), radius( u2 ) ), radius( root1 ) );
    }

    __device__ __forceinline__ float max_radius( float u1, float u2 ) const
    {
        if( !p[1].w && !p[2].w )
            return p[0].w;  // a quick bypass for constant width
        float root1 = clamp( -0.5f * p[1].w / p[2].w, u1, u2 );
        return fmaxf( fmaxf( radius( u1 ), radius( u2 ) ), radius( root1 ) );
    }

    __device__ __forceinline__ float3 velocity3( float u ) const { return (float3&)p[1] + 2 * u * (float3&)p[2]; }
    __device__ __forceinline__ float4 velocity4( float u ) const { return p[1] + 2 * u * p[2]; }

    __device__ __forceinline__ float3 acceleration3( float u ) const { return 2 * (float3&)p[2]; }
    __device__ __forceinline__ float4 acceleration4( float u ) const { return 2 * p[2]; }

    __device__ __forceinline__ float derivative_of_radius( float u ) const { return p[1].w + 2 * u * p[2].w; }

    float4 p[3];  // pre-transformed "control points" for fast evaluation
};

//
// Third order polynomial interpolator
//
struct CubicBSplineSegment
{
    __device__ __forceinline__ CubicBSplineSegment() {}
    __device__ __forceinline__ CubicBSplineSegment( const float4* q ) { initializeFromBSpline( q ); }

    __device__ __forceinline__ void initializeFromBSpline( const float4* q )
    {
        // pre-transform control points for fast evaluation
        p[0] = ( q[2] + q[0] ) / 6 + ( 4 / 6.0f ) * q[1];
        p[1] = q[2] - q[0];
        p[2] = q[2] - q[1];
        p[3] = q[3] - q[1];
    }

    __device__ __forceinline__ void export2BSpline( float4 bs[4] ) const
    {
        // inverse of initializeFromBSpline
        bs[0] = p[0] + ( 4 * p[2] - 5 * p[1] ) / 6;
        bs[1] = p[0] + ( p[1] - 2 * p[2] ) / 6;
        bs[2] = p[0] + ( p[1] + 4 * p[2] ) / 6;
        bs[3] = p[0] + p[3] + ( p[1] - 2 * p[2] ) / 6;
    }

    __device__ __forceinline__ static float3 terms( float u )
    {
        float uu = u * u;
        float u3 = ( 1 / 6.0f ) * uu * u;
        return make_float3( u3 + 0.5f * ( u - uu ), uu - 4 * u3, u3 );
    }

    __device__ __forceinline__ float3 position3( float u ) const
    {
        float3 q = terms( u );
        return (float3&)p[0] + q.x * (float3&)p[1] + q.y * (float3&)p[2] + q.z * (float3&)p[3];
    }
    __device__ __forceinline__ float4 position4( float u ) const
    {
        float3 q = terms( u );
        return p[0] + q.x * p[1] + q.y * p[2] + q.z * p[3];
    }

    __device__ __forceinline__ float radius( float u ) const
    {
        return p[0].w + u * ( p[1].w / 2 + u * ( ( p[2].w - p[1].w / 2 ) + u * ( p[1].w - 4 * p[2].w + p[3].w ) / 6 ) );
    }

    __device__ __forceinline__ float min_radius( float u1, float u2 ) const
    {
        // a + 2 b u - c u^2
        float a    = p[1].w;
        float b    = 2 * p[2].w - p[1].w;
        float c    = 4 * p[2].w - p[1].w - p[3].w;
        float rmin = fminf( radius( u1 ), radius( u2 ) );
        if( fabsf( c ) < 1e-5f )
        {
            float root1 = clamp( -0.5f * a / b, u1, u2 );
            return fminf( rmin, radius( root1 ) );
        }
        else
        {
            float det   = b * b + a * c;
            det         = det <= 0.0f ? 0.0f : sqrt( det );
            float root1 = clamp( ( b + det ) / c, u1, u2 );
            float root2 = clamp( ( b - det ) / c, u1, u2 );
            return fminf( rmin, fminf( radius( root1 ), radius( root2 ) ) );
        }
    }

    __device__ __forceinline__ float max_radius( float u1, float u2 ) const
    {
        if( !p[1].w && !p[2].w && !p[3].w )
            return p[0].w;  // a quick bypass for constant width
        // a + 2 b u - c u^2
        float a    = p[1].w;
        float b    = 2 * p[2].w - p[1].w;
        float c    = 4 * p[2].w - p[1].w - p[3].w;
        float rmax = fmaxf( radius( u1 ), radius( u2 ) );
        if( fabsf( c ) < 1e-5f )
        {
            float root1 = clamp( -0.5f * a / b, u1, u2 );
            return fmaxf( rmax, radius( root1 ) );
        }
        else
        {
            float det   = b * b + a * c;
            det         = det <= 0.0f ? 0.0f : sqrt( det );
            float root1 = clamp( ( b + det ) / c, u1, u2 );
            float root2 = clamp( ( b - det ) / c, u1, u2 );
            return fmaxf( rmax, fmaxf( radius( root1 ), radius( root2 ) ) );
        }
    }

    __device__ __forceinline__ float3 velocity3( float u ) const
    {
        // adjust u to avoid problems with tripple knots.
        if( u == 0 )
            u = 0.000001f;
        if( u == 1 )
            u = 0.999999f;
        float v = 1 - u;
        return 0.5f * v * v * (float3&)p[1] + 2 * v * u * (float3&)p[2] + 0.5f * u * u * (float3&)p[3];
    }

    __device__ __forceinline__ float4 velocity4( float u ) const
    {
        // adjust u to avoid problems with tripple knots.
        if( u == 0 )
            u = 0.000001f;
        if( u == 1 )
            u = 0.999999f;
        float v = 1 - u;
        return 0.5f * v * v * p[1] + 2 * v * u * p[2] + 0.5f * u * u * p[3];
    }

    __device__ __forceinline__ float3 acceleration3( float u ) const { return make_float3( acceleration4( u ) ); }
    __device__ __forceinline__ float4 acceleration4( float u ) const
    {
        return 2 * p[2] - p[1] + ( p[1] - 4 * p[2] + p[3] ) * u;
    }

    __device__ __forceinline__ float derivative_of_radius( float u ) const
    {
        float v = 1 - u;
        return 0.5f * v * v * p[1].w + 2 * v * u * p[2].w + 0.5f * u * u * p[3].w;
    }

    float4 p[4];  // pre-transformed "control points" for fast evaluation
};

// Compute curve primitive surface normal in object space.
//
// Template parameters:
//   CurveType - A B-Spline evaluator class.
//   type - 0     ~ cylindrical approximation (correct if radius' == 0)
//          1     ~ conic       approximation (correct if curve'' == 0)
//          other ~ the bona fide surface normal
//
// Parameters:
//   bc - A B-Spline evaluator object.
//   u  - segment parameter of hit-point.
//   ps - hit-point on curve's surface in object space; usually
//        computed like this.
//        float3 ps = ray_orig + t_hit * ray_dir;
//        the resulting point is slightly offset away from the
//        surface. For this reason (Warning!) ps gets modified by this
//        method, projecting it onto the surface
//        in case it is not already on it. (See also inline
//        comments.)
//
template <typename CurveType, int type = 2>
__device__ __forceinline__ float3 surfaceNormal( const CurveType& bc, float u, float3& ps )
{
    float3 normal;
    if( u == 0.0f )
    {
        normal = -bc.velocity3( 0 );  // special handling for flat endcaps
    }
    else if( u == 1.0f )
    {
        normal = bc.velocity3( 1 );   // special handling for flat endcaps
    }
    else
    {
        // ps is a point that is near the curve's offset surface,
        // usually ray.origin + ray.direction * rayt.
        // We will push it exactly to the surface by projecting it to the plane(p,d).
        // The function derivation:
        // we (implicitly) transform the curve into coordinate system
        // {p, o1 = normalize(ps - p), o2 = normalize(curve'(t)), o3 = o1 x o2} in which
        // curve'(t) = (0, length(d), 0); ps = (r, 0, 0);
        float4 p4 = bc.position4( u );
        float3 p  = make_float3( p4 );
        float  r  = p4.w;  // == length(ps - p) if ps is already on the surface
        float4 d4 = bc.velocity4( u );
        float3 d  = make_float3( d4 );
        float  dr = d4.w;
        float  dd = dot( d, d );

        float3 o1 = ps - p;               // dot(modified_o1, d) == 0 by design:
        o1 -= ( dot( o1, d ) / dd ) * d;  // first, project ps to the plane(p,d)
        o1 *= r / length( o1 );           // and then drop it to the surface
        ps = p + o1;                      // fine-tuning the hit point
        if( type == 0 )
        {
            normal = o1;  // cylindrical approximation
        }
        else
        {
            if( type != 1 )
            {
                dd -= dot( bc.acceleration3( u ), o1 );
            }
            normal = dd * o1 - ( dr * r ) * d;
        }
    }
    return normalize( normal );
}

template <int type = 1>
__device__ __forceinline__ float3 surfaceNormal( const LinearBSplineSegment& bc, float u, float3& ps )
{
    float3 normal;
    if( u == 0.0f )
    {
        normal = ps - (float3&)(bc.p[0]);  // special handling for round endcaps
    }
    else if( u >= 1.0f )
    {
        // reconstruct second control point (Note: the interpolator pre-transforms
        // the control-points to speed up repeated evaluation.
        const float3 p1 = (float3&)(bc.p[1]) + (float3&)(bc.p[0]);
        normal = ps - p1;  // special handling for round endcaps
    }
    else
    {
        // ps is a point that is near the curve's offset surface,
        // usually ray.origin + ray.direction * rayt.
        // We will push it exactly to the surface by projecting it to the plane(p,d).
        // The function derivation:
        // we (implicitly) transform the curve into coordinate system
        // {p, o1 = normalize(ps - p), o2 = normalize(curve'(t)), o3 = o1 x o2} in which
        // curve'(t) = (0, length(d), 0); ps = (r, 0, 0);
        float4 p4 = bc.position4( u );
        float3 p  = make_float3( p4 );
        float  r  = p4.w;  // == length(ps - p) if ps is already on the surface
        float4 d4 = bc.velocity4( u );
        float3 d  = make_float3( d4 );
        float  dr = d4.w;
        float  dd = dot( d, d );

        float3 o1 = ps - p;               // dot(modified_o1, d) == 0 by design:
        o1 -= ( dot( o1, d ) / dd ) * d;  // first, project ps to the plane(p,d)
        o1 *= r / length( o1 );           // and then drop it to the surface
        ps = p + o1;                      // fine-tuning the hit point
        if( type == 0 )
        {
            normal = o1;  // cylindrical approximation
        }
        else
        {
            normal = dd * o1 - ( dr * r ) * d;
        }
    }
    return normalize( normal );
}

// Compute curve primitive tangent in object space.
//
// Template parameters:
//   CurveType - A B-Spline evaluator class.
//
// Parameters:
//   bc - A B-Spline evaluator object.
//   u  - segment parameter of tangent location on curve.
//
template <typename CurveType>
__device__ __forceinline__ float3 curveTangent( const CurveType& bc, float u )
{
    float3 tangent = bc.velocity3( u );
    return normalize( tangent );
}
