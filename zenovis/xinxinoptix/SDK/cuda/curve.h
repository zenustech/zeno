//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
struct LinearInterpolator
{
    __device__ __forceinline__ LinearInterpolator() {}

    __device__ __forceinline__ void initialize( const float4* q )
    {
        p[0] = q[0];
        p[1] = q[1] - q[0];
    }


    __device__ __forceinline__ float4 position4( float u ) const
    {
        return p[0] + u * p[1]; // Horner scheme
    }

    __device__ __forceinline__ float3 position3( float u ) const
    {
        return make_float3( position4( u ) );
    }

    __device__ __forceinline__ float radius( const float& u ) const
    {
        return position4( u ).w;
    }

    __device__ __forceinline__ float4 velocity4( float u ) const
    {
        return p[1];
    }

    __device__ __forceinline__ float3 velocity3( float u ) const
    {
        return make_float3( velocity4( u ) );
    }

    __device__ __forceinline__ float derivative_of_radius( float u ) const
    {
        return velocity4( u ).w;
    }

    __device__ __forceinline__ float3 acceleration3( float u ) const { return make_float3( 0.f ); }
    __device__ __forceinline__ float4 acceleration4( float u ) const { return make_float4( 0.f ); }


    float4 p[2];
};


//
// Second order polynomial interpolator
//
struct QuadraticInterpolator
{
    __device__ __forceinline__ QuadraticInterpolator() {}

    __device__ __forceinline__ void initializeFromBSpline( const float4* q )
    {
        // Bspline-to-Poly = Matrix([[1/2,  -1, 1/2],
        //                           [-1,    1,   0],
        //                           [1/2, 1/2,   0]])
        p[0] = (         q[0] - 2.0f * q[1] + q[2] ) / 2.0f;
        p[1] = ( -2.0f * q[0] + 2.0f * q[1]        ) / 2.0f;
        p[2] = (         q[0] +        q[1]        ) / 2.0f;
    }

    __device__ __forceinline__ void export2BSpline( float4 bs[3] ) const
    {
        // inverse of initializeFromBSpline
        // Bspline-to-Poly = Matrix([[1/2,  -1, 1/2],
        //                           [-1,    1,   0],
        //                           [1/2, 1/2,   0]])
        // invert to get:
        // Poly-to-Bspline = Matrix([[0, -1/2, 1],
        //                           [0,  1/2, 1],
        //                           [2,  3/2, 1]])
        bs[0] = p[0] - p[1] / 2;
        bs[1] = p[0] + p[1] / 2;
        bs[2] = p[0] + 1.5f * p[1] + 2 * p[2];
    }

    __device__ __forceinline__ float4 position4( float u ) const
    {
        return ( p[0] * u + p[1] ) * u + p[2]; // Horner scheme
    }

    __device__ __forceinline__ float3 position3( float u ) const
    {
        return make_float3( position4( u ) );
    }

    __device__ __forceinline__ float radius( float u ) const
    {
        return position4( u ).w;
    }

    __device__ __forceinline__ float4 velocity4( float u ) const
    {
        return 2.0f * p[0] * u + p[1];
    }

    __device__ __forceinline__ float3 velocity3( float u ) const
    {
        return make_float3( velocity4( u ) );
    }

    __device__ __forceinline__ float derivative_of_radius( float u ) const
    {
        return velocity4( u ).w;
    }

    __device__ __forceinline__ float4 acceleration4( float u ) const
    {
        return 2.0f * p[0];
    }

    __device__ __forceinline__ float3 acceleration3( float u ) const
    {
        return make_float3( acceleration4( u ) );
    }


    float4 p[3];
};

//
// Third order polynomial interpolator
//
// Storing {p0, p1, p2, p3} for evaluation:
//     P(u) = p0 * u^3 + p1 * u^2 + p2 * u + p3
//
struct CubicInterpolator
{
    __device__ __forceinline__ CubicInterpolator() {}

    __device__ __forceinline__ void initializeFromBSpline( const float4* q )
    {
        // Bspline-to-Poly = Matrix([[-1/6, 1/2, -1/2, 1/6],
        //                           [ 1/2,  -1,  1/2,   0],
        //                           [-1/2,   0,  1/2,   0],
        //                           [ 1/6, 2/3,  1/6,   0]])
        p[0] = ( q[0] * ( -1.0f ) + q[1] * (  3.0f ) + q[2] * ( -3.0f ) + q[3] ) / 6.0f;
        p[1] = ( q[0] * (  3.0f ) + q[1] * ( -6.0f ) + q[2] * (  3.0f )        ) / 6.0f;
        p[2] = ( q[0] * ( -3.0f )                    + q[2] * (  3.0f )        ) / 6.0f;
        p[3] = ( q[0] * (  1.0f ) + q[1] * (  4.0f ) + q[2] * (  1.0f )        ) / 6.0f;
    }

    __device__ __forceinline__ void export2BSpline( float4 bs[4] ) const
    {
        // inverse of initializeFromBSpline
        // Bspline-to-Poly = Matrix([[-1/6, 1/2, -1/2, 1/6],
        //                           [ 1/2,  -1,  1/2,   0],
        //                           [-1/2,   0,  1/2,   0],
        //                           [ 1/6, 2/3,  1/6,   0]])
        // invert to get:
        // Poly-to-Bspline = Matrix([[0,  2/3, -1, 1],
        //                           [0, -1/3,  0, 1],
        //                           [0,  2/3,  1, 1],
        //                           [6, 11/3,  2, 1]])
        bs[0] = (        p[1] * (  2.0f ) + p[2] * ( -1.0f ) + p[3] ) / 3.0f;
        bs[1] = (        p[1] * ( -1.0f )                    + p[3] ) / 3.0f;
        bs[2] = (        p[1] * (  2.0f ) + p[2] * (  1.0f ) + p[3] ) / 3.0f;
        bs[3] = ( p[0] + p[1] * ( 11.0f ) + p[2] * (  2.0f ) + p[3] ) / 3.0f;
    }


    __device__ __forceinline__ void initializeFromCatrom(const float4* q)
    {
        // Catrom-to-Poly = Matrix([[-1/2, 3/2, -3/2,  1/2],
        //                          [1,   -5/2,    2, -1/2],
        //                          [-1/2,   0,  1/2,    0],
        //                          [0,      1,    0,    0]])
        p[0] = ( -1.0f * q[0] + (  3.0f ) * q[1] + ( -3.0f ) * q[2] + (  1.0f ) * q[3] ) / 2.0f;
        p[1] = (  2.0f * q[0] + ( -5.0f ) * q[1] + (  4.0f ) * q[2] + ( -1.0f ) * q[3] ) / 2.0f;
        p[2] = ( -1.0f * q[0]                    + (  1.0f ) * q[2]                    ) / 2.0f;
        p[3] = (                (  2.0f ) * q[1]                                       ) / 2.0f;
    }

    __device__ __forceinline__ void export2Catrom(float4 cr[4]) const
    {
        // Catrom-to-Poly = Matrix([[-1/2, 3/2, -3/2,  1/2],
        //                          [1,   -5/2,    2, -1/2],
        //                          [-1/2,   0,  1/2,    0],
        //                          [0,      1,    0,    0]])
        // invert to get:
        // Poly-to-Catrom = Matrix([[1, 1, -1, 1],
        //                          [0, 0, 0, 1],
        //                          [1, 1, 1, 1],
        //                          [6, 4, 2, 1]])
        cr[0] = ( p[0] * 6.f/6.f ) - ( p[1] * 5.f/6.f ) + ( p[2] * 2.f/6.f ) + ( p[3] * 1.f/6.f );
        cr[1] = ( p[0] * 6.f/6.f )                                                               ;
        cr[2] = ( p[0] * 6.f/6.f ) + ( p[1] * 1.f/6.f ) + ( p[2] * 2.f/6.f ) + ( p[3] * 1.f/6.f );
        cr[3] = ( p[0] * 6.f/6.f )                                           + ( p[3] * 6.f/6.f );
    }

    __device__ __forceinline__ void initializeFromBezier(const float4* q)
    {
        // Bezier-to-Poly = Matrix([[-1,  3, -3, 1],
        //                          [ 3, -6,  3, 0],
        //                          [-3,  3,  0, 0],
        //                          [ 1,  0,  0, 0]])
        p[0] = q[0] * ( -1.0f ) + q[1] * (  3.0f ) + q[2] * ( -3.0f ) + q[3];
        p[1] = q[0] * (  3.0f ) + q[1] * ( -6.0f ) + q[2] * (  3.0f );
        p[2] = q[0] * ( -3.0f ) + q[1] * (  3.0f );
        p[3] = q[0];
    }

    __device__ __forceinline__ void export2Bezier(float4 bz[4]) const
    {
        // inverse of initializeFromBezier
        // Bezier-to-Poly = Matrix([[-1,  3, -3, 1],
        //                          [ 3, -6,  3, 0],
        //                          [-3,  3,  0, 0],
        //                          [ 1,  0,  0, 0]])
        // invert to get:
        // Poly-to-Bezier = Matrix([[0,   0,   0, 1],
        //                          [0,   0, 1/3, 1],
        //                          [0, 1/3, 2/3, 1],
        //                          [1,   1,   1, 1]])
        bz[0] =                                              p[3];
        bz[1] =                           p[2] * (1.f/3.f) + p[3];
        bz[2] =        p[1] * (1.f/3.f) + p[2] * (2.f/3.f) + p[3];
        bz[3] = p[0] + p[1]             + p[2]             + p[3];
    }

    __device__ __forceinline__ float4 position4( float u ) const
    {
        return ( ( ( p[0] * u ) + p[1] ) * u + p[2] ) * u + p[3]; // Horner scheme
     }

    __device__ __forceinline__ float3 position3( float u ) const
    {
        // rely on compiler and inlining for dead code removal
        return make_float3( position4( u ) );
    }
    __device__ __forceinline__ float radius( float u ) const
    {
        return position4( u ).w;
    }

    __device__ __forceinline__ float4 velocity4( float u ) const
    {
        // adjust u to avoid problems with tripple knots.
        if( u == 0 )
            u = 0.000001f;
        if( u == 1 )
            u = 0.999999f;
        return ( ( 3.0f * p[0] * u ) + 2.0f * p[1] ) * u + p[2];
    }

    __device__ __forceinline__ float3 velocity3( float u ) const
    {
        return make_float3( velocity4( u ) );
    }

    __device__ __forceinline__ float derivative_of_radius( float u ) const
    {
        return velocity4( u ).w;
    }

    __device__ __forceinline__ float4 acceleration4( float u ) const
    {
        return 6.0f * p[0] * u + 2.0f * p[1]; // Horner scheme
    }

    __device__ __forceinline__ float3 acceleration3( float u ) const
    {
        return make_float3( acceleration4( u ) );
    }

    float4 p[4];
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
__device__ __forceinline__ float3 surfaceNormal( const LinearInterpolator& bc, float u, float3& ps )
{
    float3 normal;
    if( u == 0.0f )
    {
        normal = ps - ( float3 & )( bc.p[0] );  // special handling for round endcaps
    }
    else if( u >= 1.0f )
    {
        // reconstruct second control point (Note: the interpolator pre-transforms
        // the control-points to speed up repeated evaluation.
        const float3 p1 = ( float3 & ) (bc.p[1] ) + ( float3 & )( bc.p[0] );
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
