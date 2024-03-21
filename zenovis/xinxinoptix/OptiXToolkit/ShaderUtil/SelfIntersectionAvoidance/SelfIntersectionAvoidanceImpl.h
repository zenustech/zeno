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

/// \file SelfIntersectionAvoidanceImpl.h
/// Common implementation of Self Intersection Avoidance library.
///

namespace SelfIntersectionAvoidance {

// fast approximate 1/x, with up to 2 ulp error
OTK_INLINE __device__ float __frcp_approx( const float f )
{
    return __fdividef( 1.f, f );
}

// 3D dot product
OTK_INLINE __device__ float dot_rn( const float3& u, const float3& v )
{
    return __fmaf_rn( u.x, v.x, __fmaf_rn( u.y, v.y, __fmul_rn( u.z, v.z ) ) );
}

OTK_INLINE __device__ float dot_ru( const float3& u, const float3& v )
{
    return __fmaf_ru( u.x, v.x, __fmaf_ru( u.y, v.y, __fmul_ru( u.z, v.z ) ) );
}

OTK_INLINE __device__ float dot_rd( const float3& u, const float3& v )
{
    return __fmaf_rd( u.x, v.x, __fmaf_rd( u.y, v.y, __fmul_rd( u.z, v.z ) ) );
}

OTK_INLINE __device__ float dot_abs_rn( const float3& u, const float3& v )
{
    return __fmaf_rn( fabsf( u.x ), fabsf( v.x ),
                      __fmaf_rn( fabsf( u.y ), fabsf( v.y ), __fmul_rn( fabsf( u.z ), fabsf( v.z ) ) ) );
}

OTK_INLINE __device__ float dot_abs_ru( const float3& u, const float3& v )
{
    return __fmaf_ru( fabsf( u.x ), fabsf( v.x ),
                      __fmaf_ru( fabsf( u.y ), fabsf( v.y ), __fmul_ru( fabsf( u.z ), fabsf( v.z ) ) ) );
}

OTK_INLINE __device__ float length( const float3& u )
{
    return sqrtf( dot_rn( u, u ) );
}

OTK_INLINE __device__ float3 normalize( const float3& u )
{
    const float s = rsqrtf( dot_rn( u, u ) );
    return make_float3( __fmul_rn( u.x, s ), __fmul_rn( u.y, s ), __fmul_rn( u.z, s ) );
}

OTK_INLINE __device__ float length_L1( const float3& u )
{
    return __fadd_rn( __fadd_rn( fabsf( u.x ), fabsf( u.y ) ), fabsf( u.z ) );
}

OTK_INLINE __device__ float3 normalize_L1( const float3& u )
{
    const float s = __frcp_approx( length_L1( u ) );
    return make_float3( __fmul_rn( u.x, s ), __fmul_rn( u.y, s ), __fmul_rn( u.z, s ) );
}

OTK_INLINE __device__ float __fmmsf_rn( const float a, const float b, const float c, const float d )
{
    const float cd = c * d;
    const float e  = __fmaf_rn( -c, d, cd );
    const float f  = __fmaf_rn( a, b, -cd );
    return f + e;
}

OTK_INLINE __device__ float3 cross( const float3& a, const float3& b )
{
    return make_float3( __fmmsf_rn( a.y, b.z, a.z, b.y ), __fmmsf_rn( a.z, b.x, a.x, b.z ), __fmmsf_rn( a.x, b.y, a.y, b.x ) );
}

// offset a position away from a surface along the surface normal
OTK_INLINE __device__ void offsetSpawnPoint( float3&       outPosition,
                                             const float3& position,   // position to offset
                                             const float3& direction,  // offset direction
                                             const float   offset )      // offset distance
{
    // offset point along direction, rounding away from the position
    outPosition.x = ( direction.x > 0.f ) ? __fmaf_ru( offset, direction.x, position.x ) :
                                            __fmaf_rd( offset, direction.x, position.x );
    outPosition.y = ( direction.y > 0.f ) ? __fmaf_ru( offset, direction.y, position.y ) :
                                            __fmaf_rd( offset, direction.y, position.y );
    outPosition.z = ( direction.z > 0.f ) ? __fmaf_ru( offset, direction.z, position.z ) :
                                            __fmaf_rd( offset, direction.z, position.z );
}

// offset a position away from a surface along the surface normal
OTK_INLINE __device__ void offsetSpawnPoint( float3&       outFront,
                                             float3&       outBack,
                                             const float3& position,   // position to offset
                                             const float3& direction,  // offset direction
                                             const float   offset )      // offset distance
{
    // offset point along direction, rounding away from the position
    outFront.x = ( direction.x > 0.f ) ? __fmaf_ru( offset, direction.x, position.x ) :
                                         __fmaf_rd( offset, direction.x, position.x );
    outFront.y = ( direction.y > 0.f ) ? __fmaf_ru( offset, direction.y, position.y ) :
                                         __fmaf_rd( offset, direction.y, position.y );
    outFront.z = ( direction.z > 0.f ) ? __fmaf_ru( offset, direction.z, position.z ) :
                                         __fmaf_rd( offset, direction.z, position.z );

    outBack.x = ( direction.x > 0.f ) ? __fmaf_rd( -offset, direction.x, position.x ) :
                                        __fmaf_ru( -offset, direction.x, position.x );
    outBack.y = ( direction.y > 0.f ) ? __fmaf_rd( -offset, direction.y, position.y ) :
                                        __fmaf_ru( -offset, direction.y, position.y );
    outBack.z = ( direction.z > 0.f ) ? __fmaf_rd( -offset, direction.z, position.z ) :
                                        __fmaf_ru( -offset, direction.z, position.z );
}

// interpolate triangle point p=v0+alpha*e1+beta*e2 using barycentrics
// and compute object space safe offsetting distance for interpolated triangle points
OTK_INLINE __device__ float3 getTrianglePointAndError( float3&       outError,
                                                       const float3& v0,     // triangle base vertex
                                                       const float3& e1,     // triangle edge
                                                       const float3& e2,     // triangle edge
                                                       const float2& bary )  // barycentric coordinates
{
    const float ox = __fadd_rn( v0.x, __fmaf_rn( bary.x, e1.x, __fmul_rn( bary.y, e2.x ) ) );
    const float oy = __fadd_rn( v0.y, __fmaf_rn( bary.x, e1.y, __fmul_rn( bary.y, e2.y ) ) );
    const float oz = __fadd_rn( v0.z, __fmaf_rn( bary.x, e1.z, __fmul_rn( bary.y, e2.z ) ) );

    constexpr float c0 = 5.9604648328104529e-08f;
    constexpr float c1 = 1.1920930376163769e-07f;

    const float eps_x = __fmul_ru( c1, __fadd_ru( __fadd_ru( fabsf( e1.x ), fabsf( e2.x ) ), fabsf( __fsub_ru( e1.x, e2.x ) ) ) );
    const float eps_y = __fmul_ru( c1, __fadd_ru( __fadd_ru( fabsf( e1.y ), fabsf( e2.y ) ), fabsf( __fsub_ru( e1.y, e2.y ) ) ) );
    const float eps_z = __fmul_ru( c1, __fadd_ru( __fadd_ru( fabsf( e1.z ), fabsf( e2.z ) ), fabsf( __fsub_ru( e1.z, e2.z ) ) ) );

    // reconstruction + triangle intersection epsilon...
    const float eps = fmaxf( fmaxf( eps_x, eps_y ), eps_z );

    outError.x = __fmaf_ru( c0, fabsf( v0.x ), eps );
    outError.y = __fmaf_ru( c0, fabsf( v0.y ), eps );
    outError.z = __fmaf_ru( c0, fabsf( v0.z ), eps );

    return make_float3( ox, oy, oz );
}

OTK_INLINE __device__ float3 getObjectToWorldTransformedPositionAndError( float3& outError, const Matrix3x4& im, const float3& p )
{
    // assuming 1 ulp error in the translation elements in the inverse world to object matrix
    constexpr float c0 = 1.19209317972490680404007434844970703125E-7f;
    constexpr float c1 = 1.19209317972490680404007434844970703125E-7f;

    // The object to world matrix is assumed to have no more than ~1 ulp error in its elements.
    // f32 matrix inversion error will generally exceed this limit, in particular on the translation elements.
    // We could resolve this here by applying translation before transformation using the negative world to object translation.
    // This however would come at the cost of less tight error bounds wrt world space position.

    float3 q;
    q.x = __fadd_rn( __fmaf_rn( im.row0.x, p.x, __fmaf_rn( im.row0.y, p.y, __fmul_rn( im.row0.z, p.z ) ) ), im.row0.w );
    q.y = __fadd_rn( __fmaf_rn( im.row1.x, p.x, __fmaf_rn( im.row1.y, p.y, __fmul_rn( im.row1.z, p.z ) ) ), im.row1.w );
    q.z = __fadd_rn( __fmaf_rn( im.row2.x, p.x, __fmaf_rn( im.row2.y, p.y, __fmul_rn( im.row2.z, p.z ) ) ), im.row2.w );

    outError.x =
        __fmaf_ru( c1,
                   ( __fmaf_ru( fabsf( p.x ), fabsf( im.row0.x ),
                                __fmaf_ru( fabsf( p.y ), fabsf( im.row0.y ), __fmul_ru( fabsf( p.z ), fabsf( im.row0.z ) ) ) ) ),
                   __fmul_ru( c0, fabsf( im.row0.w ) ) );
    outError.y =
        __fmaf_ru( c1,
                   ( __fmaf_ru( fabsf( p.x ), fabsf( im.row1.x ),
                                __fmaf_ru( fabsf( p.y ), fabsf( im.row1.y ), __fmul_ru( fabsf( p.z ), fabsf( im.row1.z ) ) ) ) ),
                   __fmul_ru( c0, fabsf( im.row1.w ) ) );
    outError.z =
        __fmaf_ru( c1,
                   ( __fmaf_ru( fabsf( p.x ), fabsf( im.row2.x ),
                                __fmaf_ru( fabsf( p.y ), fabsf( im.row2.y ), __fmul_ru( fabsf( p.z ), fabsf( im.row2.z ) ) ) ) ),
                   __fmul_ru( c0, fabsf( im.row2.w ) ) );

    return q;
}

// error bounds on transformed position, plus a custom error term
// the matrix is assumed to be exact
OTK_INLINE __device__ float3 getWorldToObjectPreTranslatedTransformedPositionError( const Matrix3x4& m,
                                                                                    const float3& p,  // source position
                                                                                    const float3& carry_err )  // custom error term
{
    constexpr float c = 1.788139769587360206060111522674560546875E-7f;

    float3 q;
    q.x = __fadd_rn( p.x, m.row0.w );
    q.y = __fadd_rn( p.y, m.row1.w );
    q.z = __fadd_rn( p.z, m.row2.w );

    float3 err;
    err.x = __fmaf_ru( c,
                       ( __fmaf_ru( fabsf( q.x ), fabsf( m.row0.x ),
                                    __fmaf_ru( fabsf( q.y ), fabsf( m.row0.y ), __fmul_ru( fabsf( q.z ), fabsf( m.row0.z ) ) ) ) ),
                       carry_err.x );
    err.y = __fmaf_ru( c,
                       ( __fmaf_ru( fabsf( q.x ), fabsf( m.row1.x ),
                                    __fmaf_ru( fabsf( q.y ), fabsf( m.row1.y ), __fmul_ru( fabsf( q.z ), fabsf( m.row1.z ) ) ) ) ),
                       carry_err.y );
    err.z = __fmaf_ru( c,
                       ( __fmaf_ru( fabsf( q.x ), fabsf( m.row2.x ),
                                    __fmaf_ru( fabsf( q.y ), fabsf( m.row2.y ), __fmul_ru( fabsf( q.z ), fabsf( m.row2.z ) ) ) ) ),
                       carry_err.z );

    return err;
}

// error bounds on transformed position, plus a custom error term
// the matrix is assumed to be exact
OTK_INLINE __device__ float3 getWorldToObjectTransformedPositionError( const Matrix3x4& m,
                                                                       const float3&    p,        // source position
                                                                       const float3& carry_err )  // custom error term
{
    constexpr float c0 = 1.19209317972490680404007434844970703125E-7f;
    constexpr float c1 = 1.19209317972490680404007434844970703125E-7f;

    float3 err;
    err.x = __fmaf_ru( c1,
                       ( __fmaf_ru( fabsf( p.x ), fabsf( m.row0.x ),
                                    __fmaf_ru( fabsf( p.y ), fabsf( m.row0.y ), __fmul_ru( fabsf( p.z ), fabsf( m.row0.z ) ) ) ) ),
                       __fmaf_ru( c0, fabsf( m.row0.w ), carry_err.x ) );
    err.y = __fmaf_ru( c1,
                       ( __fmaf_ru( fabsf( p.x ), fabsf( m.row1.x ),
                                    __fmaf_ru( fabsf( p.y ), fabsf( m.row1.y ), __fmul_ru( fabsf( p.z ), fabsf( m.row1.z ) ) ) ) ),
                       __fmaf_ru( c0, fabsf( m.row1.w ), carry_err.y ) );
    err.z = __fmaf_ru( c1,
                       ( __fmaf_ru( fabsf( p.x ), fabsf( m.row2.x ),
                                    __fmaf_ru( fabsf( p.y ), fabsf( m.row2.y ), __fmul_ru( fabsf( p.z ), fabsf( m.row2.z ) ) ) ) ),
                       __fmaf_ru( c0, fabsf( m.row2.w ), carry_err.z ) );

    return err;
}

// transform a direction by a matrix
OTK_INLINE __device__ float3 transformDirection( const Matrix3x4& m, const float3& d )
{
    float3 q;
    q.x = __fmaf_rn( m.row0.x, d.x, __fmaf_rn( m.row0.y, d.y, __fmul_rn( m.row0.z, d.z ) ) );
    q.y = __fmaf_rn( m.row1.x, d.x, __fmaf_rn( m.row1.y, d.y, __fmul_rn( m.row1.z, d.z ) ) );
    q.z = __fmaf_rn( m.row2.x, d.x, __fmaf_rn( m.row2.y, d.y, __fmul_rn( m.row2.z, d.z ) ) );

    return q;
}

// transform a normal by the transpose of a matrix
OTK_INLINE __device__ float3 transposeTransformNormal( const Matrix3x4& m, const float3& n )
{
    float3 q;
    q.x = __fmaf_rn( m.row0.x, n.x, __fmaf_rn( m.row1.x, n.y, __fmul_rn( m.row2.x, n.z ) ) );
    q.y = __fmaf_rn( m.row0.y, n.x, __fmaf_rn( m.row1.y, n.y, __fmul_rn( m.row2.y, n.z ) ) );
    q.z = __fmaf_rn( m.row0.z, n.x, __fmaf_rn( m.row1.z, n.y, __fmul_rn( m.row2.z, n.z ) ) );

    return q;
}

#ifndef OTK_SIA_DISABLE_TRANSFORM_TRAVERSABLES

// transform vector by quaternion
OTK_INLINE __device__ float3 rotateVectorByQuaternion( const float3& v, const OptixSRTData& srt )
{
    const float3 u = make_float3( srt.qx, srt.qy, srt.qz );

    const float s = srt.qw;

    const float  udotv   = dot_rn( u, v );
    const float  udotu   = dot_rn( u, u );
    const float3 ucrossv = SelfIntersectionAvoidance::cross( u, v );

    const float c0 = __fmul_rn( 2.f, udotv );
    const float c1 = __fmaf_rn( s, s, -udotu );
    const float c2 = __fmul_rn( 2.f, s );

    float3 vprime;
    vprime.x = __fmaf_rn( c0, u.x, __fmaf_rn( c1, v.x, __fmul_rn( c2, ucrossv.x ) ) );
    vprime.y = __fmaf_rn( c0, u.y, __fmaf_rn( c1, v.y, __fmul_rn( c2, ucrossv.y ) ) );
    vprime.z = __fmaf_rn( c0, u.z, __fmaf_rn( c1, v.z, __fmul_rn( c2, ucrossv.z ) ) );

    return vprime;
}

// transform vector by quaternion q^-1, given q
OTK_INLINE __device__ float3 rotateVectorByQuaternionInversed( const float3& v, const OptixSRTData& srt )
{
    const float3 u = make_float3( srt.qx, srt.qy, srt.qz );

    const float s = srt.qw;

    const float udotv = dot_rn( u, v );
    const float udotu = dot_rn( u, u );
    // v x u instead of u x v for inverted
    const float3 ucrossv = SelfIntersectionAvoidance::cross( v, u );

    const float c0 = __fmul_rn( 2.f, udotv );
    const float c1 = __fmaf_rn( s, s, -udotu );
    const float c2 = __fmul_rn( 2.f, s );

    float3 vprime;
    vprime.x = __fmaf_rn( c0, u.x, __fmaf_rn( c1, v.x, __fmul_rn( c2, ucrossv.x ) ) );
    vprime.y = __fmaf_rn( c0, u.y, __fmaf_rn( c1, v.y, __fmul_rn( c2, ucrossv.y ) ) );
    vprime.z = __fmaf_rn( c0, u.z, __fmaf_rn( c1, v.z, __fmul_rn( c2, ucrossv.z ) ) );

    return vprime;
}

// Approximate bounds both the object to world and subsequent world to object transform error.
OTK_INLINE __device__ float3 getObjectToWorldTransformedPositionAndBiError( float3& outError, const OptixSRTData& srt, float3 p )
{
    // shear error
    const float px_err =
        __fmaf_rn( fabsf( p.x ), fabsf( srt.sx ),
                   __fmaf_rn( fabsf( p.y ), fabsf( srt.a ), __fmaf_rn( fabsf( p.z ), fabsf( srt.b ), fabsf( srt.pvx ) ) ) );
    const float py_err = __fmaf_rn( fabsf( p.y ), fabsf( srt.sy ), __fmaf_rn( fabsf( p.z ), fabsf( srt.c ), fabsf( srt.pvy ) ) );
    const float pz_err = __fmaf_rn( fabsf( p.z ), fabsf( srt.sz ), fabsf( srt.pvz ) );

    // apply shear
    p.x = __fmaf_rn( p.x, srt.sx, __fmaf_rn( p.z, srt.b, __fmaf_rn( p.y, srt.a, srt.pvx ) ) );
    p.y = __fmaf_rn( p.y, srt.sy, __fmaf_rn( p.z, srt.c, srt.pvy ) );
    p.z = __fmaf_rn( p.z, srt.sz, srt.pvz );

    // apply rotation
    p = rotateVectorByQuaternion( p, srt );

    constexpr float c0 = 4.7683744242021930404007434844970703125E-7f;
    constexpr float c1 = 5.9604644775390625E-8f;
    constexpr float c2 = 2.38418664366690791212022304534912109375E-7f;

    // axis independent O2W + W2O shear error
    // because of rotation independence, this error bound is bidirectional
    const float rot_err = __fadd_ru( __fadd_ru( px_err, py_err ), pz_err );
    // axis independent W2O translation error
    const float tra_err = __fadd_ru( __fadd_ru( fabsf( p.x ), fabsf( p.y ) ), fabsf( p.z ) );

    // rotation independent bidirectional error
    const float wld_err = __fmaf_ru( c0, rot_err, __fmul_ru( c2, tra_err ) );

    // apply translation
    p.x = __fadd_rn( p.x, srt.tx );
    p.y = __fadd_rn( p.y, srt.ty );
    p.z = __fadd_rn( p.z, srt.tz );

    // rotation independent error plus O2W translation error
    outError.x = __fmaf_rn( c1, fabsf( p.x ), wld_err );
    outError.y = __fmaf_rn( c1, fabsf( p.y ), wld_err );
    outError.z = __fmaf_rn( c1, fabsf( p.z ), wld_err );
    return p;
}

// transform a direction by an srt
OTK_INLINE __device__ float3 transformDirection( const OptixSRTData& srt, float3 d )
{
    // apply shear
    d.x = d.x * srt.sx + d.y * srt.a + d.z * srt.b;
    d.y = d.y * srt.sy + d.z * srt.c;
    d.z = d.z * srt.sz;

    // apply rotation
    d = rotateVectorByQuaternion( d, srt );

    return d;
}

// transform a normal by the transpose inverse of an srt
OTK_INLINE __device__ float3 inverseTransposeTransformNormal( const OptixSRTData& srt, float3 n )
{
    // invert the transposed shear/scale
    const float is00 = srt.sy * srt.sz;
    const float is10 = -srt.sz * srt.a;
    const float is20 = srt.a * srt.c - srt.sy * srt.b;  // __fmmsf_rn( srt.a, srt.c, srt.sy, srt.b ); // ( srt.a * srt.c - srt.sy * srt.b );
    const float is11 = srt.sx * srt.sz;
    const float is21 = -srt.c * srt.sx;
    const float is22 = srt.sx * srt.sy;

    // multiply normal by transpose of the inverse
    n.z = is20 * n.x + is21 * n.y + is22 * n.z;
    n.y = is10 * n.x + is11 * n.y;
    n.x = is00 * n.x;

    // apply rotation
    n = rotateVectorByQuaternion( n, srt );

    // multiply by inverse determinant
    const float det3 = srt.sx * srt.sy * srt.sz;
    const float inv_det3 = __frcp_approx( det3 );

    n.x *= inv_det3;
    n.y *= inv_det3;
    n.z *= inv_det3;

    return n;
}

// Inverts matrix. The upper left 3x3 sub matrix is inverted. The last translational column is negated.
// The inverse is applied by first adding in the inverse translation followed by multiplying by the inverse 3x3 sub matrix.
// This reduces the error in the inverse translation.
OTK_INLINE __device__ Matrix3x4 invertMatrixPreTranslate( const Matrix3x4& m )
{
    const float det3 = m.row0.x * ( m.row1.y * m.row2.z - m.row1.z * m.row2.y )
                       - m.row0.y * ( m.row1.x * m.row2.z - m.row1.z * m.row2.x )
                       + m.row0.z * ( m.row1.x * m.row2.y - m.row1.y * m.row2.x );

    const float inv_det3 = __frcp_approx( det3 );

    Matrix3x4 o;
    o.row0.x = inv_det3 * ( m.row1.y * m.row2.z - m.row2.y * m.row1.z );
    o.row0.y = inv_det3 * ( m.row0.z * m.row2.y - m.row2.z * m.row0.y );
    o.row0.z = inv_det3 * ( m.row0.y * m.row1.z - m.row1.y * m.row0.z );
    o.row0.w = -m.row0.w;

    o.row1.x = inv_det3 * ( m.row1.z * m.row2.x - m.row2.z * m.row1.x );
    o.row1.y = inv_det3 * ( m.row0.x * m.row2.z - m.row2.x * m.row0.z );
    o.row1.z = inv_det3 * ( m.row0.z * m.row1.x - m.row1.z * m.row0.x );
    o.row1.w = -m.row1.w;

    o.row2.x = inv_det3 * ( m.row1.x * m.row2.y - m.row2.x * m.row1.y );
    o.row2.y = inv_det3 * ( m.row0.y * m.row2.x - m.row2.y * m.row0.x );
    o.row2.z = inv_det3 * ( m.row0.x * m.row1.y - m.row1.x * m.row0.y );
    o.row2.w = -m.row2.w;

    return o;
}

template <typename Transform>
OTK_INLINE __device__ void optixGetSrtFromTransforms( OptixSRTData& srt, const Transform& transform, const float time )
{
    const OptixSRTMotionTransform* const __restrict transformData = transform.getSRTMotionTransformFromHandle();

    // Compute key and intra key time
    float keyTime;
    int   key;
    optix_impl::optixResolveMotionKey( keyTime, key, optix_impl::optixLoadReadOnlyAlign16( transformData ).motionOptions, time );

    // Get pointer to left key
    const float4* const __restrict dataPtr = reinterpret_cast<const float4*>( &transformData->srtData[key] );

    // Load and interpolated SRT keys
    float4 data[4];
    optix_impl::optixLoadInterpolatedSrtKey( data[0], data[1], data[2], data[3], dataPtr, keyTime );

    srt = { data[0].x, data[0].y, data[0].z, data[0].w, data[1].x, data[1].y, data[1].z, data[1].w,
            data[2].x, data[2].y, data[2].z, data[2].w, data[3].x, data[3].y, data[3].z, data[3].w };

    const float inv_length = rsqrtf( srt.qx * srt.qx + srt.qy * srt.qy + srt.qz * srt.qz + srt.qw * srt.qw );
    srt.qx *= inv_length;
    srt.qy *= inv_length;
    srt.qz *= inv_length;
    srt.qw *= inv_length;
}

#endif

// load the forward and inverse transformation matrix from a matrix based transform
template <typename Transform>
OTK_INLINE __device__ void optixGetMatrixFromTransforms( Matrix3x4&               o2w,
                                                         Matrix3x4&               w2o,
                                                         const Transform&         transform,
                                                         const OptixTransformType type,
                                                         const float              time )
{
#ifndef OTK_SIA_DISABLE_TRANSFORM_TRAVERSABLES
    if( type == OPTIX_TRANSFORM_TYPE_MATRIX_MOTION_TRANSFORM )
    {
        const OptixMatrixMotionTransform* const __restrict transformData = transform.getMatrixMotionTransformFromHandle();

        optix_impl::optixGetInterpolatedTransformation( o2w.row0, o2w.row1, o2w.row2, transformData, time );
        w2o = invertMatrixPreTranslate( o2w );
    }
    else
#endif
    {
#ifndef OTK_SIA_DISABLE_TRANSFORM_TRAVERSABLES
        if( type == OPTIX_TRANSFORM_TYPE_STATIC_TRANSFORM )
        {
            const OptixStaticTransform* const __restrict transformData = transform.getStaticTransformFromHandle();

            o2w = optix_impl::optixLoadReadOnlyAlign16( reinterpret_cast< const Matrix3x4* >( transformData->transform ) );
            w2o = optix_impl::optixLoadReadOnlyAlign16( reinterpret_cast< const Matrix3x4* >( transformData->invTransform ) );
        }
        else
#endif
        {
            assert( type == OPTIX_TRANSFORM_TYPE_INSTANCE );

            o2w = transform.getInstanceTransformFromHandle();
            w2o = transform.getInstanceInverseTransformFromHandle();
        }
    }
}

OTK_INLINE __device__ void getSafeTriangleSpawnOffset( float3& outPosition,  // [out] surface position in object space
                                                       float3& outNormal,  // [out] unit length surface normal in object space
                                                       float& outOffset,  // [out] safe offset along normal in object space
                                                       const float3& v0,     // triangle vertex 0 in object space
                                                       const float3& v1,     // triangle vertex 1 in object space
                                                       const float3& v2,     // triangle vertex 2 in object space
                                                       const float2& bary )  // barycentric coordinates
{
    // construct triangle edges
    const float3 e1 = make_float3( __fsub_rn( v1.x, v0.x ), __fsub_rn( v1.y, v0.y ), __fsub_rn( v1.z, v0.z ) );
    const float3 e2 = make_float3( __fsub_rn( v2.x, v0.x ), __fsub_rn( v2.y, v0.y ), __fsub_rn( v2.z, v0.z ) );

    // interpolate triangle point
    float3 tri_err;
    float3 obj_p = getTrianglePointAndError( tri_err, v0, e1, e2, bary );
    float3          obj_n = SelfIntersectionAvoidance::normalize( SelfIntersectionAvoidance::cross( e1, e2 ) );

    outPosition = obj_p;
    outNormal   = obj_n;
    outOffset   = dot_abs_rn( tri_err, obj_n );
}

// generate safe world space spawn point offset for an object space hitpoint and offset transformed by a chain of instances
template <typename TLIST>
OTK_INLINE __device__ void safeInstancedSpawnOffsetImpl( float3& outPosition,  // [out] surface position in world space
                                                         float3& outNormal,  // [out] unit length surface normal in world space
                                                         float& outOffset,  // [out] safe offset along normal in world space
                                                         float3       obj_p,       // object space hit point
                                                         float3       obj_n,       // unit length object space normal
                                                         float        obj_offset,  // object space offset
                                                         const float  time,        // motion time
                                                         const TLIST& transformList )  // abstract transform list
{
    // number of instances in the chain
    const unsigned int numTransforms = transformList.getTransformListSize();

    if( numTransforms )
    {
        // carry object space error bound for multi-level transform lists
        float3 carry_err = {};

#ifdef __CUDACC__
#pragma unroll 1
#endif
        for( unsigned int i = 0; i < numTransforms; ++i )
        {
            // query the transform from the list
            typename TLIST::value_t  transform = transformList.getTransform( numTransforms - i - 1 );
            const OptixTransformType type      = transform.getTransformTypeFromHandle();

            float3 obj_err, wld_err;
            float3 wld_n, wld_p;

#ifndef OTK_SIA_DISABLE_TRANSFORM_TRAVERSABLES
            if( type == OPTIX_TRANSFORM_TYPE_SRT_MOTION_TRANSFORM )
            {
                // query the srt data from the transform
                OptixSRTData srt;
                optixGetSrtFromTransforms( srt, transform, time );

                // transform object space point to world space and compute bidirectional (O2W + W2O) transformation error bounds in world space
                wld_p = getObjectToWorldTransformedPositionAndBiError( wld_err, srt, obj_p );

                // compute world space surface normal
                wld_n = inverseTransposeTransformNormal( srt, obj_n );

                // error carried from previous iteration is the only object space error
                obj_err = carry_err;
            }
            else
#endif
            {
                // query the matrices from the transform
                Matrix3x4 o2w, w2o;
                optixGetMatrixFromTransforms( o2w, w2o, transform, type, time );

                // transform object space point to world space and compute transformation error bounds in world space
                wld_p = getObjectToWorldTransformedPositionAndError( wld_err, o2w, obj_p );

                // compute world space surface normal
                wld_n = transposeTransformNormal( w2o, obj_n );

                // combine object space error bounds carried from previous iteration with error bounds for point world to object transform
                // static and instance world to object transforms use post translation
                if( type != OPTIX_TRANSFORM_TYPE_MATRIX_MOTION_TRANSFORM )
                {
                    obj_err = getWorldToObjectTransformedPositionError( w2o, wld_p, carry_err );
                }
                else
                {
                    obj_err = getWorldToObjectPreTranslatedTransformedPositionError( w2o, wld_p, carry_err );
                }
            }

            // accumulate object space error into object space offset
            obj_offset = __fadd_ru( dot_abs_rn( obj_err, obj_n ), obj_offset );

            // advance to the next transform
            carry_err  = wld_err;
            obj_n      = wld_n;
            obj_p      = wld_p;
        }

        // normalize object space normal
        const float obj_dot = dot_rn( obj_n, obj_n );
        const float obj_rcp = rsqrtf( obj_dot );
        obj_n               = make_float3( __fmul_rn( obj_n.x, obj_rcp ), __fmul_rn( obj_n.y, obj_rcp ), __fmul_rn( obj_n.z, obj_rcp ) );

        // add the world space error carried over from the last iteration
        // scale object space offset, accounting for world space normal normalization
        obj_offset = __fmaf_rn( obj_offset, obj_rcp, dot_abs_rn( carry_err, obj_n ) );
    }

    // set offset output
    outOffset   = obj_offset;
    outPosition = obj_p;
    outNormal   = obj_n;
}

// generate safe world space spawn point offset on a triangle transformed by a chain of instances
// WARNING: does not support motion transforms
// WARNING: only supports hits of type triangle
template <typename TLIST>
OTK_INLINE __device__ void safeInstancedTriangleSpawnOffsetImpl( float3& outPosition,  // [out] surface position in world space
                                                                 float3& outNormal,  // [out] surface normal in world space
                                                                 float& outOffset,  // [out] safe offset along normal in world space
                                                                 const float3& v0,  // triangle vertex 0 in object space
                                                                 const float3& v1,  // triangle vertex 1 in object space
                                                                 const float3& v2,  // triangle vertex 2 in object space
                                                                 const float2& bary,  // barycentric coordinates
                                                                 const float   time,  // motion time
                                                                 const TLIST& transformList )  // abstract transform list
{
    float3 obj_p, obj_n;
    float  obj_offset;
    getSafeTriangleSpawnOffset( obj_p, obj_n, obj_offset, v0, v1, v2, bary );

    safeInstancedSpawnOffsetImpl<TLIST>( outPosition, outNormal, outOffset, obj_p, obj_n, obj_offset, time, transformList );
}

}  // namespace SelfIntersectionAvoidance
