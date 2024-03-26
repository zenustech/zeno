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
// (INCLUDING NEGLIGENCE O00000R OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#pragma once 

/// \file ray_cone.h
/// Header-only library for using ray cones to drive texture filtering.  See
/// https://github.com/NVIDIA/otk-shader-util/blob/master/docs/RayCones.pdf

#include <OptiXToolkit/ShaderUtil/vec_math.h>

OTK_DEVICE const static float INV_MAX_ANISOTROPY = 1.0f/16.0f;
OTK_DEVICE const static float MAX_CONE_ANGLE = 0.25f;

struct RayCone
{
    float angle;
    float width;
};


/// Initialize a ray cone for an orthographic camera.
/// U and V are the semi-axes of the view rectangle, image_dim is the image dimensions in pixels.
OTK_INLINE OTK_HOSTDEVICE RayCone initRayConeOrthoCamera( float3 U, float3 V, uint2 image_dim )
{
    using namespace otk;
    return RayCone{0.0f, 2.0f * fminf( length( U ) / image_dim.x, length( V ) / image_dim.y )};
}

/// Initialize the ray cone for a pinhole camera. 
/// U and V are the semi-axes of the view rectangle, W is the vector from the eye to the view center,
/// image_dim is the image dimensions in pixels, D is the normalized ray direction.
OTK_INLINE OTK_HOSTDEVICE RayCone initRayConePinholeCamera( float3 U, float3 V, float3 W, uint2 image_dim, float3 D )
{
    using namespace otk;
    const float invDist = dot( D, W ) / dot( W, W );
    return RayCone{2.0f * fminf( invDist * length( U ) / image_dim.x, invDist * length( V ) / image_dim.y ), 0.0f};
}

/// Initialize the ray cone for a thin lens camera. 
/// W is the vector from the eye to the view center, D is the normalized ray direction.
OTK_INLINE OTK_HOSTDEVICE RayCone initRayConeThinLensCamera( float3 W, float lens_width, float3 D )
{
    using namespace otk;
    return RayCone{-lens_width * dot( D, W ) / dot( W, W ), lens_width};  
}

/// Propagate the cone width through the given distance.
OTK_INLINE OTK_HOSTDEVICE RayCone propagate( RayCone rayCone, float distance ) 
{ 
    RayCone rc = RayCone{rayCone.angle, rayCone.width + rayCone.angle * distance};
    if( rc.angle < 0.0f && rc.width < 0.0f ) // If angle and width are negative, the cone is diverging
        rc = RayCone{-rc.angle, -rc.width};
    return rc;
}

/// Reflect the cone angle from a mirror reflector with the given curvature.
OTK_INLINE OTK_HOSTDEVICE RayCone reflect( RayCone rayCone, float curvature ) 
{
    const float curvatureAngle = curvature * fabsf( rayCone.width );
    if ( rayCone.angle < MAX_CONE_ANGLE ) 
        return RayCone{rayCone.angle + 2.0f * curvatureAngle, rayCone.width};
    else 
        return rayCone;
}

/// Refract the cone angle from a surface with the given curvature.
OTK_INLINE OTK_HOSTDEVICE RayCone refract( RayCone rayCone, float curvature, float n_out, float n_in ) 
{
    const float curvatureAngle = curvature * fabsf( rayCone.width );
    if( rayCone.angle < MAX_CONE_ANGLE )
        return RayCone{(n_out / n_in) * (rayCone.angle + curvatureAngle) - curvatureAngle, rayCone.width};
    else
        return rayCone;
}

/// Set the ray cone angle to the max, as if from a diffuse scatter event.
OTK_INLINE OTK_HOSTDEVICE RayCone setDiffuse( RayCone rayCone ) 
{
    return RayCone{fabsf( rayCone.width ), MAX_CONE_ANGLE};
}

/// Update the cone angle for a bsdf scattering event.
OTK_INLINE OTK_HOSTDEVICE RayCone scatterBsdf( RayCone rayCone, float bsdfVal )
{
    if( rayCone.angle >= MAX_CONE_ANGLE )
        return rayCone;
    const float sgn = ( rayCone.angle < 0.0f ) ? -1.0f : 1.0f;
    const float angle = rayCone.angle + sgn * MAX_CONE_ANGLE * M_1_PIf / fmaxf( bsdfVal, M_1_PIf );
    return ( fabsf(angle) < MAX_CONE_ANGLE ) ? RayCone{angle, rayCone.width} : setDiffuse( rayCone );
}

/// Update the cone angle for a participating medium scattering event.
OTK_INLINE OTK_HOSTDEVICE void scatterPhaseFunction( RayCone rayCone, float phaseFunctionVal )
{
    scatterBsdf( rayCone, phaseFunctionVal );
}

/// Return the isotropic texture footprint width in texture space, 
/// given the world-space texture derivative lengths.
OTK_INLINE OTK_HOSTDEVICE float texFootprintWidth( float rayConeWidth, float dPdsLen, float dPdtLen ) 
{ 
    return fabsf( rayConeWidth ) / fmaxf( dPdsLen, dPdtLen ); 
}

/// Project the ray cone onto the surface to get ray differentials, 
/// given the normalized ray direction D and surface normal N.
OTK_INLINE OTK_HOSTDEVICE void projectToRayDifferentialsOnSurface( float rayConeWidth, float3 D, float3 N, float3& dPdx, float3& dPdy )
{
    using namespace otk;
    float DdotN = dot(D, N);
    dPdx = normalize( D - DdotN * N ) * ( rayConeWidth / fmaxf( fabsf( DdotN ), INV_MAX_ANISOTROPY ) );
    dPdy = normalize( cross( D, N ) ) * rayConeWidth;
}

/// Pack the ray cone into a 4 byte uint (as two bf16 values).
OTK_INLINE OTK_HOSTDEVICE unsigned int packRayCone( RayCone rayCone )
{
    unsigned int* a = (unsigned int*) &rayCone.angle;
    return ( a[0] >> 16 ) | ( a[1] & 0xffff0000 );
}

/// Unpack a packed ray cone.
OTK_INLINE OTK_HOSTDEVICE RayCone unpackRayCone( unsigned int p )
{
    RayCone rc;
    unsigned int* a = (unsigned int*) &rc.angle;
    a[0] = ( p << 16 );
    a[1] = ( p & 0xffff0000 );
    return rc;
}

/// Get the curvature of a triangle edge
OTK_INLINE OTK_HOSTDEVICE float edgeCurvature( float3 A, float3 B, float3 Na, float3 Nb )
{
    using namespace otk;
    return dot( Nb - Na, B - A ) / dot( B - A, B - A );
}

/// Get the mean curvature of a triangle based on the normals
OTK_INLINE OTK_HOSTDEVICE float meanTriangleCurvature( float3 A, float3 B, float3 C, float3 Na, float3 Nb, float3 Nc )
{
    return ( edgeCurvature( A, B, Na, Nb ) + edgeCurvature( B, C, Nb, Nc ) + edgeCurvature( A, C, Na, Nc ) ) / 3.0f;
}

/// Return the min magnitude curvature of the triangle, or zero if the curvature signs are mixed
OTK_INLINE OTK_HOSTDEVICE float minTriangleCurvature( float3 A, float3 B, float3 C, float3 Na, float3 Nb, float3 Nc )
{
    const float cab = edgeCurvature( A, B, Na, Nb );
    const float cbc = edgeCurvature( B, C, Nb, Nc );
    const float cac = edgeCurvature( A, C, Na, Nc );
    if( cab > 0.0f && cbc > 0.0f && cac > 0.0f )
        return fminf( cab, fminf( cbc, cac ) );
    if( cab < 0.0f && cbc < 0.0f && cac < 0.0f )
        return fmaxf( cab, fmaxf( cbc, cac ) );
    return 0.0f;
}

/// Compute the texture space gradients (ddx, ddy) to be used in tex2DGrad from the world space
/// texture derivatives (dPds, dPdt) and projected ray differentials (dPdx, dPdy).
OTK_INLINE OTK_HOSTDEVICE 
void computeTexGradientsFromDerivatives( float3 dPds, float3 dPdt, float3 dPdx, float3 dPdy, float2& ddx, float2& ddy )
{
    using namespace otk;
    const float dPds2 = dot(dPds, dPds);
    const float dPdt2 = dot(dPdt, dPdt);
    ddx = float2{dot(dPdx, dPds) / dPds2, dot(dPdx, dPdt) / dPdt2};
    ddy = float2{dot(dPdy, dPds) / dPds2, dot(dPdy, dPdt) / dPdt2};
}

/// Compute texture gradients for a latitude-longitude map from the cone angle.
OTK_INLINE OTK_HOSTDEVICE void computeTexGradientsForLatLongMap( float coneAngle, float2& ddx, float2& ddy )
{
    ddx = float2{coneAngle / ( 2.0f * M_PIf ), 0.0f};
    ddy = float2{0.0f, coneAngle / M_PIf};
}

/// Compute texture gradients for a cube map from the cone angle.
OTK_INLINE OTK_HOSTDEVICE void computeTexGradientsForCubeMap( float coneAngle, float2& ddx, float2& ddy )
{
    ddx = float2{coneAngle / ( 0.5f * M_PIf ), 0.0f};
    ddy = float2{0.0f, coneAngle / ( 0.5f * M_PIf )};
}

/// Compute the texture gradients (ddx, ddy) for triangle (A, B, C), with texture coordinates (Ta, Tb, Tc),
/// given world space texture footprint offset vectors dPdx and dPdy
OTK_INLINE OTK_HOSTDEVICE
void computeTexGradientsForTriangle( float3 A, float3 B, float3 C, float2 Ta, float2 Tb, float2 Tc, 
                                     float3 dPdx, float3 dPdy, float2& ddx, float2& ddy )
{
    using namespace otk;

    // Scaled normal and inverse squared area of the triangle
    const float3 ABC = cross( (B - A), (C - A) );
    const float abc2 = 1.0f / dot( ABC, ABC ); 

    // Barycentrics for dx
    const float3 P = A + dPdx;
    const float3 PBC = cross( (B - P), (C - P) );
    const float3 PCA = cross( (C - P), (A - P) );
    const float2 baryDx = float2{ dot(ABC, PBC) * abc2, dot(ABC, PCA) * abc2 };

    // Barycentrics for dy
    const float3 Q = A + dPdy;
    const float3 QBC = cross( (B - Q), (C - Q) );
    const float3 QCA = cross( (C - Q), (A - Q) );
    const float2 baryDy = float2{ dot(ABC, QBC) * abc2, dot(ABC, QCA) * abc2 };
    
    // Texture gradients based on barycentric diff
    ddx = (baryDx.x - 1.0f) * Ta + baryDx.y * Tb + (1.0f - baryDx.x - baryDx.y) * Tc;
    ddy = (baryDy.x - 1.0f) * Ta + baryDy.y * Tb + (1.0f - baryDy.x - baryDy.y) * Tc;
}

