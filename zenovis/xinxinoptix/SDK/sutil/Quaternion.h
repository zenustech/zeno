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

#include <sutil/Matrix.h>

//------------------------------------------------------------------------------
//
// Quaternion class
//
//------------------------------------------------------------------------------

namespace sutil
{

class Quaternion
{
public:
    Quaternion()
    { q[0] = q[1] = q[2] = q[3] = 0.0; }

    Quaternion( float w, float x, float y, float z )
    { q[0] = w; q[1] = x; q[2] = y; q[3] = z; }

    Quaternion( const float3& from, const float3& to );

    Quaternion( const Quaternion& a )
    { q[0] = a[0];  q[1] = a[1];  q[2] = a[2];  q[3] = a[3]; }

    Quaternion ( float angle, const float3& axis );

    // getters and setters
    void setW(float _w)       { q[0] = _w; }
    void setX(float _x)       { q[1] = _x; }
    void setY(float _y)       { q[2] = _y; }
    void setZ(float _z)       { q[3] = _z; }
    float w() const           { return q[0]; }
    float x() const           { return q[1]; }
    float y() const           { return q[2]; }
    float z() const           { return q[3]; }


    Quaternion& operator-=(const Quaternion& r)
    { q[0] -= r[0]; q[1] -= r[1]; q[2] -= r[2]; q[3] -= r[3]; return *this; }

    Quaternion& operator+=(const Quaternion& r)
    { q[0] += r[0]; q[1] += r[1]; q[2] += r[2]; q[3] += r[3]; return *this; }

    Quaternion& operator*=(const Quaternion& r);

    Quaternion& operator/=(const float a);

    Quaternion conjugate()
    { return Quaternion( q[0], -q[1], -q[2], -q[3] ); }

    void rotation( float& angle, float3& axis ) const;
    void rotation( float& angle, float& x, float& y, float& z ) const;
    Matrix4x4 rotationMatrix() const;

    float& operator[](int i)      { return q[i]; }
    float operator[](int i)const  { return q[i]; }

    // l2 norm
    float norm() const
    { return sqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]); }

    float  normalize();

private:
    float q[4];
};


inline Quaternion::Quaternion( const float3& from, const float3& to )
{
    const float3 c = cross( from, to );
    q[0] = dot(from, to);
    q[1] = c.x;
    q[2] = c.y;
    q[3] = c.z;
}


inline Quaternion::Quaternion( float angle, const float3&  axis )
{
    const float  n        = length( axis );
    const float  inverse  = 1.0f/n;
    const float3 naxis    = axis*inverse;
    const float  s        = sinf(angle/2.0f);

    q[0] = naxis.x*s*inverse;
    q[1] = naxis.y*s*inverse;
    q[2] = naxis.z*s*inverse;
    q[3] = cosf(angle/2.0f);
}


inline Quaternion& Quaternion::operator*=(const Quaternion& r)
{

    float w = q[0]*r[0] - q[1]*r[1] - q[2]*r[2] - q[3]*r[3];
    float x = q[0]*r[1] + q[1]*r[0] + q[2]*r[3] - q[3]*r[2];
    float y = q[0]*r[2] + q[2]*r[0] + q[3]*r[1] - q[1]*r[3];
    float z = q[0]*r[3] + q[3]*r[0] + q[1]*r[2] - q[2]*r[1];

    q[0] = w;
    q[1] = x;
    q[2] = y;
    q[3] = z;
    return *this;
}


inline Quaternion& Quaternion::operator/=(const float a)
{
    float inverse = 1.0f/a;
    q[0] *= inverse;
    q[1] *= inverse;
    q[2] *= inverse;
    q[3] *= inverse;
    return *this;
}

inline void Quaternion::rotation( float& angle, float3& axis ) const
{
    Quaternion n = *this;
    n.normalize();
    axis.x = n[1];
    axis.y = n[2];
    axis.z = n[3];
    angle = 2.0f * acosf(n[0]);
}

inline void Quaternion::rotation(
        float& angle,
        float& x,
        float& y,
        float& z
        ) const
{
    Quaternion n = *this;
    n.normalize();
    x = n[1];
    y = n[2];
    z = n[3];
    angle = 2.0f * acosf(n[0]);
}

inline float Quaternion::normalize()
{
    float n = norm();
    float inverse = 1.0f/n;
    q[0] *= inverse;
    q[1] *= inverse;
    q[2] *= inverse;
    q[3] *= inverse;
    return n;
}


inline Quaternion operator*(const float a, const Quaternion &r)
{ return Quaternion(a*r[0], a*r[1], a*r[2], a*r[3]); }


inline Quaternion operator*(const Quaternion &r, const float a)
{ return Quaternion(a*r[0], a*r[1], a*r[2], a*r[3]); }


inline Quaternion operator/(const Quaternion &r, const float a)
{
    float inverse = 1.0f/a;
    return Quaternion( r[0]*inverse, r[1]*inverse, r[2]*inverse, r[3]*inverse);
}


inline Quaternion operator/(const float a, const Quaternion &r)
{
    float inverse = 1.0f/a;
    return Quaternion( r[0]*inverse, r[1]*inverse, r[2]*inverse, r[3]*inverse);
}


inline Quaternion operator-(const Quaternion& l, const Quaternion& r)
{ return Quaternion(l[0]-r[0], l[1]-r[1], l[2]-r[2], l[3]-r[3]); }


inline bool operator==(const Quaternion& l, const Quaternion& r)
{ return ( l[0] == r[0] && l[1] == r[1] && l[2] == r[2] && l[3] == r[3] ); }


inline bool operator!=(const Quaternion& l, const Quaternion& r)
{ return !(l == r); }


inline Quaternion operator+(const Quaternion& l, const Quaternion& r)
{ return Quaternion(l[0]+r[0], l[1]+r[1], l[2]+r[2], l[3]+r[3]); }


inline Quaternion operator*(const Quaternion& l, const Quaternion& r)
{
    float w = l[0]*r[0] - l[1]*r[1] - l[2]*r[2] - l[3]*r[3];
    float x = l[0]*r[1] + l[1]*r[0] + l[2]*r[3] - l[3]*r[2];
    float y = l[0]*r[2] + l[2]*r[0] + l[3]*r[1] - l[1]*r[3];
    float z = l[0]*r[3] + l[3]*r[0] + l[1]*r[2] - l[2]*r[1];
    return Quaternion( w, x, y, z );
}

inline float dot( const Quaternion& l, const Quaternion& r )
{
    return l.w()*r.w() + l.x()*r.x() + l.y()*r.y() + l.z()*r.z();
}


inline Matrix4x4 Quaternion::rotationMatrix() const
{
    Matrix4x4 m;

    const float qw = q[0];
    const float qx = q[1];
    const float qy = q[2];
    const float qz = q[3];

    m[0*4+0] = 1.0f - 2.0f*qy*qy - 2.0f*qz*qz;
    m[0*4+1] = 2.0f*qx*qy - 2.0f*qz*qw;
    m[0*4+2] = 2.0f*qx*qz + 2.0f*qy*qw;
    m[0*4+3] = 0.0f;

    m[1*4+0] = 2.0f*qx*qy + 2.0f*qz*qw;
    m[1*4+1] = 1.0f - 2.0f*qx*qx - 2.0f*qz*qz;
    m[1*4+2] = 2.0f*qy*qz - 2.0f*qx*qw;
    m[1*4+3] = 0.0f;

    m[2*4+0] = 2.0f*qx*qz - 2.0f*qy*qw;
    m[2*4+1] = 2.0f*qy*qz + 2.0f*qx*qw;
    m[2*4+2] = 1.0f - 2.0f*qx*qx - 2.0f*qy*qy;
    m[2*4+3] = 0.0f;

    m[3*4+0] = 0.0f;
    m[3*4+1] = 0.0f;
    m[3*4+2] = 0.0f;
    m[3*4+3] = 1.0f;

    return m;
}

} // end namespace sutil
