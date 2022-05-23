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

#include <sutil/vec_math.h>
#include <sutil/Matrix.h>


#ifndef __CUDACC__
#  include <assert.h>
#  define SUTIL_AABB_ASSERT assert
#else
#  define SUTIL_AABB_ASSERT(x)
#endif


namespace sutil
{

 /**
  * @brief Axis-aligned bounding box
  *
  * <B>Description</B>
  *
  * @ref Aabb is a utility class for computing and manipulating axis-aligned
  * bounding boxes (aabbs).  Aabb is primarily useful in the bounding box
  * program associated with geometry objects. Aabb
  * may also be useful in other computation and can be used in both host
  * and device code.
  *
  */
  class Aabb
  {
  public:

    /** Construct an invalid box */
    SUTIL_HOSTDEVICE Aabb();

    /** Construct from min and max vectors */
    SUTIL_HOSTDEVICE Aabb( const float3& min, const float3& max );

    /** Construct from three points (e.g. triangle) */
    SUTIL_HOSTDEVICE Aabb( const float3& v0, const float3& v1, const float3& v2 );

    /** Exact equality */
    SUTIL_HOSTDEVICE bool operator==( const Aabb& other ) const;

    /** Array access */
    SUTIL_HOSTDEVICE float3& operator[]( int i );

    /** Const array access */
    SUTIL_HOSTDEVICE const float3& operator[]( int i ) const;

    /** Direct access */
    SUTIL_HOSTDEVICE float* data();

    /** Set using two vectors */
    SUTIL_HOSTDEVICE void set( const float3& min, const float3& max );

    /** Set using three points (e.g. triangle) */
    SUTIL_HOSTDEVICE void set( const float3& v0, const float3& v1, const float3& v2 );

    /** Invalidate the box */
    SUTIL_HOSTDEVICE void invalidate();

    /** Check if the box is valid */
    SUTIL_HOSTDEVICE bool valid() const;

    /** Check if the point is in the box */
    SUTIL_HOSTDEVICE bool contains( const float3& p ) const;

    /** Check if the box is fully contained in the box */
    SUTIL_HOSTDEVICE bool contains( const Aabb& bb ) const;

    /** Extend the box to include the given point */
    SUTIL_HOSTDEVICE void include( const float3& p );

    /** Extend the box to include the given box */
    SUTIL_HOSTDEVICE void include( const Aabb& other );

    /** Extend the box to include the given box */
    SUTIL_HOSTDEVICE void include( const float3& min, const float3& max );

    /** Compute the box center */
    SUTIL_HOSTDEVICE float3 center() const;

    /** Compute the box center in the given dimension */
    SUTIL_HOSTDEVICE float center( int dim ) const;

    /** Compute the box extent */
    SUTIL_HOSTDEVICE float3 extent() const;

    /** Compute the box extent in the given dimension */
    SUTIL_HOSTDEVICE float extent( int dim ) const;

    /** Compute the volume of the box */
    SUTIL_HOSTDEVICE float volume() const;

    /** Compute the surface area of the box */
    SUTIL_HOSTDEVICE float area() const;

    /** Compute half the surface area of the box */
    SUTIL_HOSTDEVICE float halfArea() const;

    /** Get the index of the longest axis */
    SUTIL_HOSTDEVICE int longestAxis() const;

    /** Get the extent of the longest axis */
    SUTIL_HOSTDEVICE float maxExtent() const;

    /** Check for intersection with another box */
    SUTIL_HOSTDEVICE bool intersects( const Aabb& other ) const;

    /** Make the current box be the intersection between this one and another one */
    SUTIL_HOSTDEVICE void intersection( const Aabb& other );

    /** Enlarge the box by moving both min and max by 'amount' */
    SUTIL_HOSTDEVICE void enlarge( float amount );

    SUTIL_HOSTDEVICE void transform( const Matrix3x4& m );
    SUTIL_HOSTDEVICE void transform( const Matrix4x4& m );

    /** Check if the box is flat in at least one dimension  */
    SUTIL_HOSTDEVICE bool isFlat() const;

    /** Compute the minimum Euclidean distance from a point on the
     surface of this Aabb to the point of interest */
    SUTIL_HOSTDEVICE float distance( const float3& x ) const;

    /** Compute the minimum squared Euclidean distance from a point on the
     surface of this Aabb to the point of interest */
    SUTIL_HOSTDEVICE float distance2( const float3& x ) const;

    /** Compute the minimum Euclidean distance from a point on the surface
      of this Aabb to the point of interest.
      If the point of interest lies inside this Aabb, the result is negative  */
    SUTIL_HOSTDEVICE float signedDistance( const float3& x ) const;

    /** Min bound */
    float3 m_min;
    /** Max bound */
    float3 m_max;
  };


  SUTIL_INLINE SUTIL_HOSTDEVICE Aabb::Aabb()
  {
    invalidate();
  }

  SUTIL_INLINE SUTIL_HOSTDEVICE Aabb::Aabb( const float3& min, const float3& max )
  {
    set( min, max );
  }

  SUTIL_INLINE SUTIL_HOSTDEVICE Aabb::Aabb( const float3& v0, const float3& v1, const float3& v2 )
  {
    set( v0, v1, v2 );
  }

  SUTIL_INLINE SUTIL_HOSTDEVICE bool Aabb::operator==( const Aabb& other ) const
  {
    return m_min.x == other.m_min.x &&
           m_min.y == other.m_min.y &&
           m_min.z == other.m_min.z &&
           m_max.x == other.m_max.x &&
           m_max.y == other.m_max.y &&
           m_max.z == other.m_max.z;
  }

  SUTIL_INLINE SUTIL_HOSTDEVICE float3& Aabb::operator[]( int i )
  {
    SUTIL_AABB_ASSERT( i>=0 && i<=1 );
    return (&m_min)[i];
  }

  SUTIL_INLINE SUTIL_HOSTDEVICE const float3& Aabb::operator[]( int i ) const
  {
    SUTIL_AABB_ASSERT( i>=0 && i<=1 );
    return (&m_min)[i];
  }

  SUTIL_INLINE SUTIL_HOSTDEVICE float* Aabb::data()
  {
      return &m_min.x;
  }

  SUTIL_INLINE SUTIL_HOSTDEVICE void Aabb::set( const float3& min, const float3& max )
  {
    m_min = min;
    m_max = max;
  }

  SUTIL_INLINE SUTIL_HOSTDEVICE void Aabb::set( const float3& v0, const float3& v1, const float3& v2 )
  {
    m_min = fminf( v0, fminf(v1,v2) );
    m_max = fmaxf( v0, fmaxf(v1,v2) );
  }

  SUTIL_INLINE SUTIL_HOSTDEVICE void Aabb::invalidate()
  {
    m_min = make_float3(  1e37f );
    m_max = make_float3( -1e37f );
  }

  SUTIL_INLINE SUTIL_HOSTDEVICE bool Aabb::valid() const
  {
    return m_min.x <= m_max.x &&
      m_min.y <= m_max.y &&
      m_min.z <= m_max.z;
  }

  SUTIL_INLINE SUTIL_HOSTDEVICE bool Aabb::contains( const float3& p ) const
  {
    return  p.x >= m_min.x && p.x <= m_max.x &&
            p.y >= m_min.y && p.y <= m_max.y &&
            p.z >= m_min.z && p.z <= m_max.z;
  }

  SUTIL_INLINE SUTIL_HOSTDEVICE bool Aabb::contains( const Aabb& bb ) const
  {
    return contains( bb.m_min ) && contains( bb.m_max );
  }

  SUTIL_INLINE SUTIL_HOSTDEVICE void Aabb::include( const float3& p )
  {
    m_min = fminf( m_min, p );
    m_max = fmaxf( m_max, p );
  }

  SUTIL_INLINE SUTIL_HOSTDEVICE void Aabb::include( const Aabb& other )
  {
    m_min = fminf( m_min, other.m_min );
    m_max = fmaxf( m_max, other.m_max );
  }

  SUTIL_INLINE SUTIL_HOSTDEVICE void Aabb::include( const float3& min, const float3& max )
  {
    m_min = fminf( m_min, min );
    m_max = fmaxf( m_max, max );
  }

  SUTIL_INLINE SUTIL_HOSTDEVICE float3 Aabb::center() const
  {
    SUTIL_AABB_ASSERT( valid() );
    return (m_min+m_max) * 0.5f;
  }

  SUTIL_INLINE SUTIL_HOSTDEVICE float Aabb::center( int dim ) const
  {
    SUTIL_AABB_ASSERT( valid() );
    SUTIL_AABB_ASSERT( dim>=0 && dim<=2 );
    return ( ((const float*)(&m_min))[dim] + ((const float*)(&m_max))[dim] ) * 0.5f;
  }

  SUTIL_INLINE SUTIL_HOSTDEVICE float3 Aabb::extent() const
  {
    SUTIL_AABB_ASSERT( valid() );
    return m_max - m_min;
  }

  SUTIL_INLINE SUTIL_HOSTDEVICE float Aabb::extent( int dim ) const
  {
    SUTIL_AABB_ASSERT( valid() );
    return ((const float*)(&m_max))[dim] - ((const float*)(&m_min))[dim];
  }

  SUTIL_INLINE SUTIL_HOSTDEVICE float Aabb::volume() const
  {
    SUTIL_AABB_ASSERT( valid() );
    const float3 d = extent();
    return d.x*d.y*d.z;
  }

  SUTIL_INLINE SUTIL_HOSTDEVICE float Aabb::area() const
  {
    return 2.0f * halfArea();
  }

  SUTIL_INLINE SUTIL_HOSTDEVICE float Aabb::halfArea() const
  {
    SUTIL_AABB_ASSERT( valid() );
    const float3 d = extent();
    return d.x*d.y + d.y*d.z + d.z*d.x;
  }

  SUTIL_INLINE SUTIL_HOSTDEVICE int Aabb::longestAxis() const
  {
    SUTIL_AABB_ASSERT( valid() );
    const float3 d = extent();

    if( d.x > d.y )
      return d.x > d.z ? 0 : 2;
    return d.y > d.z ? 1 : 2;
  }

  SUTIL_INLINE SUTIL_HOSTDEVICE float Aabb::maxExtent() const
  {
    return extent( longestAxis() );
  }

  SUTIL_INLINE SUTIL_HOSTDEVICE bool Aabb::intersects( const Aabb& other ) const
  {
    if( other.m_min.x > m_max.x || other.m_max.x < m_min.x ) return false;
    if( other.m_min.y > m_max.y || other.m_max.y < m_min.y ) return false;
    if( other.m_min.z > m_max.z || other.m_max.z < m_min.z ) return false;
    return true;
  }

  SUTIL_INLINE SUTIL_HOSTDEVICE void Aabb::intersection( const Aabb& other )
  {
    m_min.x = fmaxf( m_min.x, other.m_min.x );
    m_min.y = fmaxf( m_min.y, other.m_min.y );
    m_min.z = fmaxf( m_min.z, other.m_min.z );
    m_max.x = fminf( m_max.x, other.m_max.x );
    m_max.y = fminf( m_max.y, other.m_max.y );
    m_max.z = fminf( m_max.z, other.m_max.z );
  }

  SUTIL_INLINE SUTIL_HOSTDEVICE void Aabb::enlarge( float amount )
  {
    SUTIL_AABB_ASSERT( valid() );
    m_min -= make_float3( amount );
    m_max += make_float3( amount );
  }

    SUTIL_INLINE SUTIL_HOSTDEVICE void Aabb::transform( const Matrix3x4& m )
  {
    // row-major matrix -> column vectors:
    // x ={ m[0], m[4], m[8] }
    // y ={ m[1], m[5], m[9] }
    // z ={ m[2], m[6], m[10] }
    // 3,7,11 translation

    // no need to initialize, will be overwritten completely
    Aabb result;
    const float loxx = m[0] * m_min.x;
    const float hixx = m[0] * m_max.x;
    const float loyx = m[1] * m_min.y;
    const float hiyx = m[1] * m_max.y;
    const float lozx = m[2] * m_min.z;
    const float hizx = m[2] * m_max.z;
    result.m_min.x = fminf( loxx, hixx ) + fminf( loyx, hiyx ) + fminf( lozx, hizx ) + m[3];
    result.m_max.x = fmaxf( loxx, hixx ) + fmaxf( loyx, hiyx ) + fmaxf( lozx, hizx ) + m[3];
    const float loxy = m[4] * m_min.x;
    const float hixy = m[4] * m_max.x;
    const float loyy = m[5] * m_min.y;
    const float hiyy = m[5] * m_max.y;
    const float lozy = m[6] * m_min.z;
    const float hizy = m[6] * m_max.z;
    result.m_min.y = fminf( loxy, hixy ) + fminf( loyy, hiyy ) + fminf( lozy, hizy ) + m[7];
    result.m_max.y = fmaxf( loxy, hixy ) + fmaxf( loyy, hiyy ) + fmaxf( lozy, hizy ) + m[7];
    const float loxz = m[8] * m_min.x;
    const float hixz = m[8] * m_max.x;
    const float loyz = m[9] * m_min.y;
    const float hiyz = m[9] * m_max.y;
    const float lozz = m[10] * m_min.z;
    const float hizz = m[10] * m_max.z;
    result.m_min.z = fminf( loxz, hixz ) + fminf( loyz, hiyz ) + fminf( lozz, hizz ) + m[11];
    result.m_max.z = fmaxf( loxz, hixz ) + fmaxf( loyz, hiyz ) + fmaxf( lozz, hizz ) + m[11];
    *this = result;
  }

  SUTIL_INLINE SUTIL_HOSTDEVICE void Aabb::transform( const Matrix4x4& m )
  {
      const float3 b000 = m_min;
      const float3 b001 = make_float3( m_min.x, m_min.y, m_max.z );
      const float3 b010 = make_float3( m_min.x, m_max.y, m_min.z );
      const float3 b011 = make_float3( m_min.x, m_max.y, m_max.z );
      const float3 b100 = make_float3( m_max.x, m_min.y, m_min.z );
      const float3 b101 = make_float3( m_max.x, m_min.y, m_max.z );
      const float3 b110 = make_float3( m_max.x, m_max.y, m_min.z );
      const float3 b111 = m_max;

      invalidate();
      include( make_float3( m*make_float4( b000, 1.0f ) ) );
      include( make_float3( m*make_float4( b001, 1.0f ) ) );
      include( make_float3( m*make_float4( b010, 1.0f ) ) );
      include( make_float3( m*make_float4( b011, 1.0f ) ) );
      include( make_float3( m*make_float4( b100, 1.0f ) ) );
      include( make_float3( m*make_float4( b101, 1.0f ) ) );
      include( make_float3( m*make_float4( b110, 1.0f ) ) );
      include( make_float3( m*make_float4( b111, 1.0f ) ) );
  }

  SUTIL_INLINE SUTIL_HOSTDEVICE bool Aabb::isFlat() const
  {
    return m_min.x == m_max.x ||
           m_min.y == m_max.y ||
           m_min.z == m_max.z;
  }

  SUTIL_INLINE SUTIL_HOSTDEVICE float Aabb::distance( const float3& x ) const
  {
    return sqrtf(distance2(x));
  }

  SUTIL_INLINE SUTIL_HOSTDEVICE float Aabb::signedDistance( const float3& x ) const
  {
    if( m_min.x <= x.x && x.x <= m_max.x &&
        m_min.y <= x.y && x.y <= m_max.y &&
        m_min.z <= x.z && x.z <= m_max.z) {
      float distance_x = fminf( x.x - m_min.x, m_max.x - x.x);
      float distance_y = fminf( x.y - m_min.y, m_max.y - x.y);
      float distance_z = fminf( x.z - m_min.z, m_max.z - x.z);

      float min_distance = fminf(distance_x, fminf(distance_y, distance_z));
      return -min_distance;
    }

    return distance(x);
  }

  SUTIL_INLINE SUTIL_HOSTDEVICE float Aabb::distance2( const float3& x ) const
  {
    float3 box_dims = m_max - m_min;

    // compute vector from min corner of box
    float3 v = x - m_min;

    float dist2 = 0;
    float excess;

    // project vector from box min to x on each axis,
    // yielding distance to x along that axis, and count
    // any excess distance outside box extents

    excess = 0;
    if( v.x < 0 )
      excess = v.x;
    else if( v.x > box_dims.x )
      excess = v.x - box_dims.x;
    dist2 += excess * excess;

    excess = 0;
    if( v.y < 0 )
      excess = v.y;
    else if( v.y > box_dims.y )
      excess = v.y - box_dims.y;
    dist2 += excess * excess;

    excess = 0;
    if( v.z < 0 )
      excess = v.z;
    else if( v.z > box_dims.z )
      excess = v.z - box_dims.z;
    dist2 += excess * excess;

    return dist2;
  }

} // end namespace sutil
