// ---------------------------------------------------------
//
//  trianglequality.h
//  Tyson Brochu 2011
//  Christopher Batty, Fang Da 2014
//
//  Functions for getting various triangle mesh measures.
//
// ---------------------------------------------------------


#ifndef LOSTOPOS_TRIANGLEQUALITY_H
#define LOSTOPOS_TRIANGLEQUALITY_H

#include <vec.h>

// ---------------------------------------------------------
//  Forwards and typedefs
// ---------------------------------------------------------

namespace LosTopos {

class SurfTrack;

// ---------------------------------------------------------
//  Function declarations
// ---------------------------------------------------------

/// Convert radians to degrees
///
inline double rad2deg( double radians );

/// Convert degrees to radians
///
inline double deg2rad( double degrees );

/// Area of a triangle
///
inline double area( const Vec3d& v0, const Vec3d &v1, const Vec3d &v2 );

/// Radius of the circle passing through the triangle's three vertices.
///
inline double circumcircle_radius( const Vec3d& a, const Vec3d& b, const Vec3d& c );

/// Each angle within the triangle (in radians).
///
inline void triangle_angles( const Vec3d& a, const Vec3d& b, const Vec3d& c, 
                            double& angle_a, double& angle_b, double& angle_c );

/// Each angle's cosine within the triangle 
///
inline void triangle_angle_cosines(const Vec3d& a, const Vec3d& b, const Vec3d& c,
    double& cos_angle_a, double& cos_angle_b, double& cos_angle_c);

/// Minimum angle within the triangle (in radians)
///
inline double min_triangle_angle( const Vec3d& a, const Vec3d& b, const Vec3d& c );

/// Maximum angle within the triangle (in radians)
///
inline double max_triangle_angle( const Vec3d& a, const Vec3d& b, const Vec3d& c );

/// Return an estimate for mean curvature at the given vertex, computed using the Kappa * n estimate above.
/// 
double unsigned_vertex_mean_curvature( size_t vertex_index, const SurfTrack& surf );

/// 1 over the minimum radius of curvature around the given vertex
///
double inv_min_radius_curvature( const SurfTrack& surf, size_t vertex );

/// 1 over the minimum radius of curvature around each vertex
///
void compute_vertex_curvatures( const SurfTrack& surf, std::vector<double>& vertex_curvatures );

/// Return the unsigned maximum principal curvature, computed by quadric fitting
/// 
double estimated_max_curvature(const SurfTrack& surf, size_t vertex);

/// Determine the "mixed" voronoi and barycentric area of the vertex within the given triangle.
///
double mixed_area( size_t vertex_index, size_t triangle_index, const SurfTrack& surf );

/// Get the length of the specified edge, scaled by an estimate of curvature at each of the vertices.
///
double get_curvature_scaled_length(const SurfTrack& surf, 
                                   size_t vertex_a, 
                                   size_t vertex_b,
                                   double min_curvature_multiplier,
                                   double max_curvature_multiplier,
                                   double rest_curvature = 2.0 );

/// Get the length of the specified edge, scaled by an estimate of curvature at each of the vertices.
///
double get_edge_curvature(const SurfTrack& surf, 
  size_t vertex_a, 
  size_t vertex_b);

/// Get Kappa * n, the surface normal multiplied by mean curvature at the specified vertex.
///
void vertex_mean_curvature_normal( size_t vertex_index, const SurfTrack& surf, Vec3d& out, double& weight_sum );

/// Minumum of all triangle areas
///
double min_triangle_area( const SurfTrack& surf );

/// Minimum angle in all triangles (in radians)
///
double min_triangle_angle( const SurfTrack& surf );

/// Maximum angle in all triangles (in radians)
///
double max_triangle_angle( const SurfTrack& surf );

/// Number of angles below the given value (in radians)
///
size_t num_angles_below_threshold( const SurfTrack& surf, double low_threshold );

/// Number of angles above the given value (in radians)
///
size_t num_angles_above_threshold( const SurfTrack& surf, double high_threshold );

/// Compute the aspect ratio of the given triangle
///
double triangle_aspect_ratio( const SurfTrack& surf, size_t triangle_index );

/// Find the smallest triangle aspect ratio in the given mesh
///
double min_triangle_aspect_ratio( const SurfTrack& surf, size_t& output_triangle_index );

/// Find the greatest triangle aspect ratio in the given mesh
///
double max_triangle_aspect_ratio( const SurfTrack& surf, size_t& output_triangle_index );


// ---------------------------------------------------------
//  Inline functions
// ---------------------------------------------------------

// ---------------------------------------------------------
///
/// Radians to degrees
///
// ---------------------------------------------------------

inline double rad2deg( double radians )
{
    // d = r * 180 / pi
    static const double OVER_PI = 1.0 / M_PI;
    return radians * 180.0 * OVER_PI;
}

// ---------------------------------------------------------
///
/// Degrees to radians
///
// ---------------------------------------------------------

inline double deg2rad( double degrees )
{
    // r = d * pi / 180
    static const double OVER_180 = 1.0 / 180.0;
    return degrees * M_PI * OVER_180;
}

// ---------------------------------------------------------
///
/// Compute the area of the triangle specified by three vertex positions.
///
// ---------------------------------------------------------

inline double area( const Vec3d& v0, const Vec3d &v1, const Vec3d &v2 )
{
    return 0.5 * mag( cross( v1 - v0, v2 - v0 ) );
}

// ---------------------------------------------------------
///
/// Compute the radius of the circumcircle of the given triangle.
///
// ---------------------------------------------------------

inline double circumcircle_radius( const Vec3d& a, const Vec3d& b, const Vec3d& c )
{
    return 0.25 * dist(a,b) * dist(b,c) * dist(c,a) / area( a, b, c );
}

// ---------------------------------------------------------
///
/// Compute the interior angles at the vertices of the given triangle.
///
// ---------------------------------------------------------

inline void triangle_angles(const Vec3d& a, const Vec3d& b, const Vec3d& c, 
                            double& angle_a, double& angle_b, double& angle_c )
{   
   
   //Handle some degeneracies
   if(b==a) {
      angle_a = M_PI/2;
      angle_b = M_PI/2;
      angle_c = 0;
   }
   if(c==a){
      angle_a = M_PI/2;
      angle_b = 0;
      angle_c = M_PI/2;
   }
   if(b==c){    
      angle_a = 0;
      angle_b = M_PI/2;
      angle_c = M_PI/2;
   }

   angle_a = acos( std::max( -1.0, std::min( 1.0, dot( normalized(b-a), normalized(c-a) ) ) ) );
   angle_b = acos( std::max( -1.0, std::min( 1.0, dot( normalized(a-b), normalized(c-b) ) ) ) );
   angle_c = acos( std::max( -1.0, std::min( 1.0, dot( normalized(b-c), normalized(a-c) ) ) ) );
   assert(angle_a == angle_a);
   assert(angle_b == angle_b);
   assert(angle_c == angle_c);
}


inline void triangle_angle_cosines(const Vec3d& a, const Vec3d& b, const Vec3d& c,
    double& cos_angle_a, double& cos_angle_b, double& cos_angle_c)
{

    //Handle some degeneracies
    if (b == a) {
        cos_angle_a = 0;
        cos_angle_b = 0;
        cos_angle_c = 1;
    }
    if (c == a) {
        cos_angle_a = 0;
        cos_angle_b = 1;
        cos_angle_c = 0;
    }
    if (b == c) {
        cos_angle_a = 1;
        cos_angle_b = 0;
        cos_angle_c = 0;
    }

    Vec3d ba_norm = normalized(b - a);
    Vec3d ca_norm = normalized(c - a);
    Vec3d cb_norm = normalized(c - b);

    cos_angle_a = std::max(-1.0, std::min(1.0, dot(ba_norm, ca_norm)));
    cos_angle_b = std::max(-1.0, std::min(1.0, dot(-ba_norm, cb_norm)));
    cos_angle_c = std::max(-1.0, std::min(1.0, dot(-cb_norm, -ca_norm)));

    //cos_angle_a = std::max(-1.0, std::min(1.0, dot(normalized(b - a), normalized(c - a))));
    //cos_angle_b = std::max(-1.0, std::min(1.0, dot(normalized(a - b), normalized(c - b))));
    //cos_angle_c = std::max(-1.0, std::min(1.0, dot(normalized(b - c), normalized(a - c))));
    assert(cos_angle_a == cos_angle_a);
    assert(cos_angle_b == cos_angle_b);
    assert(cos_angle_c == cos_angle_c);
}

// ---------------------------------------------------------
///
/// Compute the minimum "triangle angle", defined as an interior angle at a vertex.
///
// ---------------------------------------------------------

inline double min_triangle_angle( const Vec3d& a, const Vec3d& b, const Vec3d& c )
{
    double angle_a, angle_b, angle_c;
    triangle_angles( a, b, c, angle_a, angle_b, angle_c );
    return min( angle_a, angle_b, angle_c );
}

// ---------------------------------------------------------
///
/// Compute the maximum "triangle angle", defined as an interior angle at a vertex.
///
// ---------------------------------------------------------

inline double max_triangle_angle( const Vec3d& a, const Vec3d& b, const Vec3d& c )
{
    double angle_a, angle_b, angle_c;
    triangle_angles( a, b, c, angle_a, angle_b, angle_c );
    return max( angle_a, angle_b, angle_c );   
}

// ---------------------------------------------------------
///
/// Compute the minimum and maximum "triangle angle", defined as an interior angle at a vertex.
///
// ---------------------------------------------------------
inline void min_and_max_triangle_angle(const Vec3d& a, const Vec3d& b, const Vec3d& c, Vec2d& minmax)
{
    double angle_a, angle_b, angle_c;
    triangle_angles(a, b, c, angle_a, angle_b, angle_c);
    minmax[0] = min(angle_a, angle_b, angle_c);
    minmax[1] = max(angle_a, angle_b, angle_c);
}

// ---------------------------------------------------------
///
/// Compute the minimum and maximum "triangle angle cosines", defined as a cosine of interior angle at a vertex.
///
// ---------------------------------------------------------

inline void min_and_max_triangle_angle_cosines(const Vec3d& a, const Vec3d& b, const Vec3d& c, Vec2d& minmax)
{
    double angle_a, angle_b, angle_c;
    triangle_angle_cosines(a, b, c, angle_a, angle_b, angle_c);
    minmax[0] = min(angle_a, angle_b, angle_c);
    minmax[1] = max(angle_a, angle_b, angle_c);
}

// ---------------------------------------------------------
///
/// Compute the aspect ratio of the given triangle
///
// ---------------------------------------------------------

inline double triangle_aspect_ratio( const Vec3d& a, const Vec3d& b, const Vec3d& c )
{
    
    static const double NORMALIZATION_FACTOR = 6.0 / sqrt(3.0);
    
    double len_01 = dist( b, a );
    double len_12 = dist( c, b );
    double len_20 = dist( a, c );
    double max_edge_length = max( len_01, len_12, len_20 );
    double semiperimeter = 0.5 * ( len_01 + len_12 + len_20 );
    double tri_area = area( a, b, c );
    
    return NORMALIZATION_FACTOR * tri_area / ( semiperimeter * max_edge_length );
}

}

#endif

