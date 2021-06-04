// ---------------------------------------------------------
//
//  meshsmoother.h
//  Tyson Brochu 2011
//  Christopher Batty, Fang Da 2014
//
//  Functions related to the tangent-space mesh smoothing operation.
//
// ---------------------------------------------------------


#ifndef LOSTOPOS_MESHSMOOTHER_H
#define LOSTOPOS_MESHSMOOTHER_H

// ---------------------------------------------------------
//  Nested includes
// ---------------------------------------------------------

#include <cstddef>
#include <vector>
#include "util.h"

// ---------------------------------------------------------
//  Forwards and typedefs
// ---------------------------------------------------------

namespace LosTopos {

class SurfTrack;
template<unsigned int N, class T> struct Vec;
typedef Vec<3,double> Vec3d;
typedef Vec<2,size_t> Vec2st;
typedef Vec<3,size_t> Vec3st;

// ---------------------------------------------------------
//  Class definitions
// ---------------------------------------------------------

// ---------------------------------------------------------
///
/// Mesh smoother object.  Performs NULL-space constrained Laplacian smoothing of mesh vertices.
///
// ---------------------------------------------------------

class MeshSmoother
{
    
public:
    
    /// Constructor
    ///
    MeshSmoother( SurfTrack& surf ) :
    m_surf( surf ),
    m_sharp_fold_regularization_threshold(15*M_PI/180)
    {}
    
    /// NULL-space smoothing of all vertices
    ///
    bool null_space_smoothing_pass( double dt );
    
    /// Compute the maximum timestep that will not invert any triangle normals, using a quadratic solve as in [Jiao 2007].
    ///
    static double compute_max_timestep_quadratic_solve( const std::vector<Vec3st>& tris, 
                                                       const std::vector<Vec3d>& positions, 
                                                       const std::vector<Vec3d>& displacements, 
                                                       bool verbose );   
    
    /// Find a new vertex location using null-space smoothing
    ///
    void null_space_smooth_vertex( size_t v, 
                                  const std::vector<double>& triangle_areas, 
                                  const std::vector<Vec3d>& triangle_normals, 
                                  const std::vector<Vec3d>& triangle_centroids, 
                                  Vec3d& displacement ) const;      
    

private:
    
   //Helper function for computing smoothing for a subset of incident triangles
   Vec3d get_smoothing_displacement( size_t v, 
      const std::vector<size_t>& triangles,
      const std::vector<double>& triangle_areas, 
      const std::vector<Vec3d>& triangle_normals, 
      const std::vector<Vec3d>& triangle_centroids) const;

   //this version uses dihedral-angle-based feature detection
   Vec3d get_smoothing_displacement_dihedral( size_t v, 
      const std::vector<size_t>& triangles,
      const std::vector<double>& triangle_areas, 
      const std::vector<Vec3d>& triangle_normals, 
      const std::vector<Vec3d>& triangle_centroids) const;

   //this version does vanilla area-weighted smoothing, rather than tangential/null-space smoothing
   Vec3d get_smoothing_displacement_naive( size_t v, 
      const std::vector<size_t>& triangles,
      const std::vector<double>& triangle_areas, 
      const std::vector<Vec3d>& triangle_normals, 
      const std::vector<Vec3d>& triangle_centroids) const;

    /// The mesh this object operates on
    /// 
    SurfTrack& m_surf;
    
    ///
    double m_sharp_fold_regularization_threshold;

};

}

#endif

