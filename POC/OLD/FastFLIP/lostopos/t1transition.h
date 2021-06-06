// ---------------------------------------------------------
//
//  t1transition.h
//  Christopher Batty, Fang Da 2014
//  
//  Functions handling T1 transitions (edge popping and vertex popping).
//
// ---------------------------------------------------------

#ifndef LOSTOPOS_T1TRANSITION_H
#define LOSTOPOS_T1TRANSITION_H

// ---------------------------------------------------------
//  Nested includes
// ---------------------------------------------------------

#include <cstddef>
#include <vector>

// ---------------------------------------------------------
//  Forwards and typedefs
// ---------------------------------------------------------

namespace LosTopos {

class SurfTrack;
template <unsigned int N, class T> struct Vec;
typedef Vec<3, double> Vec3d;
typedef Vec<2, size_t> Vec2st;
typedef Vec<3, size_t> Vec3st;
typedef Vec<2, int>    Vec2i;
typedef Vec<2, Vec2i>  Mat2i;

// ---------------------------------------------------------
//  Class definitions
// ---------------------------------------------------------

// ---------------------------------------------------------
///
/// T1Transition class. Pull apart X-junction edges and X-junction vertices
///
// ---------------------------------------------------------

class T1Transition
{
public:
    
    /// Callback to provide the velocity field
    ///
    class VelocityFieldCallback
    {
    public:
        virtual Vec3d sampleVelocity(Vec3d & pos) = 0;
        
    };
    
    typedef std::pair<size_t, bool> FaceRegion;
    
public:
    
    /// Constructor
    ///
    T1Transition(SurfTrack & surf, VelocityFieldCallback * vfc, bool remesh_boundaries);

    /// Perform a pass of t1 by vertex popping
    ///
    bool t1_pass();
    
    /// Analyze the neighborhood of a vertex, detecting the connected components of space (SCC)
    ///
    bool analyze_vertex(size_t i, std::vector<FaceRegion> & scc_list, std::vector<Vec2i> & scc_labels);
    
    /// Attempt a cut on a junction between two given regions, returning the tensile force (tendency of the two resulting vertices moving apart; positive tensile force indicates the cut can happen)
    ///
    double try_pull_vertex_apart_using_surface_tension(size_t xj, int A, int B, const std::vector<FaceRegion> & scc_list, const std::vector<Vec2i> & scc_labels, Vec3d & pull_apart_direction);
    
    /// Attempt a cut on a junction between two given regions, returning the velocity divergence along the pull apart direction
    ///
    double try_pull_vertex_apart_using_velocity_field(size_t xj, int A, int B, const std::vector<FaceRegion> & scc_list, const std::vector<Vec2i> & scc_labels, Vec3d & pull_apart_direction);
    
    /// Collision safety
    ///
    bool pulling_vertex_apart_introduces_collision(size_t v, const Vec3d & oldpos, const Vec3d & newpos0, const Vec3d & newpos1);
    

    /// Whether or not to remesh the boundary (currently no effect)
    ///
    bool m_remesh_boundaries;
  
    /// Parameters
    ///
    double m_pull_apart_distance;
    
    ///
    double m_pull_apart_tendency_threshold;
    

    
private:
    
    /// Helper data structures
    ///
    struct InteriorStencil;    
    
    /// Collision safety helper functions
    /// Move one vertex and test for collision
    ///
    bool vertex_pseudo_motion_introduces_collision(size_t v, const Vec3d & oldpos, const Vec3d & newpos);
    
    /// Move one vertex and test for collision, using only the specified subset of incident edges and triangles (ignoring the rest if any)
    ///
    bool vertex_pseudo_motion_introduces_collision(size_t v, const Vec3d & oldpos, const Vec3d & newpos, const std::vector<size_t> & tris, const std::vector<size_t> & edges);
    
    /// Vertex popping helper function: generate the new triangulation after pulling a vertex apart
    ///
    void triangulate_popped_vertex(size_t xj, int A, int B, size_t a, size_t b, const std::vector<FaceRegion> & SCC_list, const std::vector<Vec2i> & SCC_labels, std::vector<size_t> & faces_to_delete, std::vector<Vec3st> & faces_to_create, std::vector<Vec2i> & face_labels_to_create);
    
    /// The mesh this object operates on
    /// 
    SurfTrack & m_surf;   
    
    /// Velocity field callback
    VelocityFieldCallback * m_velocity_field_callback;
    
};

}

#endif
