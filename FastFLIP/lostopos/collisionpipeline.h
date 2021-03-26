// ---------------------------------------------------------
//
//  collisionpipeline.h
//  Tyson Brochu 2011
//  Christopher Batty, Fang Da 2014
//
//  Encapsulates all collision detection and resolution functions.
//
// ---------------------------------------------------------

#ifndef LOSTOPOS_COLLISIONPIPELINE_H
#define LOSTOPOS_COLLISIONPIPELINE_H

#include <deque>
#include <options.h>
#include <vec.h>

namespace LosTopos {

class BroadPhase;
class DynamicSurface;
struct ImpactZone;

// A potentially colliding pair of primitives.  Each pair is a triple of size_ts:
//  elements 0 and 1 are the indices of the primitives involved.
//  element 2 specifies if the potential collision is point-triangle or edge-edge
//typedef std::deque<Vec3st> CollisionCandidateSet;
typedef std::vector<Vec3st> CollisionCandidateSet;

// --------------------------------------------------------
///
/// A collision between a triangle and a vertex or between two edges
///
// --------------------------------------------------------

struct Collision
{
    /// Default collision constructor
    ///
    Collision() :
    m_is_edge_edge( false ),
    m_vertex_indices( Vec4st(static_cast<size_t>(~0)) ),
    m_normal( Vec3d(UNINITIALIZED_DOUBLE) ),
    m_alphas( Vec4d(UNINITIALIZED_DOUBLE) ),
    m_relative_displacement( UNINITIALIZED_DOUBLE )
    {}   
    
    /// Collision constructor
    ///
    Collision( bool in_is_edge_edge, const Vec4st& in_vertex_indices, const Vec3d& in_normal, const Vec4d& in_alphas, double in_relative_displacement ) :
    m_is_edge_edge( in_is_edge_edge ),
    m_vertex_indices( in_vertex_indices ),
    m_normal( in_normal ),
    m_alphas( in_alphas ),
    m_relative_displacement( in_relative_displacement )
    {
        if ( !m_is_edge_edge ) { assert( m_alphas[0] == 1.0 ); }
    }
    
    /// Determine if one or more vertices is shared between this Collision and other
    ///
    inline bool overlap_vertices( const Collision& other ) const;
    
    /// Determine if ALL vertices are shared between this Collision and other
    ///
    inline bool same_vertices( const Collision& other ) const;
    
    /// Are the two elements both edges
    ///
    bool m_is_edge_edge;
    
    /// Which vertices are involved in the collision
    ///
    Vec4st m_vertex_indices;
    
    /// Collision normal
    ///
    Vec3d m_normal;
    
    /// Barycentric coordinates of the point of intersection
    ///
    Vec4d m_alphas;
    
    /// Magnitude of relative motion over the timestep
    ///
    double m_relative_displacement;
    
};


// --------------------------------------------------------
///
/// The results of processing a group of collision candidates.
///
// --------------------------------------------------------

struct ProcessCollisionStatus
{
    /// Constructor
    ///
    ProcessCollisionStatus() :
    collision_found( false ),
    overflow(false),
    all_candidates_processed( false )
    {}
    
    /// Whether one or more collisions was found.
    ///
    bool collision_found;
    
    /// Whether the number of collision candidates overflowed the candidate container.
    ///
    bool overflow;
    
    /// Whether all collision candidates were processed, or if the processing was terminated early.
    /// This is not necessarily equivalent to (!overflow): processing might stop early without overflow.
    ///
    bool all_candidates_processed;
    
};


// --------------------------------------------------------
///
/// Encapsulates all collision detection and resolution.
///
// --------------------------------------------------------

class CollisionPipeline
{
    
public:
    
    /// Constructor
    ///
    CollisionPipeline(DynamicSurface& surface,
                      BroadPhase& broadphase,
                      double in_friction_coefficient );
    
    /// Repulsion forces
    ///
    void handle_proximities( double dt );
    
    /// Sequential impulses
    ///
    bool handle_collisions( double dt );
    
    /// Get all collisions at once
    ///   
    bool detect_collisions( std::vector<Collision>& collisions );
    
    /// Get collisions involving vertices in the impact zones
    /// 
    bool detect_new_collisions( const std::vector<ImpactZone> impact_zones, 
                               std::vector<Collision>& collisions );
    
    /// Get any collisions involving an edge and a triangle
    ///
    void detect_collisions( size_t edge_index, size_t triangle_index, std::vector<Collision>& collisions );
    
    /// Re-check the elements in the specified collision objectto see if there is still a collision
    ///
    bool check_if_collision_persists( const Collision& collision );
    
    /// Friction coefficient to apply during collision resolution
    ///
    double m_friction_coefficient;
    
private: 
    
    friend class DynamicSurface;
    friend class EdgeCollapser;
    friend class MeshSnapper;

    /// Apply a collision implulse between two edges
    /// 
    void apply_edge_edge_impulse( const Collision& collision, double impulse_magnitude, double dt );

    /// Apply a collision implulse between a triangle and a vertex
    /// 
    void apply_triangle_point_impulse( const Collision& collision, double impulse_magnitude, double dt );
    
    /// Apply a collision implulse to the specified vertices, weighted by the alphas, along the specified normal
    ///     
    void apply_impulse(const Vec4d& alphas, 
                       const Vec4st& vertex_indices, 
                       double impulse_magnitude, 
                       const Vec3d& normal,
                       double dt );
    
    /// Check all triangles for AABB overlaps against the specified vertex.
    ///
    void add_point_candidates(size_t vertex_index,
                              bool return_solid,
                              bool return_dynamic,
                              CollisionCandidateSet& collision_candidates );

    /// Check all edges for AABB overlaps against the specified edge.
    ///
    void add_edge_candidates(size_t edge_index,
                             bool return_solid,
                             bool return_dynamic,
                             CollisionCandidateSet& collision_candidates );

    /// Check all edges for AABB overlaps against the specified edge.
    ///
    void add_triangle_candidates(size_t triangle_index,
                                 bool return_solid,
                                 bool return_dynamic,
                                 CollisionCandidateSet& collision_candidates );
    
    /// Called when the specified vertex is moved.  Checks all incident mesh elements for AABB overlaps.
    ///
    void add_point_update_candidates(size_t vertex_index, 
                                     CollisionCandidateSet& collision_candidates );
    
    /// Run continuous collision detection on a pair of edges.
    ///
    bool detect_segment_segment_collision( const Vec3st& candidate, Collision& collision );

    /// Run continuous collision detection on a vertex-triangle pair.
    ///
    bool detect_point_triangle_collision( const Vec3st& candidate, Collision& collision );
    
    /// Test the candidates for proximity and apply impulses
    ///
    void process_proximity_candidates( double dt,
                                      CollisionCandidateSet& candidates );
    
    /// Test dynamic points vs. solid triangles for proximities, and apply repulsion forces
    /// 
    void dynamic_point_vs_solid_triangle_proximities(double dt);

    /// Test dynamic triangles vs. all vertices for proximities, and apply repulsion forces
    /// 
    void dynamic_triangle_vs_all_point_proximities(double dt);
    
    /// Test dynamic edges vs. all edges for proximities, and apply repulsion forces
    /// 
    void dynamic_edge_vs_all_edge_proximities(double dt);  
    
    
    /// Test the candidates and fix any collisions with impulses
    ///
    void process_collision_candidates(double dt,
                                      CollisionCandidateSet& candidates,
                                      bool add_to_new_candidates,
                                      CollisionCandidateSet& new_candidates,
                                      ProcessCollisionStatus& status );
    
    /// Test the candidates and return collision info
    ///
    void test_collision_candidates(CollisionCandidateSet& candidates,
                                   std::vector<Collision>& collisions,
                                   ProcessCollisionStatus& status );
    
    /// Check if any collision exists in the set of candidates.  Stop when the first collision is found.
    /// 
    bool any_collision( CollisionCandidateSet& candidates, Collision& collision );
    
    /// Check for collisions between dynamic points and solid triangles
    ///
    void dynamic_point_vs_solid_triangle_collisions(double dt,
                                                    bool collect_candidates,
                                                    CollisionCandidateSet& update_collision_candidates,
                                                    ProcessCollisionStatus& status );
    
    /// Check for collisions between dynamic triangles and all points
    ///
    void dynamic_triangle_vs_all_point_collisions(double dt,
                                                  bool collect_candidates,
                                                  CollisionCandidateSet& update_collision_candidates,
                                                  ProcessCollisionStatus& status );
    
    /// Check for collisions between dynamic edges and all other edges 
    ///
    void dynamic_edge_vs_all_edge_collisions( double dt,
                                             bool collect_candidates,
                                             CollisionCandidateSet& update_collision_candidates,
                                             ProcessCollisionStatus& status );
    
    DynamicSurface& m_surface;
    BroadPhase& m_broadphase;
    
};

// ---------------------------------------------------------
//  Inline functions
// ---------------------------------------------------------

// --------------------------------------------------------
///
/// Determine if another collision has any vertices in common with this collision.
///
// --------------------------------------------------------

inline bool Collision::overlap_vertices( const Collision& other ) const
{
    for ( unsigned short i = 0; i < 4; ++i )
    {
        if ( m_vertex_indices[i] == other.m_vertex_indices[0] || 
            m_vertex_indices[i] == other.m_vertex_indices[1] || 
            m_vertex_indices[i] == other.m_vertex_indices[2] || 
            m_vertex_indices[i] == other.m_vertex_indices[3] )
        {
            return true;
        }
    }
    
    return false;
}

// --------------------------------------------------------
///
/// Determine if another collision has all the same vertices as this collision.
///
// --------------------------------------------------------

inline bool Collision::same_vertices( const Collision& other ) const
{
    bool found[4];
    for ( unsigned short i = 0; i < 4; ++i )
    {
        if ( m_vertex_indices[i] == other.m_vertex_indices[0] || 
            m_vertex_indices[i] == other.m_vertex_indices[1] || 
            m_vertex_indices[i] == other.m_vertex_indices[2] || 
            m_vertex_indices[i] == other.m_vertex_indices[3] )
        {
            found[i] = true;
        }
        else
        {
            found[i] = false;
        }
    }
    
    return ( found[0] && found[1] && found[2] && found[3] );
}

}

#endif
