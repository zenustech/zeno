// ---------------------------------------------------------
//
//  impactzonesolver.h
//  Tyson Brochu 2011
//  Christopher Batty, Fang Da 2014
//
//  Encapsulates two impact zone solvers: inelastic impact zones, and rigid impact zones.
//
// ---------------------------------------------------------


#ifndef LOSTOPOS_IMPACTZONES_H
#define LOSTOPOS_IMPACTZONES_H

// ---------------------------------------------------------
//  Nested includes
// ---------------------------------------------------------

#include <collisionpipeline.h>
#include <vector>

// ---------------------------------------------------------
//  Forwards and typedefs
// ---------------------------------------------------------

namespace LosTopos {

class DynamicSurface;

// ---------------------------------------------------------
//  Class definitions
// ---------------------------------------------------------

// --------------------------------------------------------
///
/// Used in the simultaneous handling of collisions: a set of connected elements which are in collision
///
// --------------------------------------------------------

struct ImpactZone
{
    /// Constructor
    ///
    ImpactZone() :
    m_collisions(),
    m_all_solved( false )
    {}
    
    /// Get the set of all vertices in this impact zone
    ///
    void get_all_vertices( std::vector<size_t>& vertices ) const;
    
    /// Whether this ImpactZones shares vertices with other
    ///
    bool share_vertices( const ImpactZone& other ) const;
    
    /// Set of collisions with connected vertices
    ///
    std::vector<Collision> m_collisions;  
    
    /// Whether all collisions in this zone have been solved (i.e. no longer colliding)
    ///
    bool m_all_solved;
    
};


// ---------------------------------------------------------
///
/// Impact zone solver.  Handles inelastic impact zones (Harmon et al. 2008) and rigid impact zones (Bridson et al. 2002).
///
// ---------------------------------------------------------

class ImpactZoneSolver
{
    
public:
    
    /// Constructor
    ///
    ImpactZoneSolver( DynamicSurface& surface );
    
    /// Handle all collisions simultaneously by iteratively solving individual impact zones until no new collisions are detected.
    ///
    bool inelastic_impact_zones(double dt);

    ///  Rigid Impact Zones, as described in [Bridson et al. 2002].
    ///
    bool rigid_impact_zones(double dt);
    
protected:
    
    /// Iteratively project out relative normal velocities for a set of collisions in an impact zone until all collisions are solved.
    ///
    bool iterated_inelastic_projection( ImpactZone& iz, double dt );
    
    /// Project out relative normal velocities for a set of collisions in an impact zone.
    ///
    bool inelastic_projection( const ImpactZone& iz );
    
    /// Compute the best-fit rigid motion for the set of moving vertices
    ///
    bool calculate_rigid_motion(double dt, std::vector<size_t>& vs);
    
    /// The mesh this object operates on
    /// 
    DynamicSurface& m_surface;
    
    /// For rigid impact zones, treat solid vertices as having high but not infinite mass.  Use this value for mass.
    ///
    const double m_rigid_zone_infinite_mass;     
    
};


// ---------------------------------------------------------
//  Inline functions
// ---------------------------------------------------------

// --------------------------------------------------------
///
/// Extract the set of all vertices in all collisions in an ImpactZone
///
// --------------------------------------------------------

inline void ImpactZone::get_all_vertices( std::vector<size_t>& vertices ) const
{
    vertices.clear();
    for ( size_t i = 0; i < m_collisions.size(); ++i )
    {
        add_unique( vertices, m_collisions[i].m_vertex_indices[0] );
        add_unique( vertices, m_collisions[i].m_vertex_indices[1] );
        add_unique( vertices, m_collisions[i].m_vertex_indices[2] );
        add_unique( vertices, m_collisions[i].m_vertex_indices[3] );
    }
}


// --------------------------------------------------------
///
/// Determine whether another ImpactZone shares any vertices with this ImpactZone
///
// --------------------------------------------------------

inline bool ImpactZone::share_vertices( const ImpactZone& other ) const
{
    for ( size_t i = 0; i < m_collisions.size(); ++i )
    {
        for ( size_t j = 0; j < other.m_collisions.size(); ++j )
        {
            if ( m_collisions[i].overlap_vertices( other.m_collisions[j] ) )
            {
                return true;
            }
        }
    }
    
    return false;
}

}

#endif


