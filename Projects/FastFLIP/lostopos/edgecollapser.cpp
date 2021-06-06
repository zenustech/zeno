// ---------------------------------------------------------
//
//  edgecollapser.cpp
//  Tyson Brochu 2011
//  Christopher Batty, Fang Da 2014
//
//  Functions supporting the "edge collapse" operation: removing short edges from the mesh.
//
// ---------------------------------------------------------

#include <edgecollapser.h>

#include <broadphase.h>
#include <collisionpipeline.h>
#include <collisionqueries.h>
#include <nondestructivetrimesh.h>
#include <runstats.h>
#include <subdivisionscheme.h>
#include <surftrack.h>
#include <trianglequality.h>

#include <algorithm>
// ---------------------------------------------------------
//  Extern globals
// ---------------------------------------------------------

namespace LosTopos {
    
extern RunStats g_stats;


// ---------------------------------------------------------
// Member function definitions
// ---------------------------------------------------------

// --------------------------------------------------------
///
/// Edge collapser constructor.  Takes a SurfTrack object and curvature-adaptive parameters.
///
// --------------------------------------------------------

EdgeCollapser::EdgeCollapser( SurfTrack& surf, bool use_curvature, bool remesh_boundaries, double min_curvature_multiplier ) :
//m_min_edge_length( UNINITIALIZED_DOUBLE ),
//m_max_edge_length( UNINITIALIZED_DOUBLE ),
m_use_curvature( use_curvature ),
m_remesh_boundaries( remesh_boundaries ),
m_min_curvature_multiplier( min_curvature_multiplier ),
m_rank_region(-1),
m_surf( surf )
{}


// --------------------------------------------------------
///
/// Get all triangles which are incident on either edge end vertex.
///
// --------------------------------------------------------

void EdgeCollapser::get_moving_triangles(size_t source_vertex,
                                         size_t destination_vertex,
                                         std::vector<size_t>& moving_triangles )
{
    
    moving_triangles.clear();
    
    for ( size_t i = 0; i < m_surf.m_mesh.m_vertex_to_triangle_map[source_vertex].size(); ++i )
    {
        moving_triangles.push_back( m_surf.m_mesh.m_vertex_to_triangle_map[source_vertex][i] );
    }
    for ( size_t i = 0; i < m_surf.m_mesh.m_vertex_to_triangle_map[destination_vertex].size(); ++i )
    {
        moving_triangles.push_back( m_surf.m_mesh.m_vertex_to_triangle_map[destination_vertex][i] );
    }
    
}


// --------------------------------------------------------
///
/// Get all edges which are incident on either edge end vertex.
///
// --------------------------------------------------------

void EdgeCollapser::get_moving_edges( size_t source_vertex,
                                     size_t destination_vertex,
                                     size_t,
                                     std::vector<size_t>& moving_edges )
{
    
    moving_edges = m_surf.m_mesh.m_vertex_to_edge_map[ source_vertex ];
    moving_edges.insert( moving_edges.end(), m_surf.m_mesh.m_vertex_to_edge_map[ destination_vertex ].begin(), m_surf.m_mesh.m_vertex_to_edge_map[ destination_vertex ].end() );
    
}


// --------------------------------------------------------
///
/// Check the "pseudo motion" introduced by a collapsing edge for collision
///
// --------------------------------------------------------

bool EdgeCollapser::collapse_edge_pseudo_motion_introduces_collision( size_t source_vertex,
                                                                     size_t destination_vertex,
                                                                     size_t edge_index,
                                                                     const Vec3d& )
{
    assert( m_surf.m_collision_safety );
    
    // Get the set of triangles which move because of this motion
    std::vector<size_t> moving_triangles;
    get_moving_triangles( source_vertex, destination_vertex, moving_triangles );
    
    // And the set of edges
    std::vector<size_t> moving_edges;
    get_moving_edges( source_vertex, destination_vertex, edge_index, moving_edges );
    
    
    // Check for collisions, holding everything static except for the source and destination vertices
    
    CollisionCandidateSet collision_candidates;
    
    // triangle-point candidates
    for ( size_t i = 0; i < moving_triangles.size(); ++i )
    {
        m_surf.m_collision_pipeline->add_triangle_candidates( moving_triangles[i], true, true, collision_candidates );
    }
    
    // point-triangle candidates
    m_surf.m_collision_pipeline->add_point_candidates( source_vertex, true, true, collision_candidates );
    m_surf.m_collision_pipeline->add_point_candidates( destination_vertex, true, true, collision_candidates );
    
    // edge-edge candidates
    for ( size_t i = 0; i < moving_edges.size(); ++i )
    {
        m_surf.m_collision_pipeline->add_edge_candidates( moving_edges[i], true, true, collision_candidates );
    }
    
    // Prune collision candidates containing both the source and destination vertex (they will trivially be collisions )
    
    for ( size_t i = 0; i < collision_candidates.size(); ++i )
    {
        const Vec3st& candidate = collision_candidates[i];
        bool should_delete = false;
        
        if ( candidate[2] == 1 )
        {
            // edge-edge
            const Vec2st& e0 = m_surf.m_mesh.m_edges[ candidate[0] ];
            const Vec2st& e1 = m_surf.m_mesh.m_edges[ candidate[1] ];
            
            if ( e0[0] == source_vertex || e0[1] == source_vertex || e1[0] == source_vertex || e1[1] == source_vertex )
            {
                if ( e0[0] == destination_vertex || e0[1] == destination_vertex || e1[0] == destination_vertex || e1[1] == destination_vertex )
                {
                    should_delete = true;
                }
            }
            
        }
        else
        {
            // point-triangle
            size_t t = candidate[0];
            const Vec3st& tri = m_surf.m_mesh.get_triangle(t);
            size_t v = candidate[1];
            
            if ( v == source_vertex || tri[0] == source_vertex || tri[1] == source_vertex || tri[2] == source_vertex )
            {
                if ( v == destination_vertex || tri[0] == destination_vertex || tri[1] == destination_vertex || tri[2] == destination_vertex )
                {
                    should_delete = true;
                }
            }
        }
        
        if ( should_delete )
        {
            collision_candidates.erase( collision_candidates.begin() + i );
            --i;
        }
        
    }
    
    Collision collision;
    if ( m_surf.m_collision_pipeline->any_collision( collision_candidates, collision ) )
    {
        return true;
    }
    
    // Also check proximity: if any proximity check returns zero distance, this collapse cannot be allowed either.
    // Because the CCD above is geometrically exact, it sometimes returns different result than the proximity check below (proximity
    //  distance = 0, but CCD says no collision). If distance is 0, the subsequent proximity handling will produce NaNs, so it is
    //  problematic too.
    
    for (size_t i = 0; i < collision_candidates.size(); i++)
    {
        const Vec3st & candidate = collision_candidates[i];
        if (candidate[2] == 1)
        {
            // edge-edge
            Vec2st e0 = m_surf.m_mesh.m_edges[candidate[0]];
            Vec2st e1 = m_surf.m_mesh.m_edges[candidate[1]];
            
            if (e0[0] == e0[1]) continue;
            if (e1[0] == e1[1]) continue;
            
            if (e0[0] != e1[0] && e0[0] != e1[1] && e0[1] != e1[0] && e0[1] != e1[1])
            {
                double distance, s0, s2;
                Vec3d normal;
                check_edge_edge_proximity(m_surf.get_newposition(e0[0]), m_surf.get_newposition(e0[1]), m_surf.get_newposition(e1[0]), m_surf.get_newposition(e1[1]), distance, s0, s2, normal);
                if (distance == 0)
                    return true;
            }
            
        } else
        {
            // point-triangle
            size_t t = candidate[0];
            const Vec3st & tri = m_surf.m_mesh.get_triangle(t);
            size_t v = candidate[1];
            
            if (tri[0] == tri[1]) continue;
            
            if (tri[0] != v && tri[1] != v && tri[2] != v)
            {
                double distance, s1, s2, s3;
                Vec3d normal;
                check_point_triangle_proximity(m_surf.get_newposition(v), m_surf.get_newposition(tri[0]), m_surf.get_newposition(tri[1]), m_surf.get_newposition(tri[2]), distance, s1, s2, s3, normal);
                if (distance == 0)
                    return true;
            }
        }
    }
    
    return false;
}


// --------------------------------------------------------
///
/// Determine if the edge collapse operation would invert the normal of any incident triangles.
///
// --------------------------------------------------------

bool EdgeCollapser::collapse_edge_introduces_normal_inversion( size_t source_vertex,
                                                              size_t destination_vertex,
                                                              size_t edge_index,
                                                              const Vec3d& vertex_new_position )
{
    
    // Get the set of triangles which are going to be deleted
    std::vector< size_t >& triangles_incident_to_edge = m_surf.m_mesh.m_edge_to_triangle_map[edge_index];
    
    // Get the set of triangles which move because of this motion
    std::vector<size_t> moving_triangles;
    for ( size_t i = 0; i < m_surf.m_mesh.m_vertex_to_triangle_map[source_vertex].size(); ++i )
    {
        moving_triangles.push_back( m_surf.m_mesh.m_vertex_to_triangle_map[source_vertex][i] );
    }
    for ( size_t i = 0; i < m_surf.m_mesh.m_vertex_to_triangle_map[destination_vertex].size(); ++i )
    {
        moving_triangles.push_back( m_surf.m_mesh.m_vertex_to_triangle_map[destination_vertex][i] );
    }
    
    double min_triangle_area = -1;
    for (size_t i = 0; i < moving_triangles.size(); i++)
    {
        Vec3st current_triangle = m_surf.m_mesh.get_triangle(moving_triangles[i]);
        double area = triangle_area(m_surf.get_position(current_triangle[0]), m_surf.get_position(current_triangle[1]), m_surf.get_position(current_triangle[2]));
        if (min_triangle_area < 0 || area < min_triangle_area)
            min_triangle_area = area;
    }
    assert(min_triangle_area > 0);
    min_triangle_area = std::min(min_triangle_area, m_surf.m_min_triangle_area);
    
    //
    // check for normal inversion
    //
    
    for ( size_t i = 0; i < moving_triangles.size(); ++i )
    {
        
        // Disregard triangles which will end up being deleted - those triangles incident to the collapsing edge.
        bool triangle_will_be_deleted = false;
        for ( size_t j = 0; j < triangles_incident_to_edge.size(); ++j )
        {
            if ( moving_triangles[i] == triangles_incident_to_edge[j] )
            {
                triangle_will_be_deleted = true;
                break;
            }
        }
        
        if ( triangle_will_be_deleted ) { continue; }
        
        const Vec3st& current_triangle = m_surf.m_mesh.get_triangle( moving_triangles[i] );
        Vec3d old_normal = m_surf.get_triangle_normal( current_triangle );
        
        Vec3d new_normal;
        
        double new_area;
        if ( current_triangle[0] == source_vertex || current_triangle[0] == destination_vertex )
        {
            new_normal = triangle_normal( vertex_new_position, m_surf.get_position(current_triangle[1]), m_surf.get_position(current_triangle[2]) );
            new_area = triangle_area( vertex_new_position, m_surf.get_position(current_triangle[1]), m_surf.get_position(current_triangle[2]) );
        }
        else if ( current_triangle[1] == source_vertex || current_triangle[1] == destination_vertex )
        {
            new_normal = triangle_normal( m_surf.get_position(current_triangle[0]), vertex_new_position, m_surf.get_position(current_triangle[2]) );
            new_area = triangle_area( m_surf.get_position(current_triangle[0]), vertex_new_position, m_surf.get_position(current_triangle[2]) );
        }
        else
        {
            assert( current_triangle[2] == source_vertex || current_triangle[2] == destination_vertex );
            new_normal = triangle_normal( m_surf.get_position(current_triangle[0]), m_surf.get_position(current_triangle[1]), vertex_new_position );
            new_area = triangle_area( m_surf.get_position(current_triangle[0]), m_surf.get_position(current_triangle[1]), vertex_new_position );
        }
        
        if ( dot( new_normal, old_normal ) < 1e-5 )
        {
            if ( m_surf.m_verbose ) 
            { std::cout << "collapse edge introduces normal inversion" << std::endl; 
            std::cout << "old normal: " << old_normal << std::endl;
            std::cout << "new normal: " << new_normal << std::endl;
            std::cout << "triangleid: " << moving_triangles[i] << std::endl;
            }
            
            g_stats.add_to_int( "EdgeCollapser:collapse_normal_inversion", 1 );
            return true;
        }
        
        if ( new_area < min_triangle_area )
        {
            if ( m_surf.m_verbose ) { std::cout << "collapse edge introduces tiny triangle area" << std::endl; }
            
            g_stats.add_to_int( "EdgeCollapser:collapse_degenerate_triangle", 1 );
            return true;
        }
        
    }
    
    return false;
    
}


// --------------------------------------------------------
///
/// Determine whether collapsing an edge will introduce an unacceptable change in volume.
///
// --------------------------------------------------------

bool EdgeCollapser::collapse_edge_introduces_volume_change( size_t source_vertex,
                                                           size_t edge_index,
                                                           const Vec3d& vertex_new_position )
{
    //
    // If any incident triangle has a tiny area, collapse the edge without regard to volume change
    //
    
    const std::vector<size_t>& inc_tris = m_surf.m_mesh.m_edge_to_triangle_map[edge_index];
    
    for ( size_t i = 0; i < inc_tris.size(); ++i )
    {
        if ( m_surf.get_triangle_area( inc_tris[i] ) < m_surf.m_min_triangle_area )
        {
            return false;
        }
    }
    
    //
    // Check volume change
    //
    
    const std::vector< size_t >& triangles_incident_to_vertex = m_surf.m_mesh.m_vertex_to_triangle_map[source_vertex];
    double volume_change = 0;
    
    for ( size_t i = 0; i < triangles_incident_to_vertex.size(); ++i )
    {
        const Vec3st& inc_tri = m_surf.m_mesh.get_triangle( triangles_incident_to_vertex[i] );
        volume_change += signed_volume( vertex_new_position, m_surf.get_position(inc_tri[0]), m_surf.get_position(inc_tri[1]), m_surf.get_position(inc_tri[2]) );
    }
    
    if ( std::fabs(volume_change) > m_surf.m_max_volume_change )
    {
        if ( m_surf.m_verbose ) { std::cout << "collapse edge introduces volume change"  << std::endl; }
        return true;
    }
    
    return false;
    
}


// ---------------------------------------------------------
///
/// Returns true if the edge collapse would introduce a triangle with a min or max angle outside of the specified min or max.
///
// ---------------------------------------------------------

bool EdgeCollapser::collapse_edge_introduces_bad_angle(size_t source_vertex,
                                                       size_t destination_vertex,
                                                       const Vec3d& vertex_new_position )
{
    
    std::vector<size_t> moving_triangles;
    get_moving_triangles( source_vertex, destination_vertex,  moving_triangles );
    
    int edge_id = (int)m_surf.m_mesh.get_edge_index(source_vertex, destination_vertex);
    
    double min_tri_angle = -1;
    double max_tri_angle = -1;
    for ( size_t i = 0; i < moving_triangles.size(); ++i )
    {
        
        const Vec3st& tri = m_surf.m_mesh.get_triangle( moving_triangles[i] );
        
        double mina = min_triangle_angle(m_surf.get_position(tri[0]), m_surf.get_position(tri[1]), m_surf.get_position(tri[2]));
        double maxa = max_triangle_angle(m_surf.get_position(tri[0]), m_surf.get_position(tri[1]), m_surf.get_position(tri[2]));
        
        //assert(mina >= 0); //This was failing. Is it a NaN or really a negative angle?
        assert(mina == mina);
        assert(maxa == maxa);
        
        if (min_tri_angle < 0 || mina < min_tri_angle)
            min_tri_angle = mina;
        if (max_tri_angle < 0 || maxa > max_tri_angle)
            max_tri_angle = maxa;
    }
    
    assert(max_tri_angle >= min_tri_angle);
    
    min_tri_angle = std::min(min_tri_angle, deg2rad(m_surf.m_min_triangle_angle));
    max_tri_angle = std::max(max_tri_angle, deg2rad(m_surf.m_max_triangle_angle));
    
    for ( size_t i = 0; i < moving_triangles.size(); ++i )
    {
        //skip the tris incident on the collapsing edge.
        const std::vector< size_t >& triangles_incident_to_edge = m_surf.m_mesh.m_edge_to_triangle_map[edge_id];
        bool irrelevant_tri = false;
        for(size_t j = 0; j < triangles_incident_to_edge.size(); ++j) {
            if(triangles_incident_to_edge[j] == moving_triangles[i]) {
                irrelevant_tri = true;
                break;
            }
        }
        
        if(irrelevant_tri)
            continue;
        
        const Vec3st& tri = m_surf.m_mesh.get_triangle( moving_triangles[i] );
        
        
        Vec3d a = m_surf.get_position( tri[0] );
        
        if ( tri[0] == source_vertex || tri[0] == destination_vertex )
        {
            a = vertex_new_position;
        }
        
        Vec3d b = m_surf.get_position( tri[1] );
        
        if ( tri[1] == source_vertex || tri[1] == destination_vertex )
        {
            b = vertex_new_position;
        }
        
        Vec3d c = m_surf.get_position( tri[2] );
        
        if ( tri[2] == source_vertex || tri[2] == destination_vertex )
        {
            c = vertex_new_position;
        }
        
        ///////////////////////////////////////////////////////////////////////
        
        double min_angle = min_triangle_angle( a, b, c );
        if ( min_angle < min_tri_angle )
        {
            return true;
        }
        
        double max_angle = max_triangle_angle( a, b, c );
        if ( max_angle > max_tri_angle )
        {
            return true;
        }
        
    }
    
    return false;
    
}





// --------------------------------------------------------
///
/// Choose the vertex to keep and delete, and the remaining vertices' position.
/// Return false if the edge turns out not to be collapsible
///

// --------------------------------------------------------
bool EdgeCollapser::get_new_vertex_position_dihedral(Vec3d& vertex_new_position, size_t& vertex_to_keep, size_t& vertex_to_delete, const size_t& edge, Vec3c & new_vert_solid_label) {
    
    
    // rank 1, 2, 3 = smooth, ridge, peak
    // if the vertex ranks don't match, keep the higher rank vertex
    
    //TODO: We should eliminate the use of the term "rank", since we are technically no longer using ranks to determine features
    
    int keep_incident_features = m_surf.vertex_feature_edge_count(vertex_to_keep);
    int delete_incident_features = m_surf.vertex_feature_edge_count(vertex_to_delete);
    
    //For now, map these to the existing rank scores
    unsigned int keep_rank = keep_incident_features < 2 ? 1 : (keep_incident_features >= 3 ? 3 : 2);
    unsigned int delete_rank = delete_incident_features < 2 ? 1 : (delete_incident_features >= 3 ? 3 : 2);
    
    bool edge_is_a_feature = m_surf.edge_is_feature(edge);
    
    bool keep_vert_is_boundary = m_surf.m_mesh.m_is_boundary_vertex[vertex_to_keep];
    bool del_vert_is_boundary = m_surf.m_mesh.m_is_boundary_vertex[vertex_to_delete];
    
    // situations where large collapse threshold applies:
    //  1. one of the two vertices has no feature edges
    //  2. both have feature edges, and the edge is a feature, and one of the two vertices has exactly two feature edges
    bool large_threshold = ((keep_rank == 1 || delete_rank == 1) || (((keep_rank == 2 && m_surf.vertex_feature_is_smooth_ridge(vertex_to_keep)) || (delete_rank == 2 && m_surf.vertex_feature_is_smooth_ridge(vertex_to_delete))) && m_surf.edge_is_feature(edge)));
    large_threshold = true; // always use large thresholds for bubbles! because there're no features other than triple junctions that we care about.
    
    large_threshold = large_threshold || m_surf.m_aggressive_mode; //if we are in aggressive mode, use the large threshold
    
    double len = mag(m_surf.get_position(vertex_to_keep) - m_surf.get_position(vertex_to_delete));
    if (!large_threshold && len >= m_t1_pull_apart_distance) {
        if(m_surf.m_verbose)
            std::cout << "!large thresh and len >= t1_dist\n";
        return false;
    }
    
    // boundary vertices have precedence
    if (keep_vert_is_boundary) keep_rank = 4;
    if (del_vert_is_boundary) delete_rank = 4;
    
    // constraint vertices have higher precedence
    Vec3c keep_vert_is_solid =   m_surf.vertex_is_solid_3(vertex_to_keep);
    Vec3c delete_vert_is_solid = m_surf.vertex_is_solid_3(vertex_to_delete);
    bool keep_vert_is_any_solid =   m_surf.vertex_is_any_solid(vertex_to_keep);
    bool delete_vert_is_any_solid = m_surf.vertex_is_any_solid(vertex_to_delete);
    
    bool keep_vert_is_manifold = !m_surf.m_mesh.is_vertex_nonmanifold(vertex_to_keep);
    bool delete_vert_is_manifold = !m_surf.m_mesh.is_vertex_nonmanifold(vertex_to_delete);
    
    new_vert_solid_label = Vec3c(false, false, false);
    if (keep_vert_is_any_solid || delete_vert_is_any_solid)
    {
        assert(m_surf.m_solid_vertices_callback);
        new_vert_solid_label = m_surf.m_solid_vertices_callback->generate_collapsed_solid_label(m_surf, vertex_to_keep, vertex_to_delete, keep_vert_is_solid, delete_vert_is_solid);
    }
    
    if (keep_vert_is_any_solid)   keep_rank = 5;
    if (delete_vert_is_any_solid) delete_rank = 5;
    
    // Handle different cases of constrained, boundary and interior vertices
    if (m_surf.m_allow_vertex_movement_during_collapse && !(keep_vert_is_boundary || del_vert_is_boundary) && !(keep_vert_is_any_solid || delete_vert_is_any_solid))
    {
        //Ranks dominate (i.e. use ranks to decide collapsing first, and if they match then use nonmanifoldness to decide).
        //-> This is particularly important for outward normal flow: it snaps the non-manifold curve back onto the
        //feature curve produced at merge points. (Other scenarios might(?) work better with nonmanifoldness dominating; so I've left
        //the option in the code for now - Christopher Batty.)
        bool ranks_dominate = true;
        if(ranks_dominate) {
            if ( keep_rank > delete_rank ) {
                vertex_new_position = m_surf.get_position(vertex_to_keep);
            }
            else if ( delete_rank > keep_rank ) {
                std::swap(vertex_to_keep, vertex_to_delete);
                vertex_new_position = m_surf.get_position(vertex_to_keep);
            }
            else
            {
                //same ranks, but one is non-manifold; may as well prefer to keep non-manifold points.
                if(!keep_vert_is_manifold && delete_vert_is_manifold) {
                    vertex_new_position = m_surf.get_position(vertex_to_keep);
                }
                else if(keep_vert_is_manifold && !delete_vert_is_manifold) {
                    std::swap(vertex_to_keep, vertex_to_delete);
                    vertex_new_position = m_surf.get_position(vertex_to_keep);
                }
                else {
                    // ranks are equal and manifoldness matches too
                    m_surf.m_subdivision_scheme->generate_new_midpoint( edge, m_surf, vertex_new_position );
                }
            }
        }
        else {
            //Manifoldness dominates
            if(!keep_vert_is_manifold && delete_vert_is_manifold) {
                vertex_new_position = m_surf.get_position(vertex_to_keep);
            }
            else if(!delete_vert_is_manifold && keep_vert_is_manifold) {
                std::swap(vertex_to_keep, vertex_to_delete);
                vertex_new_position = m_surf.get_position(vertex_to_keep);
            }
            else {
                if ( keep_rank > delete_rank ) {
                    vertex_new_position = m_surf.get_position(vertex_to_keep);
                }
                else if ( delete_rank > keep_rank ) {
                    std::swap(vertex_to_keep, vertex_to_delete);
                    vertex_new_position = m_surf.get_position(vertex_to_keep);
                }
                else {
                    // ranks are equal and manifoldness matches too
                    m_surf.m_subdivision_scheme->generate_new_midpoint( edge, m_surf, vertex_new_position );
                }
            }
        }
        
    }
    else if (keep_vert_is_any_solid || delete_vert_is_any_solid)
    {
        assert(m_surf.m_solid_vertices_callback);
        
        if (!keep_vert_is_any_solid)
            std::swap(vertex_to_keep, vertex_to_delete);
        
        Vec3d newpos = (m_surf.get_position(vertex_to_keep) + m_surf.get_position(vertex_to_delete)) / 2;
        if (!m_surf.m_solid_vertices_callback->generate_collapsed_position(m_surf, vertex_to_keep, vertex_to_delete, newpos))
        {
            // the callback decides this edge should not be collapsed
            if (m_surf.m_verbose)
                std::cout << "Constraint callback vetoed collapsing." << std::endl;
            return false;
        }
        
        vertex_new_position = newpos;
    } else if (keep_vert_is_boundary || del_vert_is_boundary)
    {
        if (!keep_vert_is_boundary)
        {
            std::swap(keep_vert_is_boundary, del_vert_is_boundary);
            std::swap(vertex_to_keep, vertex_to_delete);
            std::swap(keep_rank, delete_rank);
        }
        
        vertex_new_position = m_surf.get_position(vertex_to_keep);
    } else  // m_surf.m_allow_vertex_movement_during_collapse == false
    {
        if (keep_rank < delete_rank)
        {
            std::swap(vertex_to_keep, vertex_to_delete);
            std::swap(keep_rank, delete_rank);
        }
        
        vertex_new_position = m_surf.get_position(vertex_to_keep);
    }
    
    return true;
}


// --------------------------------------------------------
///
/// Delete an edge by moving its source vertex to its destination vertex
///
// --------------------------------------------------------

//force_verbose and no_really_collapse is used when we only wish to see if this operation is possible
bool EdgeCollapser::collapse_edge( size_t edge, bool force_verbose, bool no_really_collapse)
{
    size_t vertex_to_keep = m_surf.m_mesh.m_edges[edge][0];
    size_t vertex_to_delete = m_surf.m_mesh.m_edges[edge][1];
    if (m_surf.m_aggressive_mode)
    {
        std::vector<size_t> incident_triangles;
        for (size_t i = 0; i < m_surf.m_mesh.m_vertex_to_triangle_map[vertex_to_keep].size(); i++)
            incident_triangles.push_back(m_surf.m_mesh.m_vertex_to_triangle_map[vertex_to_keep][i]);
        for (size_t i = 0; i < m_surf.m_mesh.m_vertex_to_triangle_map[vertex_to_delete].size(); i++)
            incident_triangles.push_back(m_surf.m_mesh.m_vertex_to_triangle_map[vertex_to_delete][i]);
        bool bad_angle = false;
        for (size_t i = 0; i < incident_triangles.size(); i++)
            if (m_surf.triangle_with_bad_angle(incident_triangles[i]))
                bad_angle = true;
        if (!bad_angle) {
            if (force_verbose) {
                printf("Aggresive mode, bad angle, reject\n");
            }
            // do not collapse in aggressive mode if none of the involved triangles have bad angles.
            return false;
        }
    }
    
    bool keep_vert_is_boundary = m_surf.m_mesh.m_is_boundary_vertex[vertex_to_keep];
    bool del_vert_is_boundary = m_surf.m_mesh.m_is_boundary_vertex[vertex_to_delete];
    bool edge_is_boundary = m_surf.m_mesh.m_is_boundary_edge[edge];
    
    //either we're allowing remeshing of boundary edges, or this edge is not on the boundary.
    assert(m_remesh_boundaries || !edge_is_boundary);
    
    //Vec3d rel_vel = m_surf.get_remesh_velocity(vertex_to_keep) - m_surf.get_remesh_velocity(vertex_to_delete);
    //Vec3d edge_vec = m_surf.get_position(vertex_to_keep) - m_surf.get_position(vertex_to_delete) - rel_vel;
    //double edge_len = mag(edge_vec);
    
    
    bool causes_irregular = collapse_will_produce_irregular_junction(edge);
    
    //disallow irregular configurations from being created if T1's are turned off
    if (!m_surf.m_t1_transition_enabled && causes_irregular) {
        if (force_verbose) {
            printf("t1 not enabled, but cause irregular junction, reject\n");
        }
        return false;
    }
        
    
    //if ((dot(rel_vel, edge_vec) > 0 || edge_len >= m_t1_pull_apart_distance) && causes_irregular && !m_surf.m_aggressive_mode)
    //{
       // if (m_surf.m_verbose)
            //std::cout << "The collapse will produce irregular junction, but the endpoints are moving apart. No need to collapse." << std::endl;
        //return false;
    //}
    
    if ( m_surf.m_verbose ||force_verbose) { std::cout << "Collapsing edge.  Doomed vertex: " << vertex_to_delete << " --- Vertex to keep: " << vertex_to_keep << std::endl; }
    
    // --------------
    
    // If we're disallowing topology changes, don't let an edge collapse form a degenerate tet
    
    if ( false == m_surf.m_allow_non_manifold )
    {
        
        bool would_be_non_manifold = false;
        
        // for each triangle that *would* be created, make sure that there isn't already a triangle with those 3 vertices
        
        for ( size_t i = 0; i < m_surf.m_mesh.m_vertex_to_triangle_map[vertex_to_delete].size(); ++i )
        {
            Vec3st new_triangle = m_surf.m_mesh.get_triangle( m_surf.m_mesh.m_vertex_to_triangle_map[vertex_to_delete][i] );
            if ( new_triangle[0] == vertex_to_delete )   { new_triangle[0] = vertex_to_keep; }
            if ( new_triangle[1] == vertex_to_delete )   { new_triangle[1] = vertex_to_keep; }
            if ( new_triangle[2] == vertex_to_delete )   { new_triangle[2] = vertex_to_keep; }
            
            for ( size_t j = 0; j < m_surf.m_mesh.m_vertex_to_triangle_map[vertex_to_keep].size(); ++j )
            {
                if ( NonDestructiveTriMesh::triangle_has_these_verts( m_surf.m_mesh.get_triangle( m_surf.m_mesh.m_vertex_to_triangle_map[vertex_to_keep][j] ), new_triangle ) )
                {
                    if ( m_surf.m_verbose ||force_verbose) { std::cout << "would_be_non_manifold" << std::endl; }
                    would_be_non_manifold = true;
                    return false;
                }
            }
        }
        
        assert ( !would_be_non_manifold );
        
        // Look for a vertex x which is adjacent to both vertices a and b on the edge e, and which isn't on one of the incident triangles
        /*
                             a
                         . . |
                     .   .   |
                  x.....     e
                     .   .   |
                         . . |
                             b
        */




        const std::vector< size_t >& triangles_incident_to_edge = m_surf.m_mesh.m_edge_to_triangle_map[edge];
        std::vector< size_t > third_vertices;
        
        for ( size_t i = 0; i < triangles_incident_to_edge.size(); ++i )
        {
            const Vec3st& inc_triangle = m_surf.m_mesh.get_triangle( triangles_incident_to_edge[i] );
            size_t opposite = m_surf.m_mesh.get_third_vertex( edge, inc_triangle );
            third_vertices.push_back( opposite );
        }
        
        std::vector<size_t> adj_vertices0, adj_vertices1;
        m_surf.m_mesh.get_adjacent_vertices( vertex_to_delete, adj_vertices0 );
        m_surf.m_mesh.get_adjacent_vertices( vertex_to_keep, adj_vertices1 );
        
        for ( size_t i = 0; i < adj_vertices0.size(); ++i )
        {
            for ( size_t j = 0; j < adj_vertices1.size(); ++j )
            {
                if ( adj_vertices0[i] == adj_vertices1[j] )
                {
                    bool is_on_inc_triangle = false;
                    for ( size_t k = 0; k < third_vertices.size(); ++k )
                    {
                        if ( adj_vertices0[i] == third_vertices[k] )
                        {
                            is_on_inc_triangle = true;
                            break;
                        }
                    }
                    
                    if ( !is_on_inc_triangle )
                    {
                        // found a vertex adjacent to both edge vertices, which doesn't lie on the incident triangles
                        
                        //
                        if ( m_surf.m_verbose || force_verbose)
                        {
                            std::cout << " --- Edge Collapser: found a vertex adjacent to both edge vertices, which doesn't lie on the incident triangles " << std::endl;
                            std::cout << " --- Adjacent vertex: " << adj_vertices0[i] << ", incident triangles: ";
                        }
                        
                        return false;
                    }
                    
                }
            }
        }
        
        
    }
    
    
    // --------------
    
    {
        const std::vector< size_t >& r_triangles_incident_to_edge = m_surf.m_mesh.m_edge_to_triangle_map[edge];
        
        // Do not collapse edge on a degenerate tet or degenerate triangle
        for ( size_t i=0; i < r_triangles_incident_to_edge.size(); ++i )
        {
            const Vec3st& triangle_i = m_surf.m_mesh.get_triangle( r_triangles_incident_to_edge[i] );
            
            if ( triangle_i[0] == triangle_i[1] || triangle_i[1] == triangle_i[2] || triangle_i[2] == triangle_i[0] )
            {
                if ( m_surf.m_verbose || force_verbose) { std::cout << "duplicate vertices on triangle" << std::endl; }
                return false;
            }
            
            for ( size_t j=i+1; j < r_triangles_incident_to_edge.size(); ++j )
            {
                const Vec3st& triangle_j = m_surf.m_mesh.get_triangle( r_triangles_incident_to_edge[j] );
                
                if ( NonDestructiveTriMesh::triangle_has_these_verts( triangle_i, triangle_j ) )
                {
                    if ( m_surf.m_verbose || force_verbose) { std::cout << "two triangles share vertices" << std::endl; }
                    g_stats.add_to_int( "EdgeCollapser:collapse_degen_tet", 1 );
                    return false;
                }
            }
        }
    }
    
    
    // --------------
    // decide on new vertex position
    
    //Choose the vertex to keep and its new position.
    Vec3d vertex_new_position;
    Vec3c new_vert_solid_label;
    bool can_collapse = get_new_vertex_position_dihedral(vertex_new_position, vertex_to_keep, vertex_to_delete, edge, new_vert_solid_label);
    if (!can_collapse) {
        if (force_verbose) {
            printf("cannot get new vertex position_dihedral, reject\n");
        }
        return false;
    }
    
    
    if ( m_surf.m_verbose ) {
        std::cout << "Collapsing edge.  Doomed vertex: " << vertex_to_delete << " --- Vertex to keep: " << vertex_to_keep << std::endl; 
        std::cout << " vtx_to_delete pos: " << m_surf.get_position(vertex_to_delete) << std::endl;
        std::cout << " vtx_to_keep pos: " << m_surf.get_position(vertex_to_keep) << std::endl;
        std::cout << "middle point: " << 0.5 * (m_surf.get_position(vertex_to_delete) + m_surf.get_position(vertex_to_keep)) << std::endl;
        std::cout << "new pos: " << vertex_new_position << std::endl;
    }
    
    // --------------
    
    // Check vertex pseudo motion for collisions and volume change
    
    if ( m_surf.get_position(m_surf.m_mesh.m_edges[edge][1]) != m_surf.get_position(m_surf.m_mesh.m_edges[edge][0]) )
        //if ( mag ( m_surf.get_position(m_surf.m_mesh.m_edges[edge][1]) - m_surf.get_position(m_surf.m_mesh.m_edges[edge][0]) ) > 0 )
    {
        
        // Change source vertex predicted position to superimpose onto destination vertex
        m_surf.set_newposition( vertex_to_keep, vertex_new_position );
        m_surf.set_newposition( vertex_to_delete, vertex_new_position );
        
        bool volume_change = collapse_edge_introduces_volume_change( vertex_to_delete, edge, vertex_new_position );
        volume_change = false;  // ignore volume changes for bubbles!
        
        if ( volume_change && !m_surf.m_aggressive_mode)
        {
            // Restore saved positions which were changed by the function we just called.
            m_surf.set_newposition( vertex_to_keep, m_surf.get_position(vertex_to_keep) );
            m_surf.set_newposition( vertex_to_delete, m_surf.get_position(vertex_to_delete) );
            
            g_stats.add_to_int( "EdgeCollapser:collapse_volume_change", 1 );
            
            if ( m_surf.m_verbose || force_verbose) { std::cout << "collapse_volume_change" << std::endl; }
            return false;
        }
        
        bool normal_inversion = collapse_edge_introduces_normal_inversion(  vertex_to_delete, vertex_to_keep, edge, vertex_new_position );
        
        if ( normal_inversion && !m_surf.m_aggressive_mode)//&& (edge_len >= m_t1_pull_apart_distance) )
        {
            // Restore saved positions which were changed by the function we just called.
            m_surf.set_newposition( vertex_to_keep, m_surf.get_position(vertex_to_keep) );
            m_surf.set_newposition( vertex_to_delete, m_surf.get_position(vertex_to_delete) );
            
            if ( m_surf.m_verbose || force_verbose) { std::cout << "normal_inversion" << std::endl; }
            return false;
        }
        
        bool bad_angle = collapse_edge_introduces_bad_angle( vertex_to_delete, vertex_to_keep, vertex_new_position);
        
        if ( bad_angle && !m_surf.m_aggressive_mode )//&& edge_len >= m_t1_pull_apart_distance )
        {
            // Restore saved positions which were changed by the function we just called.
            m_surf.set_newposition( vertex_to_keep, m_surf.get_position(vertex_to_keep) );
            m_surf.set_newposition( vertex_to_delete, m_surf.get_position(vertex_to_delete) );
            
            if ( m_surf.m_verbose || force_verbose) { std::cout << "bad_angle" << std::endl; }
            
            g_stats.add_to_int( "EdgeCollapser:collapse_bad_angle", 1 );
            return false;
            
        }
        
        bool collision = false;
        
        if ( m_surf.m_collision_safety )
        {
            collision = collapse_edge_pseudo_motion_introduces_collision( vertex_to_delete, vertex_to_keep, edge, vertex_new_position );
        }
        if ( collision )
        {
            if ( m_surf.m_verbose || force_verbose) { std::cout << "collision" << std::endl; }
            g_stats.add_to_int( "EdgeCollapser:collapse_collisions", 1 );
        }
        
        // Restore saved positions which were changed by the function we just called.
        m_surf.set_newposition( vertex_to_keep, m_surf.get_position(vertex_to_keep) );
        m_surf.set_newposition( vertex_to_delete, m_surf.get_position(vertex_to_delete) );
        
        if ( collision )
        {
            // edge collapse would introduce collision or change volume too much or invert triangle normals
            return false;
        }
    }
    
    // --------------
    // all clear, now perform the collapse
    if (no_really_collapse) {
        printf("end before collapse happen\n");
        return false;
    }
    void * data = NULL;
    if (m_surf.m_mesheventcallback)
        m_surf.m_mesheventcallback->pre_collapse(m_surf, edge, &data);
    
    // start building history data
    MeshUpdateEvent collapse(MeshUpdateEvent::EDGE_COLLAPSE);
    collapse.m_vert_position = vertex_new_position;
    collapse.m_v0 = vertex_to_keep;
    collapse.m_v1 = vertex_to_delete;
    
    // move the vertex we decided to keep
    
    m_surf.set_position( vertex_to_keep, vertex_new_position );
    m_surf.set_newposition( vertex_to_keep, vertex_new_position );
    
    // update the vertex constraint label
    for (int i = 0; i < 3; i++)
    {
        if (new_vert_solid_label[i])
            m_surf.m_masses[vertex_to_keep][i] = DynamicSurface::solid_mass();
        else
            m_surf.m_masses[vertex_to_keep][i] = 1;
    }
    
    ///////////////////////////////////////////////////////////////////////
    
    // Copy this vector, don't take a reference, as deleting will change the original
    std::vector< size_t > triangles_incident_to_edge = m_surf.m_mesh.m_edge_to_triangle_map[edge];
    
    // Delete triangles incident on the edge
    
    for ( size_t i=0; i < triangles_incident_to_edge.size(); ++i )
    {
        if ( m_surf.m_verbose )
        {
            std::cout << "removing edge-incident triangle: " << m_surf.m_mesh.get_triangle( triangles_incident_to_edge[i] ) << std::endl;
        }
        
        m_surf.remove_triangle( triangles_incident_to_edge[i] );
        collapse.m_deleted_tris.push_back(triangles_incident_to_edge[i] );
    }
    
    // Find anything pointing to the doomed vertex and change it
    
    // copy the list of triangles, don't take a reference to it
    std::vector< size_t > triangles_incident_to_vertex = m_surf.m_mesh.m_vertex_to_triangle_map[vertex_to_delete];
    
    for ( size_t i=0; i < triangles_incident_to_vertex.size(); ++i )
    {
        for(size_t local_ind = 0; local_ind < triangles_incident_to_edge.size(); ++local_ind) {
            assert( triangles_incident_to_vertex[i] != triangles_incident_to_edge[local_ind] );
        }
        
        Vec3st new_triangle = m_surf.m_mesh.get_triangle( triangles_incident_to_vertex[i] );
        
        if ( new_triangle[0] == vertex_to_delete )   { new_triangle[0] = vertex_to_keep; }
        if ( new_triangle[1] == vertex_to_delete )   { new_triangle[1] = vertex_to_keep; }
        if ( new_triangle[2] == vertex_to_delete )   { new_triangle[2] = vertex_to_keep; }
        
        if ( m_surf.m_verbose ) { std::cout << "adding updated triangle: " << new_triangle << std::endl; }
        
        // the old label carries over to the new triangle.
        // no need to test for orientation because the new triangle
        // generation code above does not change orientation.
        //
        Vec2i label = m_surf.m_mesh.get_triangle_label(triangles_incident_to_vertex[i]);
        
        size_t new_triangle_index = m_surf.add_triangle( new_triangle, label );
        collapse.m_created_tris.push_back( new_triangle_index );
        collapse.m_created_tri_data.push_back(new_triangle);
        collapse.m_created_tri_labels.push_back(label);
        
        ////////////////////////////////////////////////////////////
        
        m_surf.m_dirty_triangles.push_back( new_triangle_index );
    }
    
    for ( size_t i=0; i < triangles_incident_to_vertex.size(); ++i )
    {
        if ( m_surf.m_verbose )
        {
            std::cout << "removing vertex-incident triangle: " << m_surf.m_mesh.get_triangle( triangles_incident_to_vertex[i] ) << std::endl;
        }
        
        m_surf.remove_triangle( triangles_incident_to_vertex[i] );
        collapse.m_deleted_tris.push_back(triangles_incident_to_vertex[i]);
    }
    
    // Delete vertex
    assert( m_surf.m_mesh.m_vertex_to_triangle_map[vertex_to_delete].empty() );
    m_surf.remove_vertex( vertex_to_delete );
    collapse.m_deleted_verts.push_back( vertex_to_delete );
    
    m_surf.m_mesh.update_is_boundary_vertex( vertex_to_keep );
    
    // update the remeshing velocities
    m_surf.pm_velocities[vertex_to_keep] = (m_surf.pm_velocities[vertex_to_keep] + m_surf.pm_velocities[vertex_to_delete]) / 2;

    // update the target edge length
    double new_target_edge_length = m_surf.compute_vertex_target_edge_length(vertex_to_keep);
    std::vector<size_t> new_onering;
    m_surf.m_mesh.get_adjacent_vertices(vertex_to_keep, new_onering);
    for (size_t i = 0; i < new_onering.size(); i++)
        if (new_target_edge_length > m_surf.vertex_target_edge_length(new_onering[i]) * m_surf.m_max_adjacent_target_edge_length_ratio)
            new_target_edge_length = m_surf.vertex_target_edge_length(new_onering[i]) * m_surf.m_max_adjacent_target_edge_length_ratio;
    m_surf.m_target_edge_lengths[vertex_to_keep] = new_target_edge_length;
    
    // Store the history
    m_surf.m_mesh_change_history.push_back(collapse);

    // remove degeneracies
    m_surf.trim_degeneracies( m_surf.m_dirty_triangles );

    if (m_surf.m_mesheventcallback)
        m_surf.m_mesheventcallback->post_collapse(m_surf, edge, vertex_to_keep, data);
    return true;
}

// --------------------------------------------------------
///
/// Determine if the edge should be allowed to collapse
///
// --------------------------------------------------------

bool EdgeCollapser::edge_is_collapsible( size_t edge_index, double& current_length )
{
    static double cos_cutoff = cos(deg2rad(m_surf.m_min_triangle_angle));

    // skip deleted and solid edges
    if ( m_surf.m_mesh.edge_is_deleted(edge_index) ) { return false; }
    //  if ( m_surf.edge_is_any_solid(edge_index) ) { return false; }
    
    //skip boundary edges if we're not remeshing those
    if(!m_remesh_boundaries && m_surf.m_mesh.m_is_boundary_edge[edge_index]) { return false; }
    
    //try to collapse based on small angles
    size_t vertex_a = m_surf.m_mesh.m_edges[edge_index][0];
    size_t vertex_b = m_surf.m_mesh.m_edges[edge_index][1];
    Vec3d a = m_surf.get_position(vertex_a);
    Vec3d b = m_surf.get_position(vertex_b);
    for(size_t i = 0; i < m_surf.m_mesh.m_edge_to_triangle_map[edge_index].size(); ++i) {
        size_t tri_id = m_surf.m_mesh.m_edge_to_triangle_map[edge_index][i];
        Vec3st tri = m_surf.m_mesh.m_tris[tri_id];
        size_t vertex_c = m_surf.m_mesh.get_third_vertex(edge_index, tri_id);
        Vec3d c = m_surf.get_position(vertex_c);
        
        //The current angle is less than our threshold, so try to collapse
        //double cur_dot = dot(normalized(b - c), normalized(a - c));
        double cur_dot = corner_normalized_dot(a, b, c);
        /*double cur_dot = ((b[0] - c[0]) * (a[0] - c[0]) + (b[1] - c[1]) * (a[1] - c[1]) + (b[2] - c[2]) * (a[2] - c[2])) ;
         / sqrt(sqr(b[0] - c[0]) + sqr(b[1] - c[1]) + sqr(b[2] - c[2]))
         / sqrt(sqr(a[0] - c[0]) + sqr(a[1] - c[1]) + sqr(a[2] - c[2]));*/
        if (cur_dot > cos_cutoff)
            return true;
        //slow direct way -- silly micro-optimized into the above
        //double cur_angle = acos(cur_dot);
        //if(rad2deg(cur_angle) < m_surf.m_min_triangle_angle)
        //    return true;
    }
    
    current_length = m_surf.get_edge_length(edge_index);
    if ( m_use_curvature )
    {
        
        //collapse if we're below the lower limit
        if(current_length < m_surf.edge_min_edge_length(edge_index))
            return true;
        
        //don't collapse if we're near the upper limit, since it can produce edges above the limit
        if(current_length > m_surf.edge_max_edge_length(edge_index)*0.5)
            return false;
        
        //check all incident edges to see if any of them are super short, and if so, split this guy accordingly.
        //this enforces slow grading of the mesh.
        double min_nbr_len = current_length;
        size_t vertex_a = m_surf.m_mesh.m_edges[edge_index][0];
        size_t vertex_b = m_surf.m_mesh.m_edges[edge_index][1];
        for(size_t edge_id = 0; edge_id < m_surf.m_mesh.m_vertex_to_edge_map[vertex_a].size(); ++edge_id) {
            min_nbr_len = min(min_nbr_len, m_surf.get_edge_length(m_surf.m_mesh.m_vertex_to_edge_map[vertex_a][edge_id]));
        }
        for(size_t edge_id = 0; edge_id < m_surf.m_mesh.m_vertex_to_edge_map[vertex_b].size(); ++edge_id) {
            min_nbr_len = min(min_nbr_len, m_surf.get_edge_length(m_surf.m_mesh.m_vertex_to_edge_map[vertex_b][edge_id]));
        }
        
        if(current_length < 3*min_nbr_len )
            return false;
        
        double curvature_value = get_edge_curvature( m_surf, m_surf.m_mesh.m_edges[edge_index][0], m_surf.m_mesh.m_edges[edge_index][1] );
        
        //Assume we want to discretize any circle with at least ten segments.
        //Then give the curvature (i.e. inverse radius) the target edge length
        //here should be computed as... 
        int circlesegs = 32;
        double curvature_min_length = 2*M_PI / (double)circlesegs / max(curvature_value, 1e-8);
        
        //collapse if curvature dictates we should. 
        return current_length < curvature_min_length;
        
    }
    else
    {
        /*if (current_length < m_surf.edge_min_edge_length(edge_index)) {
            std::cout << "EDGE" << edge_index << "l=" << current_length << "< tglen" << m_surf.edge_min_edge_length(edge_index) << std::endl;
        }*/
        return current_length < m_surf.edge_min_edge_length(edge_index);
    }
    
    
}


// --------------------------------------------------------
///
/// Collapse all short edges
///
// --------------------------------------------------------

bool EdgeCollapser::collapse_pass()
{
    
    if ( m_surf.m_verbose )
    {
        std::cout << "\n\n\n---------------------- EdgeCollapser: collapsing ----------------------" << std::endl;
//        std::cout << "m_min_edge_length: " << m_min_edge_length;
        std::cout << ", m_use_curvature: " << m_use_curvature;
        std::cout << ", m_min_curvature_multiplier: " << m_min_curvature_multiplier << std::endl;
        std::cout << "m_surf.m_collision_safety: " << m_surf.m_collision_safety << std::endl;
        std::cout << "m_min_to_target_ratio: " << m_surf.m_min_to_target_ratio << std::endl;
    }
    
    bool collapse_occurred = false;
    
    assert( m_surf.m_dirty_triangles.size() == 0 );
    
    std::vector<SortableEdge> sortable_edges_to_try;
    
    
    //
    // get set of edges to collapse
    //
    
    for( size_t i = 0; i < m_surf.m_mesh.m_edges.size(); i++ )
    {    
        double current_length;
        if(edge_is_collapsible(i, current_length)) 
            sortable_edges_to_try.push_back( SortableEdge( i, current_length ) );
    }
    
    //
    // sort in ascending order by length (collapse shortest edges first)
    //
    
    std::sort( sortable_edges_to_try.begin(), sortable_edges_to_try.end() );
    
    if ( m_surf.m_verbose )
    {
        std::cout << sortable_edges_to_try.size() << " candidate edges sorted" << std::endl;
        std::cout << "total edges: " << m_surf.m_mesh.m_edges.size() << std::endl;
    }
    
    //
    // attempt to collapse each edge in the sorted list
    //
    
    for ( size_t si = 0; si < sortable_edges_to_try.size(); ++si )
    {
        size_t e = sortable_edges_to_try[si].m_edge_index;
        
        assert( e < m_surf.m_mesh.m_edges.size() );
        
        double dummy;
        if(edge_is_collapsible(e, dummy)) {
            bool result = collapse_edge( e );
            collapse_occurred |= result;
        }
    }
    
    return collapse_occurred;
    
}

bool EdgeCollapser::collapse_will_produce_irregular_junction(size_t edge)
{
    NonDestructiveTriMesh& mesh = m_surf.m_mesh;

    size_t a = mesh.m_edges[edge][0];
    size_t b = mesh.m_edges[edge][1];


    std::vector<int> regions; //list of regions that are present


    //collect all incident regions
    for (size_t i = 0; i < mesh.m_vertex_to_triangle_map[a].size(); i++)
    {
        const Vec2i& l = mesh.get_triangle_label(mesh.m_vertex_to_triangle_map[a][i]);
        if (l[0] >= 0)
        {
            regions.push_back(l[0]);
        }
        if (l[1] >= 0)
        {
            regions.push_back(l[1]);
        }
    }

    for (size_t i = 0; i < mesh.m_vertex_to_triangle_map[b].size(); i++)
    {
        const Vec2i& l = mesh.get_triangle_label(mesh.m_vertex_to_triangle_map[b][i]);
        if (l[0] >= 0)
        {
            regions.push_back(l[0]);
        }
        if (l[1] >= 0)
        {
            regions.push_back(l[1]);
        }
    }

    //remove duplicate regions
    sort(regions.begin(), regions.end());
    regions.erase(unique(regions.begin(), regions.end()), regions.end());

    //build a map from the region # to its position in the list above
    int max_val = *(std::max_element(regions.begin(), regions.end()));
    std::vector<int> reverse_map(max_val + 1, -1);
    for (int i = 0; i < regions.size(); ++i)
        reverse_map[regions[i]] = i;

    //build a (local!) adjacency graph indicating which pairs of materials touch (by a face)
    int nr = (int)regions.size();
    bool* regiongraph = new bool[nr * nr];
    memset(regiongraph, 0, sizeof(bool) * nr * nr);
    for (size_t i = 0; i < mesh.m_vertex_to_triangle_map[a].size(); i++)
    {
        const Vec2i& l = mesh.get_triangle_label(mesh.m_vertex_to_triangle_map[a][i]);
        if (l[0] >= 0 && l[1] >= 0) {
            Vec2i local_label(reverse_map[l[0]], reverse_map[l[1]]);
            regiongraph[local_label[0] * nr + local_label[1]] = regiongraph[local_label[1] * nr + local_label[0]] = true;
        }
    }
    for (size_t i = 0; i < mesh.m_vertex_to_triangle_map[b].size(); i++)
    {
        const Vec2i& l = mesh.get_triangle_label(mesh.m_vertex_to_triangle_map[b][i]);
        if (l[0] >= 0 && l[1] >= 0) {
            Vec2i local_label(reverse_map[l[0]], reverse_map[l[1]]);
            regiongraph[local_label[0] * nr + local_label[1]] = regiongraph[local_label[1] * nr + local_label[0]] = true;
        }
    }

    bool irregular = false;
    for (int i = 0; i < nr && !irregular; i++)
        for (int j = i + 1; j < nr; j++)
        {
            if (!regiongraph[i * nr + j]) {
                irregular = true;
                break;
            }
        }

    delete[]regiongraph;

    return irregular;
}
    
}

