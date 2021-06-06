// ---------------------------------------------------------
//
//  edgesplitter.cpp
//  Tyson Brochu 2011
//  Christopher Batty, Fang Da 2014
//
//  Functions supporting the "edge split" operation: subdividing an edge into two shorter edges.
//
// ---------------------------------------------------------

#include <edgesplitter.h>
#include <broadphase.h>
#include <collisionqueries.h>
#include <runstats.h>
#include <subdivisionscheme.h>
#include <surftrack.h>
#include <trianglequality.h>
#include <typeinfo>

// ---------------------------------------------------------
//  Extern globals
// ---------------------------------------------------------

namespace LosTopos {
    
extern RunStats g_stats;

// ---------------------------------------------------------
// Member function definitions
// ---------------------------------------------------------

// ---------------------------------------------------------
///
/// Constructor.  Active SurfTrack object must be supplied.
///
// ---------------------------------------------------------

EdgeSplitter::EdgeSplitter( SurfTrack& surf, bool use_curvature, bool remesh_boundaries, double max_curvature_multiplier ) :
//m_min_edge_length( UNINITIALIZED_DOUBLE ),
//m_max_edge_length( UNINITIALIZED_DOUBLE ),
m_use_curvature( use_curvature ),
m_max_curvature_multiplier( max_curvature_multiplier ),
m_remesh_boundaries( remesh_boundaries),
m_surf( surf )
{}


// --------------------------------------------------------
///
/// Check collisions between the edge [neighbour, new] and the given edge
///
// --------------------------------------------------------

bool EdgeSplitter::split_edge_edge_collision( size_t neighbour_index,
                                             const Vec3d& new_vertex_position,
                                             const Vec3d& new_vertex_smooth_position,
                                             const Vec2st& edge )
{
    
    size_t edge_vertex_0 = edge[0];
    size_t edge_vertex_1 = edge[1];
    size_t dummy_index = m_surf.get_num_vertices();
    
    if ( neighbour_index == edge_vertex_0 || neighbour_index == edge_vertex_1 )  { return false; }
    
    const std::vector<Vec3d>& x = m_surf.get_positions();
    
    double t_zero_distance;
    check_edge_edge_proximity( new_vertex_position,
                              x[ neighbour_index ],
                              x[ edge_vertex_0 ],
                              x[ edge_vertex_1 ],
                              t_zero_distance );
    
    if ( t_zero_distance < m_surf.m_improve_collision_epsilon )
    {
        return true;
    }
    
    if ( edge_vertex_1 < edge_vertex_0 ) { swap( edge_vertex_0, edge_vertex_1 ); }
    
    if ( segment_segment_collision(x[ neighbour_index ], x[ neighbour_index ], neighbour_index,
                                   new_vertex_position, new_vertex_smooth_position, dummy_index,
                                   x[ edge_vertex_0 ], x[ edge_vertex_0 ], edge_vertex_0,
                                   x[ edge_vertex_1 ], x[ edge_vertex_1 ], edge_vertex_1 ) )
        
    {
        return true;
    }
    
    return false;
    
}


// ---------------------------------------------------------
///
/// Determine if the new vertex introduced by the edge split has a collision along its pseudo-trajectory.
///
// ---------------------------------------------------------

bool EdgeSplitter::split_triangle_vertex_collision( const Vec3st& triangle_indices,
                                                   const Vec3d& new_vertex_position,
                                                   const Vec3d& new_vertex_smooth_position,
                                                   size_t overlapping_vert_index,
                                                   const Vec3d& vert )
{
    
    if ( overlapping_vert_index == triangle_indices[0] || overlapping_vert_index == triangle_indices[1] || overlapping_vert_index == triangle_indices[2] )
    {
        return false;
    }
    
    Vec3st sorted_triangle = sort_triangle( triangle_indices );
    
    Vec3d tri_positions[3];
    Vec3d tri_smooth_positions[3];
    
    for ( unsigned int i = 0; i < 3; ++i )
    {
        if ( sorted_triangle[i] == m_surf.get_num_vertices() )
        {
            tri_positions[i] = new_vertex_position;
            tri_smooth_positions[i] = new_vertex_smooth_position;
        }
        else
        {
            tri_positions[i] = m_surf.get_position( sorted_triangle[i] );
            tri_smooth_positions[i] = m_surf.get_position( sorted_triangle[i] );
        }
    }
    
    
    // check distance at time t=0
    double t_zero_distance;
    check_point_triangle_proximity( vert, tri_positions[0], tri_positions[1], tri_positions[2], t_zero_distance );
    
    
    if ( t_zero_distance < m_surf.m_improve_collision_epsilon )
    {
        return true;
    }
    
    
    // now check continuous collision
    
    if ( point_triangle_collision( vert, vert, overlapping_vert_index,
                                  tri_positions[0], tri_smooth_positions[0], sorted_triangle[0],
                                  tri_positions[1], tri_smooth_positions[1], sorted_triangle[1],
                                  tri_positions[2], tri_smooth_positions[2], sorted_triangle[2] ) )
    {
        return true;
    }
    
    return false;
    
    
}



// ---------------------------------------------------------
///
/// Determine if the pseudo-trajectory of the new vertex has a collision with the existing mesh.
///
// ---------------------------------------------------------
bool EdgeSplitter::split_edge_pseudo_motion_introduces_intersection( const Vec3d& new_vertex_position,
                                                                    const Vec3d& new_vertex_smooth_position,
                                                                    size_t edge,
                                                                    size_t vertex_a,
                                                                    size_t vertex_b,
                                                                    const std::vector<size_t>& tris,
                                                                    const std::vector<size_t>& verts,
                                                                    const std::vector<size_t> & ignore_vertices)
{
    
    NonDestructiveTriMesh& m_mesh = m_surf.m_mesh;
    
    if ( !m_surf.m_collision_safety)
    {
        return false;
    }
    
    //
    // new point vs all triangles
    //
    
    {
        
        Vec3d aabb_low, aabb_high;
        minmax( new_vertex_position, new_vertex_smooth_position, aabb_low, aabb_high );
        
        aabb_low -= m_surf.m_aabb_padding * Vec3d(1,1,1);
        aabb_high += m_surf.m_aabb_padding * Vec3d(1,1,1);
        
        std::vector<size_t> overlapping_triangles;
        m_surf.m_broad_phase->get_potential_triangle_collisions( aabb_low, aabb_high, true, true, overlapping_triangles );
        
        for ( size_t i = 0; i < overlapping_triangles.size(); ++i )
        {
            
            // Exclude all incident triangles from the check
            bool overlap = false;
            for(size_t j = 0; j < tris.size(); ++j) {
                if(overlapping_triangles[i] == tris[j]) {
                    overlap = true;
                    break;
                }
            }
            if( overlap )
                continue;
            
            bool ignore = false;
            for (size_t j = 0; j < ignore_vertices.size(); j++) {
                if (m_mesh.get_triangle(overlapping_triangles[i])[0] == ignore_vertices[j] || m_mesh.get_triangle(overlapping_triangles[i])[1] == ignore_vertices[j] || m_mesh.get_triangle(overlapping_triangles[i])[2] == ignore_vertices[j]) {
                    ignore = true;
                    break;
                }
            }
            if (ignore)
                continue;
            
            size_t triangle_vertex_0 = m_mesh.get_triangle( overlapping_triangles[i] )[0];
            size_t triangle_vertex_1 = m_mesh.get_triangle( overlapping_triangles[i] )[1];
            size_t triangle_vertex_2 = m_mesh.get_triangle( overlapping_triangles[i] )[2];
            
            double t_zero_distance;
            
            check_point_triangle_proximity( new_vertex_position,
                                           m_surf.get_position( triangle_vertex_0 ),
                                           m_surf.get_position( triangle_vertex_1 ),
                                           m_surf.get_position( triangle_vertex_2 ),
                                           t_zero_distance );
            
            size_t dummy_index = m_surf.get_num_vertices();
            
            if ( t_zero_distance < m_surf.m_improve_collision_epsilon )
            {
                return true;
            }
            
            Vec3st sorted_triangle = sort_triangle( Vec3st( triangle_vertex_0, triangle_vertex_1, triangle_vertex_2 ) );
            
            
            if ( point_triangle_collision(  new_vertex_position, new_vertex_smooth_position, dummy_index,
                                          m_surf.get_position( sorted_triangle[0] ), m_surf.get_position( sorted_triangle[0] ), sorted_triangle[0],
                                          m_surf.get_position( sorted_triangle[1] ), m_surf.get_position( sorted_triangle[1] ), sorted_triangle[1],
                                          m_surf.get_position( sorted_triangle[2] ), m_surf.get_position( sorted_triangle[2] ), sorted_triangle[2] ) )
                
            {
                return true;
            }
        }
        
    }
    
    //
    // new edges vs all edges
    //
    
    {
        
        Vec3d edge_aabb_low, edge_aabb_high;
        
        // do one big query into the broad phase for all new edges
        minmax( new_vertex_position, new_vertex_smooth_position,
               m_surf.get_position( vertex_a ), m_surf.get_position( vertex_b ),
               edge_aabb_low, edge_aabb_high );
        for(size_t i = 0; i < verts.size(); ++i)
            update_minmax(m_surf.get_position(verts[i]), edge_aabb_low, edge_aabb_high);
        
        edge_aabb_low -= m_surf.m_aabb_padding * Vec3d(1,1,1);
        edge_aabb_high += m_surf.m_aabb_padding * Vec3d(1,1,1);
        
        std::vector<size_t> overlapping_edges;
        m_surf.m_broad_phase->get_potential_edge_collisions( edge_aabb_low, edge_aabb_high, true, true, overlapping_edges );
        
        std::vector<size_t> vertex_neighbourhood;
        vertex_neighbourhood.push_back(vertex_a); vertex_neighbourhood.push_back(vertex_b);
        vertex_neighbourhood.insert(vertex_neighbourhood.end(), verts.begin(), verts.end());
        
        for ( size_t i = 0; i < overlapping_edges.size(); ++i )
        {
            
            if ( overlapping_edges[i] == edge ) { continue; }
            if ( m_mesh.m_edges[ overlapping_edges[i] ][0] == m_mesh.m_edges[ overlapping_edges[i] ][1] ) { continue; }
            
            bool ignore = false;
            for (size_t j = 0; j < ignore_vertices.size(); j++) {
                if (m_mesh.m_edges[overlapping_edges[i]][0] == ignore_vertices[j] || m_mesh.m_edges[overlapping_edges[i]][1] == ignore_vertices[j]) {
                    ignore = true;
                    break;
                }
            }
            if (ignore)
                continue;
            
            for ( size_t v = 0; v < vertex_neighbourhood.size(); ++v )
            {
                bool collision = split_edge_edge_collision( vertex_neighbourhood[v],
                                                           new_vertex_position,
                                                           new_vertex_smooth_position,
                                                           m_mesh.m_edges[overlapping_edges[i]]);
                
                if ( collision ) { return true; }
            }
        }
    }
    
    //
    // new triangles vs all points
    //
    
    {
        Vec3d triangle_aabb_low, triangle_aabb_high;
        
        // do one big query into the broad phase for all new triangles
        minmax( new_vertex_position, new_vertex_smooth_position,
               m_surf.get_position( vertex_a ), m_surf.get_position( vertex_b ),
               triangle_aabb_low, triangle_aabb_high );
        for(size_t i = 0; i < verts.size(); ++i)
            update_minmax(m_surf.get_position(verts[i]), triangle_aabb_low, triangle_aabb_high);
        
        triangle_aabb_low -= m_surf.m_aabb_padding * Vec3d(1,1,1);
        triangle_aabb_high += m_surf.m_aabb_padding * Vec3d(1,1,1);
        
        std::vector<size_t> overlapping_vertices;
        m_surf.m_broad_phase->get_potential_vertex_collisions( triangle_aabb_low, triangle_aabb_high, true, true, overlapping_vertices );
        
        size_t dummy_e = m_surf.get_num_vertices();
        
        std::vector< Vec3st > triangle_indices;
        
        for( size_t i = 0; i < verts.size(); ++i) {
            triangle_indices.push_back( Vec3st( vertex_a, dummy_e, verts[i] ) );    // triangle aec
            triangle_indices.push_back( Vec3st( verts[i], dummy_e, vertex_b ) );    // triangle ceb
        }
        
        for ( size_t i = 0; i < overlapping_vertices.size(); ++i )
        {
            if ( m_mesh.m_vertex_to_triangle_map[overlapping_vertices[i]].empty() )
            {
                continue;
            }
            
            size_t overlapping_vert_index = overlapping_vertices[i];
            const Vec3d& vert = m_surf.get_position(overlapping_vert_index);
            
            bool ignore = false;
            for (size_t j = 0; j < ignore_vertices.size(); j++) {
                if (overlapping_vert_index == ignore_vertices[j]) {
                    ignore = true;
                    break;
                }
            }
            if (ignore)
                continue;
            
            for ( size_t j = 0; j < triangle_indices.size(); ++j )
            {
                bool collision = split_triangle_vertex_collision( triangle_indices[j],
                                                                 new_vertex_position,
                                                                 new_vertex_smooth_position,
                                                                 overlapping_vert_index,
                                                                 vert);
                
                if ( collision )
                {
                    return true;
                }
                
            }
        }
        
    }
    
    return false;
    
}


// --------------------------------------------------------
///
/// Split an edge, using subdivision_scheme to determine the new vertex location, if safe to do so.
///
// --------------------------------------------------------

bool EdgeSplitter::split_edge( size_t edge, size_t& result_vert, bool ignore_bad_angles, bool use_specified_point, Vec3d const * pos, const std::vector<size_t> & ignore_vertices, bool ignore_min_length )
{
    //if (edge == 382) std::cout << "5.1" << std::endl;
    
    g_stats.add_to_int( "EdgeSplitter:edge_split_attempts", 1 );
    
    assert( edge_is_splittable(edge, ignore_min_length) );
    
    NonDestructiveTriMesh& mesh = m_surf.m_mesh;
    
    // --------------
    // Collect all the triangles around the edge
    
    const std::vector<size_t> incident_tris = mesh.m_edge_to_triangle_map[edge];
    std::vector<double> tri_areas;
    for(size_t i = 0; i < incident_tris.size(); ++i) {
        tri_areas.push_back(m_surf.get_triangle_area(incident_tris[i]));
        
        // Splitting near-degenerate triangles can cause problems - but ignore this if we're being aggressive
        if(tri_areas[i] < m_surf.m_min_triangle_area && !ignore_bad_angles && !m_surf.m_aggressive_mode) {
            g_stats.add_to_int( "EdgeSplitter:split_edge_incident_to_tiny_triangle", 1 );
            return false;
        }
    }
    
    //if (edge == 382) std::cout << "5.2" << std::endl;
    
    // --------------
    // convert each incident triangle abc into a pair of triangles aec, ebc
    
    size_t vertex_a = mesh.m_edges[edge][0];
    size_t vertex_b = mesh.m_edges[edge][1];
    
    // Collect the incident verts
    std::vector<size_t> other_verts;
    for(size_t i = 0; i < incident_tris.size(); ++i) {
        size_t cur_tri = incident_tris[i];
        Vec3st tri_data = mesh.get_triangle(cur_tri);
        other_verts.push_back(mesh.get_third_vertex(vertex_a, vertex_b, tri_data));
    }
    
    // --------------
    // set up point data for the various options
    
    Vec3d new_vertex_average_position = 0.5 * ( m_surf.get_position( vertex_a ) + m_surf.get_position( vertex_b ) );
    Vec3d new_vertex_smooth_position;
    Vec3d new_vertex_constrained_position;
    Vec3d new_vertex_specified_position = use_specified_point? *pos : Vec3d(0,0,0);
    
    Vec3d new_vertex_proposed_final_position;
    
    //if (edge == 382) std::cout << "5.3" << std::endl;
    
    // Track which one we decide on.
    // Smooth point will fall back to midpoint, whereas specified and constrained points simply fail out.
    bool use_smooth_point;
    bool use_average_point;
    bool use_constrained_point;
    
    Vec3c new_vert_solid_label = Vec3c(false, false, false);
    
    // Try to decide what point to use
    if(use_specified_point) {
        // Use the specified if one is provided as input.
        
        use_smooth_point = false;
        use_average_point = false;
        use_constrained_point = false;
        
        new_vertex_proposed_final_position = *pos;
        if (m_surf.m_solid_vertices_callback)
            new_vert_solid_label = m_surf.m_solid_vertices_callback->generate_split_solid_label(m_surf, vertex_a, vertex_b, m_surf.vertex_is_solid_3(vertex_a), m_surf.vertex_is_solid_3(vertex_b));
    }
    else if (m_surf.vertex_is_any_solid(vertex_a) || m_surf.vertex_is_any_solid(vertex_b))
    {
        // Use the constraint callbacks if the edge has constraints
        
        use_constrained_point = true;
        use_smooth_point = false;
        use_average_point = false;
        use_specified_point = false;
        
        assert(m_surf.m_solid_vertices_callback);
        if (!m_surf.m_solid_vertices_callback->generate_split_position(m_surf, vertex_a, vertex_b, new_vertex_constrained_position))
        {
            if (m_surf.m_verbose || true) std::cout << "Constraint callback vetoed splitting" << std::endl;
            return false;
        }
        new_vert_solid_label = m_surf.m_solid_vertices_callback->generate_split_solid_label(m_surf, vertex_a, vertex_b, m_surf.vertex_is_solid_3(vertex_a), m_surf.vertex_is_solid_3(vertex_b));
    }
    else if( incident_tris.size() == 2 || typeid(*m_surf.m_subdivision_scheme) == typeid(ModifiedButterflyScheme)) {
        // Use smooth subdivision if the geometry and subd scheme will allow us
        use_smooth_point = true;
        use_average_point = false;
        use_constrained_point = false;
        m_surf.m_subdivision_scheme->generate_new_midpoint( edge, m_surf, new_vertex_smooth_position );
    }
    else {
        //otherwise, we'll just use the average/midpoint of the edge.
        use_smooth_point = false;
        use_constrained_point = false;
        use_average_point = true;
    }

    //if (edge == 382) std::cout << "5.4" << std::endl;

    // If we have chosen smooth subd, it may introduce intersections or normal flips,
    // and if so we will fall back to midpoint
    if(use_smooth_point) {
        
        use_smooth_point = !split_edge_pseudo_motion_introduces_intersection( new_vertex_average_position,
                                                                             new_vertex_smooth_position,
                                                                             edge,
                                                                             vertex_a,
                                                                             vertex_b,
                                                                             incident_tris,
                                                                             other_verts,
                                                                             ignore_vertices);
        
        if ( !use_smooth_point ) {
            g_stats.add_to_int( "EdgeSplitter:split_smooth_vertex_collisions", 1 ); }
        
        //only check normals if we passed collision-safety
        if ( use_smooth_point )
        {
            size_t tri0 = incident_tris[0];
            size_t tri1 = incident_tris[1];
            size_t vertex_c = other_verts[0];
            size_t vertex_d = other_verts[1];
            
            // ensure we're using the right triangle orientations (consistent with old splitting code)
            if ( !mesh.oriented( vertex_a, vertex_b, mesh.get_triangle(tri0) ) )
                swap(vertex_c, vertex_d);
            
            Vec3d tri0_normal = m_surf.get_triangle_normal( tri0 );
            Vec3d tri1_normal = m_surf.get_triangle_normal( tri1 );
            
            //
            //  note: we consider the case where tri0 and tri1 have opposite orientation
            //
            
            if ( dot( tri0_normal, tri1_normal ) >= 0.0 && mesh.oriented(vertex_a, vertex_b, mesh.get_triangle(tri0)) == mesh.oriented(vertex_b, vertex_a, mesh.get_triangle(tri1)) )
            {
                Vec3d new_normal = triangle_normal( m_surf.get_position(vertex_a), new_vertex_smooth_position, m_surf.get_position(vertex_c) );
                if ( dot( new_normal, tri0_normal ) < 0.0 || dot( new_normal, tri1_normal ) < 0.0 )
                {
                    use_smooth_point = false;
                }
                new_normal = triangle_normal( m_surf.get_position(vertex_c), new_vertex_smooth_position, m_surf.get_position(vertex_b) );
                if ( dot( new_normal, tri0_normal ) < 0.0 || dot( new_normal, tri1_normal ) < 0.0 )
                {
                    use_smooth_point = false;
                }
                new_normal = triangle_normal( m_surf.get_position(vertex_d), m_surf.get_position(vertex_b), new_vertex_smooth_position );
                if ( dot( new_normal, tri0_normal ) < 0.0 || dot( new_normal, tri1_normal ) < 0.0 )
                {
                    use_smooth_point = false;
                }
                new_normal = triangle_normal( m_surf.get_position(vertex_d), new_vertex_smooth_position, m_surf.get_position(vertex_a) );
                if ( dot( new_normal, tri0_normal ) < 0.0 || dot( new_normal, tri1_normal ) < 0.0 )
                {
                    use_smooth_point = false;
                }
            } else if ( dot( tri0_normal, tri1_normal ) <= 0.0 && mesh.oriented(vertex_a, vertex_b, mesh.get_triangle(tri0)) != mesh.oriented(vertex_b, vertex_a, mesh.get_triangle(tri1)) )
            {
                Vec3d new_normal = triangle_normal( m_surf.get_position(vertex_a), new_vertex_smooth_position, m_surf.get_position(vertex_c) );
                if ( dot( new_normal, tri0_normal ) < 0.0 || dot( new_normal, tri1_normal ) > 0.0 )
                {
                    use_smooth_point = false;
                }
                new_normal = triangle_normal( m_surf.get_position(vertex_c), new_vertex_smooth_position, m_surf.get_position(vertex_b) );
                if ( dot( new_normal, tri0_normal ) < 0.0 || dot( new_normal, tri1_normal ) > 0.0 )
                {
                    use_smooth_point = false;
                }
                new_normal = triangle_normal( m_surf.get_position(vertex_d), m_surf.get_position(vertex_b), new_vertex_smooth_position );
                if ( dot( new_normal, tri0_normal ) < 0.0 || dot( new_normal, tri1_normal ) > 0.0 )
                {
                    use_smooth_point = false;
                }
                new_normal = triangle_normal( m_surf.get_position(vertex_d), new_vertex_smooth_position, m_surf.get_position(vertex_a) );
                if ( dot( new_normal, tri0_normal ) < 0.0 || dot( new_normal, tri1_normal ) > 0.0 )
                {
                    use_smooth_point = false;
                }
            }
            
        }
        
        //if we've now decided not to use the smooth point, switch to average/mid-point
        use_average_point = !use_smooth_point;
        
    }

    //if (edge == 382) std::cout << "5.5" << std::endl;

    //at this stage, if smooth is successful, we can simply go ahead with the smooth subdivision point - we're safe.
    if(use_smooth_point) {
        new_vertex_proposed_final_position = new_vertex_smooth_position;
    }
    else {
        //otherwise, we're going with either average point, specified point, or constrained point.
        //we choose its vertex, and check it for safety.
        if(use_constrained_point)
            new_vertex_proposed_final_position = new_vertex_constrained_position;
        else if(use_specified_point)
            new_vertex_proposed_final_position = new_vertex_specified_position;
        else if(use_average_point)
            new_vertex_proposed_final_position = new_vertex_average_position;
        else {
            assert(false); //not allowed to get here
        }
        
        //now check the proposed final position for collision safety.
        //if this fails, we simply drop out - there's nowhere else to go.
        if ( m_surf.m_verbose ) { std::cout << "checking proposed final point for safety" << std::endl; }
        
        if ( split_edge_pseudo_motion_introduces_intersection( new_vertex_average_position,
                                                              new_vertex_proposed_final_position,
                                                              edge,
                                                              vertex_a,
                                                              vertex_b,
                                                              incident_tris,
                                                              other_verts,
                                                              ignore_vertices) )
        {
            
            g_stats.add_to_int( "EdgeSplitter:split_final_collisions", 1 );
            if ( m_surf.m_verbose )  { std::cout << "Final proposed point introduces collision.  Backing out." << std::endl; }
            
            return false;
        }
        
    }

    //if (edge == 382) std::cout << "5.6" << std::endl;

    // --------------
    
    //At this stage, we have chosen the point we want to stick with and it is collision-safe.
    //now we need to do some final checks, and then proceed.
    
    //Don't allow splitting to create edges shorter than half the minimum.
    const Vec3d& va = m_surf.get_position(vertex_a);
    const Vec3d& vb = m_surf.get_position(vertex_b);
    if (!ignore_min_length)
    {
        if(!m_surf.m_aggressive_mode) {
            if(mag(new_vertex_proposed_final_position-va) < 0.5*m_surf.edge_min_edge_length(edge) ||
               mag(new_vertex_proposed_final_position-vb) < 0.5*m_surf.edge_min_edge_length(edge))
                return false;
        }
        else { //if we're being aggressive, only stop if we're getting really extreme.
            if(mag(new_vertex_proposed_final_position-va) < m_surf.m_hard_min_edge_len ||
               mag(new_vertex_proposed_final_position-vb) < m_surf.m_hard_min_edge_len)
                return false;
        }
    }
    
    // --------------
    
    //if (edge == 382) std::cout << "5.7" << std::endl;

    // Check angles on new triangles

    std::vector<Vec3d> other_vert_pos;
    for(size_t i = 0; i < other_verts.size(); ++i)
        other_vert_pos.push_back(m_surf.get_position(other_verts[i]));
    
    double min_current_angle = 2*M_PI;
    for(size_t i = 0; i < incident_tris.size(); ++i) {
        min_current_angle = min( min_current_angle, min_triangle_angle( va, vb, other_vert_pos[i] ) );
    }
    
    double min_new_angle = 2*M_PI;
    for(size_t i = 0; i < other_vert_pos.size(); ++i) {
        min_new_angle = min( min_new_angle, min_triangle_angle( va, new_vertex_proposed_final_position, other_vert_pos[i] ) );
        min_new_angle = min( min_new_angle, min_triangle_angle( vb, new_vertex_proposed_final_position, other_vert_pos[i] ) );
    }
    
    if ( !ignore_bad_angles && rad2deg(min_new_angle) < m_surf.m_min_triangle_angle )
    {
        //only cancel if it actively makes things worse!
        if(min_new_angle < min_current_angle) {
            g_stats.add_to_int( "EdgeSplitter:edge_split_small_angle", 1 );
            return false;
        }
    }
    
    double max_current_angle = 0;
    for(size_t i = 0; i < incident_tris.size(); ++i) {
        max_current_angle = max( max_current_angle, max_triangle_angle( va, vb, other_vert_pos[i] ) );
    }
    
    double max_new_angle = 0;
    
    for(size_t i = 0; i < other_vert_pos.size(); ++i) {
        max_new_angle = max( max_new_angle, max_triangle_angle( va, new_vertex_proposed_final_position, other_vert_pos[i] ) );
        max_new_angle = max( max_new_angle, max_triangle_angle( vb, new_vertex_proposed_final_position, other_vert_pos[i] ) );
    }
    
    // if new angle is greater than the allowed angle, and doesn't
    // improve the current max angle, prevent the split
    
    if ( !ignore_bad_angles && rad2deg(max_new_angle) > m_surf.m_max_triangle_angle )
    {
        
        // if new triangle improves a large angle, allow it
        if ( rad2deg(max_new_angle) < rad2deg(max_current_angle) )
        {
            g_stats.add_to_int( "EdgeSplitter:edge_split_large_angle", 1 );
            return false;
        }
    }
    
    //if (edge == 382) std::cout << "5.8" << std::endl;

    // --------------

    // Do the actual splitting

    void * data = NULL;
    if (m_surf.m_mesheventcallback)
        m_surf.m_mesheventcallback->pre_split(m_surf, edge, &data);

    Vec3d new_vertex_mass = 0.5 * ( m_surf.m_masses[ vertex_a ] + m_surf.m_masses[ vertex_b ] );
    for (int i = 0; i < 3; i++)
    {
        if (new_vert_solid_label[i])
            new_vertex_mass[i] = DynamicSurface::solid_mass();
        else
            new_vertex_mass[i] = 1;
    }
    size_t vertex_e = m_surf.add_vertex( new_vertex_proposed_final_position, new_vertex_mass );
    
    // Add to change history
    m_surf.m_vertex_change_history.push_back( VertexUpdateEvent( VertexUpdateEvent::VERTEX_ADD, vertex_e, Vec2st( vertex_a, vertex_b) ) );
    
    if ( m_surf.m_verbose ) { std::cout << "new vertex: " << vertex_e << std::endl; }
    
    
    // Determine the labels of the new triangles
    //
    std::vector<Vec2i> created_tri_label;
    
    // Create new triangles with proper orientations (match their parents)
    std::vector<Vec3st> created_tri_data;
    for(size_t i = 0; i < other_verts.size(); ++i) {
        Vec3st newtri0, newtri1;
        if ( mesh.oriented( vertex_a, vertex_b, mesh.get_triangle( incident_tris[i] ) ) ) {
            newtri0 = Vec3st( vertex_a, vertex_e, other_verts[i] );
            newtri1 = Vec3st( other_verts[i], vertex_e, vertex_b );
        }
        else {
            newtri0 = Vec3st( vertex_a, other_verts[i], vertex_e );
            newtri1 = Vec3st( other_verts[i], vertex_b, vertex_e );
        }
        created_tri_data.push_back(newtri0);
        created_tri_data.push_back(newtri1);
        
        // the old label carries over to the new triangle
        //
        Vec2i old_label = m_surf.m_mesh.get_triangle_label(incident_tris[i]);
        created_tri_label.push_back(old_label);
        created_tri_label.push_back(old_label);
        
    }
    
    // Delete the parent triangles
    for(size_t i = 0; i < incident_tris.size(); ++i) {
        m_surf.remove_triangle( incident_tris[i] );
    }
    
    // Now actually add the triangles to the mesh
    std::vector<size_t> created_tris;
    for(size_t i = 0; i < created_tri_data.size(); ++i) {
        
        //add the triangle, with the same old label
        size_t newtri0_id = m_surf.add_triangle( created_tri_data[i], created_tri_label[i] );
        
        //record the data created
        created_tris.push_back(newtri0_id);
    }
    
    // interpolate the remeshing velocities onto the new vertex
    m_surf.pm_velocities[vertex_e] = Vec3d(0, 0, 0);
    std::pair<Vec3d, double> laplacian = m_surf.laplacian(vertex_e, m_surf.pm_velocities);
    m_surf.pm_velocities[vertex_e] = laplacian.first / laplacian.second;

    // update the target edge length
    double new_target_edge_length = m_surf.compute_vertex_target_edge_length(vertex_e);
    std::vector<size_t> new_onering;
    m_surf.m_mesh.get_adjacent_vertices(vertex_e, new_onering);
    for (size_t i = 0; i < new_onering.size(); i++)
        if (new_target_edge_length > m_surf.vertex_target_edge_length(new_onering[i]) * m_surf.m_max_adjacent_target_edge_length_ratio)
            new_target_edge_length = m_surf.vertex_target_edge_length(new_onering[i]) * m_surf.m_max_adjacent_target_edge_length_ratio;
    m_surf.m_target_edge_lengths[vertex_e] = new_target_edge_length;
    
    // Add to new history log
    MeshUpdateEvent split(MeshUpdateEvent::EDGE_SPLIT);
    split.m_v0 = vertex_a;
    split.m_v1 = vertex_b;
    split.m_vert_position = new_vertex_proposed_final_position;
    split.m_created_verts.push_back(vertex_e);
    for(size_t i = 0; i < incident_tris.size(); ++i)
        split.m_deleted_tris.push_back(incident_tris[i]);
    split.m_created_tris = created_tris;
    split.m_created_tri_data = created_tri_data;
    split.m_created_tri_labels = created_tri_label;
    
    m_surf.m_mesh_change_history.push_back(split);
    
    if (m_surf.m_mesheventcallback)
    {
        if (ignore_bad_angles && use_specified_point)
            m_surf.m_mesheventcallback->log() << "Edge split: large angle split" << std::endl;
        else if (!ignore_bad_angles && use_specified_point)
            m_surf.m_mesheventcallback->log() << "Edge split: snap" << std::endl;
        else
            m_surf.m_mesheventcallback->log() << "Edge split: long edge split" << std::endl;
        
        m_surf.m_mesheventcallback->post_split(m_surf, edge, vertex_e, data);
    }
    
    ////////////////////////////////////////////////////////////
    
    //store the resulting vertex as output.
    result_vert = vertex_e;

    return true;
    
}


// --------------------------------------------------------
///
/// Determine if an edge split is desirable
///
// --------------------------------------------------------


bool EdgeSplitter::edge_length_needs_split(size_t edge_index) {
    
    double edge_length = m_surf.get_edge_length(edge_index);
    size_t vertex_a = m_surf.m_mesh.m_edges[edge_index][0];
    size_t vertex_b = m_surf.m_mesh.m_edges[edge_index][1];
    
    if ( m_use_curvature )
    {
        //split if we're above the upper limit
        if(edge_length > m_surf.edge_max_edge_length(edge_index))
            return true;
        
        //don't split if splitting would take us below the lower limit
        if(edge_length < 2*m_surf.edge_min_edge_length(edge_index))
            return false;
        
        double curvature_value = get_edge_curvature( m_surf, vertex_a, vertex_b );
        int circlesegs = 16;
        double curvature_max_length = 2*M_PI / (double)circlesegs / max(curvature_value, 1e-8);
        
        //split if curvature dictates
        if(edge_length > curvature_max_length)
            return true;
        
        //check all incident edges to see if any of them are super short, and if so, split this guy accordingly.
        //this enforces slow grading of the mesh.
        double min_nbr_len = edge_length;
        for(size_t edge_id = 0; edge_id < m_surf.m_mesh.m_vertex_to_edge_map[vertex_a].size(); ++edge_id) {
            min_nbr_len = min(min_nbr_len, m_surf.get_edge_length(m_surf.m_mesh.m_vertex_to_edge_map[vertex_a][edge_id]));
        }
        for(size_t edge_id = 0; edge_id < m_surf.m_mesh.m_vertex_to_edge_map[vertex_b].size(); ++edge_id) {
            min_nbr_len = min(min_nbr_len, m_surf.get_edge_length(m_surf.m_mesh.m_vertex_to_edge_map[vertex_b][edge_id]));
        }
        
        if(edge_length > min_nbr_len * 3)
            return true;
        
    }
    else {
        return edge_length > m_surf.edge_max_edge_length(edge_index);
    }
    
    return false;
    
}

// --------------------------------------------------------
///
/// Determine if edge should be allowed to be split
///
// --------------------------------------------------------

bool EdgeSplitter::edge_is_splittable( size_t edge_index, bool ignore_min_length )
{
    
    // skip deleted and solid edges
    if ( m_surf.m_mesh.edge_is_deleted(edge_index) ) { return false; }
    //  if ( m_surf.edge_is_all_solid(edge_index) ) { return false; }
    
    //if not remeshing boundary edges, skip those too
    if ( !m_remesh_boundaries && m_surf.m_mesh.m_is_boundary_edge[edge_index]) { return false; }
    
    if (!ignore_min_length)
    {
        if(m_surf.m_aggressive_mode) {
            //only disallow splitting if the edge is smaller than a quite small bound.
            if(m_surf.get_edge_length(edge_index) < m_surf.m_hard_min_edge_len)
                return false;
        }
        else {
            if(m_surf.get_edge_length(edge_index) < m_surf.edge_min_edge_length(edge_index))
                return false;
        }
    }
    
    return true;
    
}
// --------------------------------------------------------
///
/// Split edges opposite large angles
///
// --------------------------------------------------------

bool EdgeSplitter::large_angle_split_pass()
{
    static double large_angle_split_cos_bound = cos(deg2rad(m_surf.m_large_triangle_angle_to_split));
    static double max_tri_angle_cos_bound = cos(deg2rad(m_surf.m_max_triangle_angle));

    NonDestructiveTriMesh& mesh = m_surf.m_mesh;
    
    bool split_occurred = false;
    
    for ( size_t e = 0; e < mesh.m_edges.size(); ++e )
    {
        
        if ( !edge_is_splittable(e) ) { continue; }
        
        // get edge end points
        const Vec2st& edge = m_surf.m_mesh.m_edges[e];      
        const Vec3d& edge_point0 = m_surf.get_position( edge[0] );
        const Vec3d& edge_point1 = m_surf.get_position( edge[1] );
        
        double edge_length = m_surf.get_edge_length(e);
        
        // get triangles incident to the edge
        const std::vector<size_t>& incident_tris = mesh.m_edge_to_triangle_map[e];
        for(size_t t = 0; t < incident_tris.size(); ++t) {
            if(mesh.triangle_is_deleted(incident_tris[t]))
                continue;
            
            const Vec3st& tri0 = mesh.get_triangle(incident_tris[t]);
            
            
            // get vertex opposite the edge for each triangle
            size_t opposite0 = mesh.get_third_vertex( e, tri0 );
            
            // compute the angle at each opposite vertex
            const Vec3d& opposite_point0 = m_surf.get_position(opposite0);
            //double acos_input = dot( normalized(edge_point0-opposite_point0), normalized(edge_point1-opposite_point0) );
            double acos_input = corner_normalized_dot(edge_point0, edge_point1, opposite_point0);
            if (acos_input != acos_input || acos_input <= -1 || acos_input >= 1) {
                std::cout << "Value: " << acos_input << std::endl;
                std::cout << "edgepoint0:" << edge_point0 << "  edge_point1: " << edge_point1 << "   Opp0:" << opposite_point0 << std::endl;
                std::cout << "Difone: " << edge_point0-opposite_point0 <<  "  Diftwo: " << edge_point1-opposite_point0 << std::endl;
                std::cout << "Left: " << mag(edge_point0-opposite_point0)  <<  "  Right: " << mag(edge_point1-opposite_point0) << std::endl;
                assert(false);
            }
            //double angle0 = rad2deg( acos( acos_input ) );
            
            // if an angle is above the max threshold, split the edge
            
            //if aggressive, use hard max, otherwise use soft max
             //if ( !m_surf.m_aggressive_mode && angle0 > m_surf.m_large_triangle_angle_to_split ||
                //angle0 > m_surf.m_max_triangle_angle)
            if (!m_surf.m_aggressive_mode && acos_input < large_angle_split_cos_bound ||
                acos_input < max_tri_angle_cos_bound)
            {
                if (!edge_is_splittable(e))
                    continue;
                
                Vec3d ev = edge_point1 - edge_point0;
                Vec3d split_pos = dot(opposite_point0 - edge_point0, ev) / dot(ev, ev) * ev + edge_point0;
                
                size_t result_vert;
                bool result = split_edge( e, result_vert, true, true, &split_pos ); // use the projection of opposite_point0 onto the edge as the split point
                
                if ( result )
                {
                    g_stats.add_to_int( "EdgeSplitter:large_angle_split_success", 1 );
                }
                else
                {
                    g_stats.add_to_int( "EdgeSplitter:large_angle_split_failed", 1 );
                }
                
                split_occurred |= result;
            }
        }
    }
    
    
    return split_occurred;
    
}


// --------------------------------------------------------
///
/// Split all long edges
///
// --------------------------------------------------------

bool EdgeSplitter::split_pass()
{
    
    if ( m_surf.m_verbose )
    {
        std::cout << "---------------------- Edge Splitter: splitting ----------------------" << std::endl;
    }
    
    // whether a split operation was successful in this pass
    bool split_occurred = false;
    
//    assert( m_max_edge_length != UNINITIALIZED_DOUBLE );
    
    NonDestructiveTriMesh& mesh = m_surf.m_mesh;
    std::vector<SortableEdge> sortable_edges_to_try;
    
    //only do length-based splitting in regular mode.
    if(!m_surf.m_aggressive_mode) {
        for( size_t i = 0; i < mesh.m_edges.size(); i++ )
        {    
            if ( !edge_is_splittable(i) ) { continue; }
            
            bool should_split = edge_length_needs_split(i);
            if(should_split)
                sortable_edges_to_try.push_back( SortableEdge( i, m_surf.get_edge_length(i)) );
        }
        
        
        //
        // sort in ascending order, then iterate backwards to go from longest edge to shortest
        //
        
        std::sort( sortable_edges_to_try.begin(), sortable_edges_to_try.end() );
        
        std::vector<SortableEdge>::reverse_iterator iter = sortable_edges_to_try.rbegin();
        
        for ( ; iter != sortable_edges_to_try.rend(); ++iter )
        {
            size_t longest_edge = iter->m_edge_index;
            if ( !edge_is_splittable(longest_edge) ) { continue; }
            
            bool should_split = edge_length_needs_split(longest_edge);
            
            if(should_split) {
                size_t result_vert;
                
                bool result = split_edge(longest_edge, result_vert);
                
                split_occurred |= result;
            }
            
        }
    }
    
    // Now split to reduce large angles
    bool large_angle_split_occurred = large_angle_split_pass();
    
    return split_occurred || large_angle_split_occurred;
    
}
    
}
