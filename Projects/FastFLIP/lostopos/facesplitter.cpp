// ---------------------------------------------------------
//
//  facesplitter.cpp
//  Christopher Batty, Fang Da 2014
//
//  Functions supporting the "face split" operation: subdividing a face into three smaller triangles.
//
// ---------------------------------------------------------

#include <facesplitter.h>
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

FaceSplitter::FaceSplitter( SurfTrack& surf ) :
m_surf( surf )
{}


// --------------------------------------------------------
///
/// Check collisions between the edge [neighbour, new] and the given edge
///
// --------------------------------------------------------

bool FaceSplitter::split_edge_edge_collision( size_t neighbour_index,
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
/// Determine if the new vertex introduced by the face split has a collision along its pseudo-trajectory.
///
// ---------------------------------------------------------

bool FaceSplitter::split_triangle_vertex_collision( const Vec3st& triangle_indices,
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
bool FaceSplitter::split_face_pseudo_motion_introduces_intersection( const Vec3d& new_vertex_position,
                                                                    const Vec3d& new_vertex_smooth_position,
                                                                    size_t face)
{
    
    NonDestructiveTriMesh& m_mesh = m_surf.m_mesh;
    
    if ( !m_surf.m_collision_safety)
    {
        return false;
    }
    
    //Get the main face's vertices
    size_t vertex_a = m_mesh.m_tris[face][0],
    vertex_b = m_mesh.m_tris[face][1],
    vertex_c = m_mesh.m_tris[face][2];
    
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
            
            // Exclude the central triangle from the check
            if(overlapping_triangles[i] == face)
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
               m_surf.get_position( vertex_a ),
               m_surf.get_position( vertex_b ),
               m_surf.get_position( vertex_c ),
               edge_aabb_low, edge_aabb_high );
        
        edge_aabb_low -= m_surf.m_aabb_padding * Vec3d(1,1,1);
        edge_aabb_high += m_surf.m_aabb_padding * Vec3d(1,1,1);
        
        std::vector<size_t> overlapping_edges;
        m_surf.m_broad_phase->get_potential_edge_collisions( edge_aabb_low, edge_aabb_high, true, true, overlapping_edges );
        
        std::vector<size_t> vertex_neighbourhood;
        vertex_neighbourhood.push_back(vertex_a);
        vertex_neighbourhood.push_back(vertex_b);
        vertex_neighbourhood.push_back(vertex_c);
        
        for ( size_t i = 0; i < overlapping_edges.size(); ++i )
        {
            
            if ( m_mesh.edge_is_deleted(overlapping_edges[i])) continue;
            
            for ( size_t v = 0; v < vertex_neighbourhood.size(); ++v )
            {
                bool collision = split_edge_edge_collision( vertex_neighbourhood[v],
                                                           new_vertex_position,
                                                           new_vertex_smooth_position,
                                                           m_mesh.m_edges[overlapping_edges[i]] );
                
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
               m_surf.get_position( vertex_a ),
               m_surf.get_position( vertex_b ),
               m_surf.get_position( vertex_c ),
               triangle_aabb_low, triangle_aabb_high );
        
        triangle_aabb_low -= m_surf.m_aabb_padding * Vec3d(1,1,1);
        triangle_aabb_high += m_surf.m_aabb_padding * Vec3d(1,1,1);
        
        std::vector<size_t> overlapping_vertices;
        m_surf.m_broad_phase->get_potential_vertex_collisions( triangle_aabb_low, triangle_aabb_high, true, true, overlapping_vertices );
        
        size_t dummy_e = m_surf.get_num_vertices();
        
        std::vector< Vec3st > triangle_indices;
        
        triangle_indices.push_back( Vec3st( vertex_a, vertex_b, dummy_e ) );    // triangle abd
        triangle_indices.push_back( Vec3st( vertex_b, vertex_c, dummy_e ) );    // triangle bce
        triangle_indices.push_back( Vec3st( vertex_c, vertex_a, dummy_e ) );    // triangle cae
        
        for ( size_t i = 0; i < overlapping_vertices.size(); ++i )
        {
            if ( m_mesh.m_vertex_to_triangle_map[overlapping_vertices[i]].empty() )
            {
                continue;
            }
            
            size_t overlapping_vert_index = overlapping_vertices[i];
            const Vec3d& vert = m_surf.get_position(overlapping_vert_index);
            
            for ( size_t j = 0; j < triangle_indices.size(); ++j )
            {
                bool collision = split_triangle_vertex_collision( triangle_indices[j],
                                                                 new_vertex_position,
                                                                 new_vertex_smooth_position,
                                                                 overlapping_vert_index,
                                                                 vert );
                
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
/// Split a face, using the barycenter as the subdivision point, if safe to do so.
///
// --------------------------------------------------------

bool FaceSplitter::split_face( size_t face, size_t& result_vertex, bool specify_point, Vec3d const * goal_point )
{
    
    g_stats.add_to_int( "FaceSplitter:face_split_attempts", 1 );
    
    assert( face_is_splittable(face) );
    
    NonDestructiveTriMesh& mesh = m_surf.m_mesh;
    
    if(m_surf.get_triangle_area(face) < m_surf.m_min_triangle_area) {
        g_stats.add_to_int( "FaceSplitter:split_face_incident_to_tiny_triangle", 1 );
        return false;
    }
    
    
    // --------------
    
    // convert the central triangle abc into a trio of triangles abd, bcd, cad
    
    size_t vertex_a = mesh.m_tris[face][0];
    size_t vertex_b = mesh.m_tris[face][1];
    size_t vertex_c = mesh.m_tris[face][2];
    
    
    // --------------
    
    // get face midpoint as default split point.
    Vec3d new_vertex_position =  (1.0/3.0) * ( m_surf.get_position( vertex_a ) +
                                              m_surf.get_position( vertex_b )  +
                                              m_surf.get_position( vertex_c ) );
    
    //TODO Consider adding some kind of smooth subdivision, e.g. "interpolatory sqrt(3) subdivision"
    
    //TODO Treat partially constrained triangles more carefully.
    
    if (m_surf.triangle_is_all_solid(face)) return false;
    
    
    // Check angles on new triangles
    
    const Vec3d& va = m_surf.get_position( vertex_a );
    const Vec3d& vb = m_surf.get_position( vertex_b );
    const Vec3d& vc = m_surf.get_position( vertex_c );
    
    double min_new_angle = 2*M_PI;
    min_new_angle = min( min_new_angle, min_triangle_angle( va, vb, new_vertex_position ) );
    min_new_angle = min( min_new_angle, min_triangle_angle( vb, vc, new_vertex_position ) );
    min_new_angle = min( min_new_angle, min_triangle_angle( vc, va, new_vertex_position ) );
    
    if ( rad2deg(min_new_angle) < m_surf.m_min_triangle_angle )
    {
        g_stats.add_to_int( "FaceSplitter:face_split_small_angle", 1 );
        return false;
    }
    
    double max_current_angle = max_triangle_angle(va, vb, vc);
    
    double max_new_angle = 0;
    max_new_angle = min( max_new_angle, max_triangle_angle( va, vb, new_vertex_position ) );
    max_new_angle = min( max_new_angle, max_triangle_angle( vb, vc, new_vertex_position ) );
    max_new_angle = min( max_new_angle, max_triangle_angle( vc, va, new_vertex_position ) );
    
    // if new angle is greater than the allowed angle, and doesn't
    // improve the current max angle, prevent the split
    
    if ( rad2deg(max_new_angle) > m_surf.m_max_triangle_angle )
    {
        
        // if new triangle improves a large angle, allow it
        if ( rad2deg(max_new_angle) < rad2deg(max_current_angle) )
        {
            g_stats.add_to_int( "EdgeSplitter:edge_split_large_angle", 1 );
            return false;
        }
    }
    
    //Check that we do not introduce (very) short edges
    
    if(mag(va - new_vertex_position) < 0.5*m_surf.vertex_min_edge_length(vertex_a) ||
       mag(vb - new_vertex_position) < 0.5*m_surf.vertex_min_edge_length(vertex_b) ||
       mag(vc - new_vertex_position) < 0.5*m_surf.vertex_min_edge_length(vertex_c))
        return false;
    
    
    // --------------
    
    // check if the generated point introduces an intersection
    
    // --------------
    
    Vec3d new_vertex_smooth_position = new_vertex_position;
    bool point_okay = false;
    if(specify_point) {
        point_okay = ! ( split_face_pseudo_motion_introduces_intersection( new_vertex_position, *goal_point, face) );
        
        if ( point_okay )
            new_vertex_smooth_position = *goal_point;
    }
    
    if(!point_okay) { //try the default barycenter point instead.
        if ( !split_face_pseudo_motion_introduces_intersection( new_vertex_position, new_vertex_smooth_position, face) )
        {
            g_stats.add_to_int( "FaceSplitter:split_midpoint_collisions", 1 );
            
            if ( m_surf.m_verbose )  { 
                std::cout << "Even mid-point subdivision introduces collision.  Backing out." << std::endl; 
            } 
            
            return false;
        }
    }
    
    // --------------
    
    // Do the actual splitting

    void * data = NULL;
    if (m_surf.m_mesheventcallback)
        m_surf.m_mesheventcallback->pre_facesplit(m_surf, face, &data);

    Vec3d  new_vertex_mass = Vec3d(1, 1, 1);
    if (m_surf.vertex_is_solid(vertex_a, 0) && m_surf.vertex_is_solid(vertex_b, 0) && m_surf.vertex_is_solid(vertex_c, 0)) new_vertex_mass[0] = DynamicSurface::solid_mass();
    if (m_surf.vertex_is_solid(vertex_a, 1) && m_surf.vertex_is_solid(vertex_b, 1) && m_surf.vertex_is_solid(vertex_c, 1)) new_vertex_mass[1] = DynamicSurface::solid_mass();
    if (m_surf.vertex_is_solid(vertex_a, 2) && m_surf.vertex_is_solid(vertex_b, 2) && m_surf.vertex_is_solid(vertex_c, 2)) new_vertex_mass[2] = DynamicSurface::solid_mass();
    
    size_t vertex_d = m_surf.add_vertex( new_vertex_smooth_position, new_vertex_mass );
    
    
    //TODO Fix constraint handling all around.
    //mesh.set_vertex_constraint_label(vertex_e, new_vert_constraint_label);
    
    
    // Add to change history
    m_surf.m_vertex_change_history.push_back( VertexUpdateEvent( VertexUpdateEvent::VERTEX_ADD, vertex_d, Vec2st( vertex_a, vertex_b) ) );
    
    if ( m_surf.m_verbose ) { std::cout << "new vertex: " << vertex_d << std::endl; }
    
    
    Vec2i old_label = m_surf.m_mesh.get_triangle_label(face);
    Vec3st old_tri = m_surf.m_mesh.m_tris[face];
    
    // Create new triangles with proper orientations (match the parent)
    std::vector<Vec3st> created_tri_data;
    std::vector<Vec2i> created_tri_label;
    for(size_t i = 0; i < 3; ++i) {
        Vec3st newtri;
        newtri[0] = (size_t)old_tri[(int)i];
        newtri[1] = (size_t)old_tri[(i+1)%3];
        newtri[2] = vertex_d;
        
        created_tri_data.push_back(newtri);
        created_tri_label.push_back(old_label);
    }
    
    // Delete the parent triangle
    
    m_surf.remove_triangle( face );
    
    // Now actually add the triangles to the mesh
    std::vector<size_t> created_tris;
    for(size_t i = 0; i < created_tri_data.size(); ++i) {
        //add the triangle
        size_t newtri0_id = m_surf.add_triangle( created_tri_data[i], created_tri_label[i] );
        
        //record the data created
        created_tris.push_back(newtri0_id);
    }
    
    // interpolate the remeshing velocities onto the new vertex
    m_surf.pm_velocities[vertex_d] = Vec3d(0, 0, 0);
    std::pair<Vec3d, double> laplacian = m_surf.laplacian(vertex_d, m_surf.pm_velocities);
    m_surf.pm_velocities[vertex_d] = laplacian.first / laplacian.second;

    // update the target edge length
    double new_target_edge_length = m_surf.compute_vertex_target_edge_length(vertex_d);
    std::vector<size_t> new_onering;
    m_surf.m_mesh.get_adjacent_vertices(vertex_d, new_onering);
    for (size_t i = 0; i < new_onering.size(); i++)
        if (new_target_edge_length > m_surf.vertex_target_edge_length(new_onering[i]) * m_surf.m_max_adjacent_target_edge_length_ratio)
            new_target_edge_length = m_surf.vertex_target_edge_length(new_onering[i]) * m_surf.m_max_adjacent_target_edge_length_ratio;
    m_surf.m_target_edge_lengths[vertex_d] = new_target_edge_length;
    
    // Add to new history log
    MeshUpdateEvent facesplit(MeshUpdateEvent::FACE_SPLIT);
    facesplit.m_created_verts.push_back(vertex_d);
    facesplit.m_deleted_tris.push_back(face);
    facesplit.m_created_tris = created_tris;
    facesplit.m_created_tri_data = created_tri_data;
    facesplit.m_created_tri_labels = created_tri_label;
    
    
    m_surf.m_mesh_change_history.push_back(facesplit);
    
    if (m_surf.m_mesheventcallback)
        m_surf.m_mesheventcallback->post_facesplit(m_surf, face, vertex_d, data);
    
    //return output vertex
    result_vertex = vertex_d;
    
    return true;
    
}


// --------------------------------------------------------
///
/// Determine if face should be allowed to be split
///
// --------------------------------------------------------

bool FaceSplitter::face_is_splittable( size_t face_index )
{
    
    // skip deleted and solid edges
    if ( m_surf.m_mesh.triangle_is_deleted(face_index) ) { return false; }
    
    //TODO handle solids and boundaries
    
    return true;
    
}
    
    
}
