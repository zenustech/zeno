// ---------------------------------------------------------
//
//  edgeflipper.cpp
//  Tyson Brochu 2011
//  Christopher Batty, Fang Da 2014
//
//  Functions supporting the "edge flip" operation: replacing non-delaunay edges with their dual edge.
//
// ---------------------------------------------------------

#include <edgeflipper.h>

#include <broadphase.h>
#include <collisionqueries.h>
#include <nondestructivetrimesh.h>
#include <runstats.h>
#include <surftrack.h>
#include <trianglequality.h>

#include <lapack_wrapper.h>

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
/// Check whether the new triangles created by flipping an edge introduce any intersection
///
// --------------------------------------------------------

bool EdgeFlipper::flip_introduces_collision( size_t edge_index,
                                            const Vec2st& new_edge,
                                            const Vec3st& new_triangle_a,
                                            const Vec3st& new_triangle_b )
{
    
    NonDestructiveTriMesh& m_mesh = m_surf.m_mesh;
    const std::vector<Vec3d>& xs = m_surf.get_positions();
    
    if ( !m_surf.m_collision_safety )
    {
        return false;
    }
    
    const Vec2st& old_edge = m_mesh.m_edges[edge_index];
    
    size_t tet_vertex_indices[4] = { old_edge[0], old_edge[1], new_edge[0], new_edge[1] };
    
    const Vec3d tet_vertex_positions[4] = { xs[ tet_vertex_indices[0] ],
        xs[ tet_vertex_indices[1] ],
        xs[ tet_vertex_indices[2] ],
        xs[ tet_vertex_indices[3] ] };
    
    Vec3d low, high;
    minmax( tet_vertex_positions[0], tet_vertex_positions[1], tet_vertex_positions[2], tet_vertex_positions[3], low, high );
    
    std::vector<size_t> overlapping_vertices;
    m_surf.m_broad_phase->get_potential_vertex_collisions( low, high, true, true, overlapping_vertices );
    
    // do point-in-tet tests
    for ( size_t i = 0; i < overlapping_vertices.size(); ++i )
    {
        if ( (overlapping_vertices[i] == old_edge[0]) || (overlapping_vertices[i] == old_edge[1]) ||
            (overlapping_vertices[i] == new_edge[0]) || (overlapping_vertices[i] == new_edge[1]) )
        {
            continue;
        }
        
        if ( point_tetrahedron_intersection( xs[overlapping_vertices[i]], overlapping_vertices[i],
                                            tet_vertex_positions[0], tet_vertex_indices[0],
                                            tet_vertex_positions[1], tet_vertex_indices[1],
                                            tet_vertex_positions[2], tet_vertex_indices[2],
                                            tet_vertex_positions[3], tet_vertex_indices[3] ) )
        {
            return true;
        }
    }
    
    //
    // Check new triangle A vs existing edges
    //
    
    minmax( xs[new_triangle_a[0]], xs[new_triangle_a[1]], xs[new_triangle_a[2]], low, high );
    std::vector<size_t> overlapping_edges;
    m_surf.m_broad_phase->get_potential_edge_collisions( low, high, true, true, overlapping_edges );
    
    for ( size_t i = 0; i < overlapping_edges.size(); ++i )
    {
        size_t overlapping_edge_index = overlapping_edges[i];
        const Vec2st& edge = m_mesh.m_edges[overlapping_edge_index];
        
        if ( check_edge_triangle_intersection_by_index( edge[0], edge[1],
                                                       new_triangle_a[0], new_triangle_a[1], new_triangle_a[2],
                                                       xs, m_surf.m_verbose ) )
        {
            return true;
        }
    }
    
    //
    // Check new triangle B vs existing edges
    //
    
    minmax( xs[new_triangle_b[0]], xs[new_triangle_b[1]], xs[new_triangle_b[2]], low, high );
    
    overlapping_edges.clear();
    m_surf.m_broad_phase->get_potential_edge_collisions( low, high, true, true, overlapping_edges );
    
    for ( size_t i = 0; i < overlapping_edges.size(); ++i )
    {
        size_t overlapping_edge_index = overlapping_edges[i];
        const Vec2st& edge = m_mesh.m_edges[overlapping_edge_index];
        
        if ( check_edge_triangle_intersection_by_index( edge[0], edge[1],
                                                       new_triangle_b[0], new_triangle_b[1], new_triangle_b[2],
                                                       xs, m_surf.m_verbose ) )
        {
            return true;
        }
    }
    
    //
    // Check new edge vs existing triangles
    //
    
    minmax( xs[new_edge[0]], xs[new_edge[1]], low, high );
    std::vector<size_t> overlapping_triangles;
    m_surf.m_broad_phase->get_potential_triangle_collisions( low, high, true, true, overlapping_triangles );
    
    for ( size_t i = 0; i <  overlapping_triangles.size(); ++i )
    {
        const Vec3st& tri = m_mesh.get_triangle(overlapping_triangles[i]);
        
        if ( check_edge_triangle_intersection_by_index( new_edge[0], new_edge[1],
                                                       tri[0], tri[1], tri[2],
                                                       xs, m_surf.m_verbose ) )
        {
            return true;
        }
    }
    
    return false;
    
}


// --------------------------------------------------------
///
/// Flip an edge: remove the edge and its incident triangles, then add a new edge and two new triangles
///
// --------------------------------------------------------

bool EdgeFlipper::flip_edge( size_t edge,
                            size_t tri0,
                            size_t tri1,
                            size_t third_vertex_0,
                            size_t third_vertex_1 )
{
    
    g_stats.add_to_int( "EdgeFlipper:edge_flip_attempt", 1 );
    
    NonDestructiveTriMesh& m_mesh = m_surf.m_mesh;
    const std::vector<Vec3d>& xs = m_surf.get_positions();
    
    Vec2st& edge_vertices = m_mesh.m_edges[edge];
    
    // Find the vertices which will form the new edge
    Vec2st new_edge( third_vertex_0, third_vertex_1);
    
    // --------------
    
    // Control volume change
    double vol = std::fabs( signed_volume( xs[edge_vertices[0]],
                                          xs[edge_vertices[1]],
                                          xs[new_edge[0]],
                                          xs[new_edge[1]] ) );
    
    if ( vol > m_surf.m_max_volume_change )
    {
        g_stats.add_to_int( "EdgeFlipper:edge_flip_volume_change", 1 );
        //if ( m_surf.m_verbose ) { std::cout << "edge flip rejected: volume change = " << vol << std::endl; }
        return false;
    }
    
    // --------------
    
    // Prevent non-manifold surfaces if we're not allowing them
    if ( false == m_surf.m_allow_non_manifold )
    {
        for ( size_t i = 0; i < m_mesh.m_vertex_to_edge_map[ third_vertex_0 ].size(); ++i )
        {
            if ( ( m_mesh.m_edges[ m_mesh.m_vertex_to_edge_map[third_vertex_0][i] ][0] == third_vertex_1 ) ||
                ( m_mesh.m_edges[ m_mesh.m_vertex_to_edge_map[third_vertex_0][i] ][1] == third_vertex_1 ) )
            {
                // edge already exists
                if ( m_surf.m_verbose ) { std::cout << "edge flip rejected: edge exists" << std::endl;             }
                
                g_stats.add_to_int( "EdgeFlipper:edge_flip_would_be_nonmanifold", 1 );
                
                return false;
            }
        }
    }
    
    
    // --------------
    
    // Don't flip edge on a degenerate tet
    if ( third_vertex_0 == third_vertex_1 )
    {
        if ( m_surf.m_verbose ) { std::cout << "edge flip rejected: degenerate tet" << std::endl; }
        g_stats.add_to_int( "EdgeFlipper:edge_flip_on_degenerate_tet", 1 );
        return false;
    }
    
    // --------------
    
    // Create the new triangles
    // new edge winding order == winding order of old triangle0 == winding order of new triangle0
    // then we make the second new triangle match that, and set labels appropriately.
    
    size_t new_triangle_third_vertex_0, new_triangle_third_vertex_1;
    if ( m_mesh.oriented( m_mesh.m_edges[edge][0], m_mesh.m_edges[edge][1], m_mesh.get_triangle(tri0) ) )
    {
        new_triangle_third_vertex_0 = m_mesh.m_edges[edge][1];
        new_triangle_third_vertex_1 = m_mesh.m_edges[edge][0];
    }
    else
    {
        new_triangle_third_vertex_0 = m_mesh.m_edges[edge][0];
        new_triangle_third_vertex_1 = m_mesh.m_edges[edge][1];
    }
    
    
    //the new patch has orientations that match each other, for simplicity.
    Vec3st new_triangle0( new_edge[0], new_edge[1], new_triangle_third_vertex_0 );
    Vec3st new_triangle1( new_edge[1], new_edge[0], new_triangle_third_vertex_1 );
    
    if ( m_surf.m_verbose )
    {
        std::cout << "flip --- new triangle 0: " << new_triangle0 << std::endl;
        std::cout << "flip --- new triangle 1: " << new_triangle1 << std::endl;
    }
    
    // --------------
    
    // if both triangle normals agree before flipping, make sure they agree after flipping
    if(!m_surf.m_aggressive_mode) {
        if ( dot( m_surf.get_triangle_normal(tri0), m_surf.get_triangle_normal(tri1) ) > 0.0 &&
            m_mesh.oriented(m_mesh.m_edges[edge][0], m_mesh.m_edges[edge][1], m_mesh.get_triangle(tri0)) ==
            m_mesh.oriented(m_mesh.m_edges[edge][1], m_mesh.m_edges[edge][0], m_mesh.get_triangle(tri1)) )
        {
            if ( dot( m_surf.get_triangle_normal(new_triangle0), m_surf.get_triangle_normal(new_triangle1) ) < 0.0 )
            {
                if ( m_surf.m_verbose ) { std::cout << "edge flip rejected: normal inversion" << std::endl; }
                g_stats.add_to_int( "EdgeFlipper:edge_flip_normal_inversion", 1 );
                return false;
            }
            
            if ( dot( m_surf.get_triangle_normal(new_triangle0), m_surf.get_triangle_normal(tri0) ) < 0.0 )
            {
                if ( m_surf.m_verbose ) { std::cout << "edge flip rejected: normal inversion" << std::endl; }
                g_stats.add_to_int( "EdgeFlipper:edge_flip_normal_inversion", 1 );
                return false;
            }
            
            if ( dot( m_surf.get_triangle_normal(new_triangle1), m_surf.get_triangle_normal(tri1) ) < 0.0 )
            {
                if ( m_surf.m_verbose ) { std::cout << "edge flip rejected: normal inversion" << std::endl; }
                g_stats.add_to_int( "EdgeFlipper:edge_flip_normal_inversion", 1 );
                return false;
            }
            
            if ( dot( m_surf.get_triangle_normal(new_triangle0), m_surf.get_triangle_normal(tri1) ) < 0.0 )
            {
                if ( m_surf.m_verbose ) { std::cout << "edge flip rejected: normal inversion" << std::endl; }
                g_stats.add_to_int( "EdgeFlipper:edge_flip_normal_inversion", 1 );
                return false;
            }
            
            if ( dot( m_surf.get_triangle_normal(new_triangle1), m_surf.get_triangle_normal(tri0) ) < 0.0 )
            {
                if ( m_surf.m_verbose ) { std::cout << "edge flip rejected: normal inversion" << std::endl; }
                g_stats.add_to_int( "EdgeFlipper:edge_flip_normal_inversion", 1 );
                return false;
            }
        } else if (dot(m_surf.get_triangle_normal(tri0), m_surf.get_triangle_normal(tri1)) < 0.0 &&
                   m_mesh.oriented(m_mesh.m_edges[edge][0], m_mesh.m_edges[edge][1], m_mesh.get_triangle(tri0)) !=
                   m_mesh.oriented(m_mesh.m_edges[edge][1], m_mesh.m_edges[edge][0], m_mesh.get_triangle(tri1)))
        {
            if ( dot( m_surf.get_triangle_normal(new_triangle0), m_surf.get_triangle_normal(new_triangle1) ) < 0.0 )
            {
                if ( m_surf.m_verbose ) { std::cout << "edge flip rejected: normal inversion" << std::endl; }
                g_stats.add_to_int( "EdgeFlipper:edge_flip_normal_inversion", 1 );
                return false;
            }
            
            if ( dot( m_surf.get_triangle_normal(new_triangle0), m_surf.get_triangle_normal(tri0) ) < 0.0 )
            {
                if ( m_surf.m_verbose ) { std::cout << "edge flip rejected: normal inversion" << std::endl; }
                g_stats.add_to_int( "EdgeFlipper:edge_flip_normal_inversion", 1 );
                return false;
            }
            
            if ( dot( m_surf.get_triangle_normal(new_triangle1), m_surf.get_triangle_normal(tri1) ) > 0.0 )
            {
                if ( m_surf.m_verbose ) { std::cout << "edge flip rejected: normal inversion" << std::endl; }
                g_stats.add_to_int( "EdgeFlipper:edge_flip_normal_inversion", 1 );
                return false;
            }
            
            if ( dot( m_surf.get_triangle_normal(new_triangle0), m_surf.get_triangle_normal(tri1) ) > 0.0 )
            {
                if ( m_surf.m_verbose ) { std::cout << "edge flip rejected: normal inversion" << std::endl; }
                g_stats.add_to_int( "EdgeFlipper:edge_flip_normal_inversion", 1 );
                return false;
            }
            
            if ( dot( m_surf.get_triangle_normal(new_triangle1), m_surf.get_triangle_normal(tri0) ) < 0.0 )
            {
                if ( m_surf.m_verbose ) { std::cout << "edge flip rejected: normal inversion" << std::endl; }
                g_stats.add_to_int( "EdgeFlipper:edge_flip_normal_inversion", 1 );
                return false;
            }
        }
    }
    

    
    // --------------
    
    // Prevent degenerate triangles
    Vec3st old_tri0 = m_mesh.get_triangle(tri0);
    Vec3st old_tri1 = m_mesh.get_triangle(tri1);
    Vec3d old_normal0 = cross(xs[old_tri0[1]] - xs[old_tri0[0]], xs[old_tri0[2]] - xs[old_tri0[0]]);
    Vec3d old_normal1 = cross(xs[old_tri1[1]] - xs[old_tri1[0]], xs[old_tri1[2]] - xs[old_tri1[0]]);
    double old_area0 = mag(old_normal0) / 2;
    double old_area1 = mag(old_normal1) / 2;
    normalize(old_normal0);
    normalize(old_normal1);
    Vec3d new_normal0 = cross(xs[new_triangle0[1]] - xs[new_triangle0[0]], xs[new_triangle0[2]] - xs[new_triangle0[0]]);
    Vec3d new_normal1 = cross(xs[new_triangle1[1]] - xs[new_triangle1[0]], xs[new_triangle1[2]] - xs[new_triangle1[0]]);
    double new_area0 = mag(new_normal0) / 2;
    double new_area1 = mag(new_normal1) / 2;
    normalize(new_normal0);
    normalize(new_normal1);
    
    if ( new_area0 < std::min(m_surf.m_min_triangle_area, std::min(old_area0, old_area0) * 0.5) )
    {
        if ( m_surf.m_verbose ) { std::cout << "edge flip rejected: area0 too small" << std::endl;    }
        g_stats.add_to_int( "EdgeFlipper:edge_flip_new_area_too_small", 1 );
        return false;
    }
    
    if ( new_area1 < std::min(m_surf.m_min_triangle_area, std::min(old_area0, old_area0) * 0.5) )
    {
        if ( m_surf.m_verbose ) { std::cout << "edge flip rejected: area1 too small" << std::endl; }
        g_stats.add_to_int( "EdgeFlipper:edge_flip_new_area_too_small", 1 );
        return false;
    }
    
    
    // --------------
    
    // Control change in area
    
    if ( std::fabs( old_area0 + old_area1 - new_area0 - new_area1 ) > 0.1 * (old_area0 + old_area1) )
    {
        if ( m_surf.m_verbose ) {std::cout << "edge flip rejected: area change too great" << std::endl; }
        g_stats.add_to_int( "EdgeFlipper:edge_flip_area_change_too_large", 1 );
        return false;
    }
    
    // --------------
    
    // Don't flip if the quad is not planar enough
    if (m_surf.edge_is_feature(edge))
    {
        if ( m_surf.m_verbose ) {std::cout << "edge flip rejected: edge is a feature" << std::endl;  }
        g_stats.add_to_int( "EdgeFlipper:edge_flip_not_smooth", 1 );
        return false;
    }
    
    // Don't flip if it produces triangles with bad aspect ratio (regardless of area)
    double old_min_edge_0 = std::min(std::min(mag(xs[old_tri0[1]] - xs[old_tri0[0]]), mag(xs[old_tri0[2]] - xs[old_tri0[1]])), mag(xs[old_tri0[0]] - xs[old_tri0[2]]));
    double old_min_edge_1 = std::min(std::min(mag(xs[old_tri1[1]] - xs[old_tri1[0]]), mag(xs[old_tri1[2]] - xs[old_tri1[1]])), mag(xs[old_tri1[0]] - xs[old_tri1[2]]));
    double old_max_edge_0 = std::max(std::max(mag(xs[old_tri0[1]] - xs[old_tri0[0]]), mag(xs[old_tri0[2]] - xs[old_tri0[1]])), mag(xs[old_tri0[0]] - xs[old_tri0[2]]));
    double old_max_edge_1 = std::max(std::max(mag(xs[old_tri1[1]] - xs[old_tri1[0]]), mag(xs[old_tri1[2]] - xs[old_tri1[1]])), mag(xs[old_tri1[0]] - xs[old_tri1[2]]));
    double new_min_edge_0 = std::min(std::min(mag(xs[new_triangle0[1]] - xs[new_triangle0[0]]), mag(xs[new_triangle0[2]] - xs[new_triangle0[1]])), mag(xs[new_triangle0[0]] - xs[new_triangle0[2]]));
    double new_min_edge_1 = std::min(std::min(mag(xs[new_triangle1[1]] - xs[new_triangle1[0]]), mag(xs[new_triangle1[2]] - xs[new_triangle1[1]])), mag(xs[new_triangle1[0]] - xs[new_triangle1[2]]));
    double new_max_edge_0 = std::max(std::max(mag(xs[new_triangle0[1]] - xs[new_triangle0[0]]), mag(xs[new_triangle0[2]] - xs[new_triangle0[1]])), mag(xs[new_triangle0[0]] - xs[new_triangle0[2]]));
    double new_max_edge_1 = std::max(std::max(mag(xs[new_triangle1[1]] - xs[new_triangle1[0]]), mag(xs[new_triangle1[2]] - xs[new_triangle1[1]])), mag(xs[new_triangle1[0]] - xs[new_triangle1[2]]));
    double AR_THRESHOLD = 10.0;
    double arthreshold = AR_THRESHOLD;
    arthreshold = std::max(arthreshold, 1 / (old_area0 * 2 / (old_max_edge_0 * old_max_edge_0)));
    arthreshold = std::max(arthreshold, 1 / (old_area1 * 2 / (old_max_edge_1 * old_max_edge_1)));
    arthreshold = std::max(arthreshold, old_area0 * 2 / (old_min_edge_0 * old_min_edge_0));
    arthreshold = std::max(arthreshold, old_area1 * 2 / (old_min_edge_1 * old_min_edge_1));
    if ((new_area0 * 2 / (new_max_edge_0 * new_max_edge_0) < 1 / arthreshold || new_area0 * 2 / (new_min_edge_0 * new_min_edge_0) > arthreshold) ||
        (new_area1 * 2 / (new_max_edge_1 * new_max_edge_1) < 1 / arthreshold || new_area1 * 2 / (new_min_edge_1 * new_min_edge_1) > arthreshold))
    {
        if ( m_surf.m_verbose ) {std::cout << "edge flip rejected: flip will produce triangles with bad aspect ratio" << std::endl;  }
        g_stats.add_to_int( "EdgeFlipper:edge_flip_bad_triangle", 1 );
        return false;
    }
    
    // --------------
    
    // Don't introduce a large or small angle
    
    double min_angle = min_triangle_angle( xs[new_triangle0[0]], xs[new_triangle0[1]], xs[new_triangle0[2]] );
    min_angle = min( min_angle, min_triangle_angle( xs[new_triangle1[0]], xs[new_triangle1[1]], xs[new_triangle1[2]] ) );
    
    if ( rad2deg(min_angle) < m_surf.m_min_triangle_angle )
    {
        g_stats.add_to_int( "EdgeFlipper:edge_flip_bad_angle", 1 );
        return false;
    }
    
    double max_angle = max_triangle_angle( xs[new_triangle0[0]], xs[new_triangle0[1]], xs[new_triangle0[2]] );
    max_angle = max( max_angle, max_triangle_angle( xs[new_triangle1[0]], xs[new_triangle1[1]], xs[new_triangle1[2]] ) );
    
    if ( rad2deg(max_angle) > m_surf.m_max_triangle_angle )
    {
        g_stats.add_to_int( "EdgeFlipper:edge_flip_bad_angle", 1 );
        return false;
    }

    // --------------

    // Prevent intersection
    if (m_surf.m_collision_safety && flip_introduces_collision(edge, new_edge, new_triangle0, new_triangle1))
    {
        if (m_surf.m_verbose) { std::cout << "edge flip rejected: intersection" << std::endl; }

        g_stats.add_to_int("EdgeFlipper:edge_flip_collision", 1);
        return false;
    }
    // --------------
    
    // Okay, now do the actual operation
    
    void * data = NULL;
    if (m_surf.m_mesheventcallback)
        m_surf.m_mesheventcallback->pre_flip(m_surf, edge, &data);

    // Start history log
    MeshUpdateEvent flip(MeshUpdateEvent::EDGE_FLIP);
    flip.m_v0 = edge_vertices[0];
    flip.m_v1 = edge_vertices[1];
    
    Vec2i old_label_0 = m_surf.m_mesh.get_triangle_label(tri0);
    Vec2i new_label_0 = old_label_0;
    
    // new_triangle1 has consistent triangle orientation as new_triangle0,
    // by construction so just set labels to match.
    Vec2i new_label_1;
    new_label_1 = old_label_0;
    
    assert(m_mesh.oriented(new_edge[0], new_edge[1], new_triangle0) !=
           m_mesh.oriented(new_edge[0], new_edge[1], new_triangle1));
    
    m_surf.remove_triangle( tri0 );
    m_surf.remove_triangle( tri1 );
    flip.m_deleted_tris.push_back(tri0);
    flip.m_deleted_tris.push_back(tri1);
    
    // the old label carries over to the new triangle
    size_t new_triangle_index_0 = m_surf.add_triangle( new_triangle0, new_label_0 );
    size_t new_triangle_index_1 = m_surf.add_triangle( new_triangle1, new_label_1 );
    flip.m_created_tris.push_back(new_triangle_index_0);
    flip.m_created_tris.push_back(new_triangle_index_1);
    flip.m_created_tri_data.push_back(new_triangle0);
    flip.m_created_tri_data.push_back(new_triangle1);
    
    flip.m_created_tri_labels.push_back(new_label_0);
    flip.m_created_tri_labels.push_back(new_label_1);
    
    ////////////////////////////////////////////////////////////
    
    if ( m_surf.m_collision_safety )
    {
        if ( m_surf.check_triangle_vs_all_triangles_for_intersection( new_triangle_index_0 ) )
        {
            std::cout << "missed an intersection.  New triangles: " << new_triangle0 << ", " << new_triangle1 << std::endl;
            std::cout << "old triangles: " << old_tri0 << ", " << old_tri1 << std::endl;
            assert(0);
        }
        
        if ( m_surf.check_triangle_vs_all_triangles_for_intersection( new_triangle_index_1 ) )
        {
            std::cout << "missed an intersection.  New triangles: " << new_triangle0 << ", " << new_triangle1 << std::endl;
            std::cout << "old triangles: " << old_tri0 << ", " << old_tri1 << std::endl;
            assert(0);
        }
    }
    
    m_surf.m_dirty_triangles.push_back( new_triangle_index_0 );
    m_surf.m_dirty_triangles.push_back( new_triangle_index_1 );
    
    if ( m_surf.m_verbose ) { std::cout << "edge flip: ok" << std::endl; }
    
    g_stats.add_to_int( "EdgeFlipper:edge_flip_success", 1 );
    
    m_surf.m_mesh_change_history.push_back(flip);
    
    m_surf.trim_degeneracies( m_surf.m_dirty_triangles );

    if (m_surf.m_mesheventcallback)
        m_surf.m_mesheventcallback->post_flip(m_surf, edge, data);
    
    return true;
    
}

void EdgeFlipper::getQuadric(size_t vertex, Mat33d& A) {
    
    const NonDestructiveTriMesh& mesh = m_surf.m_mesh;
    const std::vector<size_t>& incident_triangles = mesh.m_vertex_to_triangle_map[vertex];
    
    std::vector< Vec3d > N;
    std::vector< double > W;
    
    for ( size_t i = 0; i < incident_triangles.size(); ++i )
    {
        size_t triangle_index = incident_triangles[i];
        N.push_back( m_surf.get_triangle_normal(triangle_index) );
        W.push_back( m_surf.get_triangle_area(triangle_index) );
    }
    
    zero(A);
    
    // Compute A = N^T W N
    // where N are normals and W are area weights.
    for ( size_t i = 0; i < N.size(); ++i )
    {
        A(0,0) += N[i][0] * W[i] * N[i][0];
        A(1,0) += N[i][1] * W[i] * N[i][0];
        A(2,0) += N[i][2] * W[i] * N[i][0];
        
        A(0,1) += N[i][0] * W[i] * N[i][1];
        A(1,1) += N[i][1] * W[i] * N[i][1];
        A(2,1) += N[i][2] * W[i] * N[i][1];
        
        A(0,2) += N[i][0] * W[i] * N[i][2];
        A(1,2) += N[i][1] * W[i] * N[i][2];
        A(2,2) += N[i][2] * W[i] * N[i][2];
    }
}

bool EdgeFlipper::is_Delaunay_anisotropic( size_t edge, size_t tri0, size_t tri1, size_t third_vertex_0, size_t third_vertex_1 )
{
    //std::cout << "Checking Delaunay criteria.\n";
    // per Jiao et al. 2010, "4.3 Anisotropic edge flipping"
    // section 4.3, Anisotropic edge flipping
    
    //compute the quadric tensors at the four vertices
    Mat33d mat0, mat1, mat2, mat3;
    getQuadric(m_surf.m_mesh.m_edges[edge][0], mat0);
    getQuadric(m_surf.m_mesh.m_edges[edge][1], mat1);
    getQuadric(third_vertex_0, mat2);
    getQuadric(third_vertex_1, mat3);
    
    //sum them to get a combined quadric tensor A for the patch
    Mat33d A = mat0 + mat1 + mat2 + mat3;
    
    //perform eigen decomposition to determine the eigen vectors/values
    double eigenvalues[3];
    double work[9];
    int info = ~0, n = 3, lwork = 9;
    LAPACK::get_eigen_decomposition( &n, A.a, &n, eigenvalues, work, &lwork, &info );
    
    //Note that this returns the eigenvalues in ascending order, as opposed to descending order described by Jiao et al.
    
    //compute the metric tensor M_Q, with Q = Identity
    Mat22d M( eigenvalues[1] / eigenvalues[2], 0, 0, eigenvalues[0] / eigenvalues[2]);
    
    //clamp the eigenvalues for safety
    M(0,0) = clamp(M(0,0), 0.005, 0.07);
    M(1,1) = clamp(M(1,1), 0.005, 0.07);
    
    //convert the relevant vertices from Cartesian coords into the coordinate system of the local frame (defined by the eigenvectors)
    Mat33d conversion_matrix;
    conversion_matrix = A; //note that A is now holding the eigenvectors, not the original quadric tensor.
    
    conversion_matrix = inverse(conversion_matrix);
    Vec3d v0 = m_surf.get_position(m_surf.m_mesh.m_edges[edge][0]);
    Vec3d v1 = m_surf.get_position(m_surf.m_mesh.m_edges[edge][1]);
    Vec3d v2 = m_surf.get_position(third_vertex_0);
    Vec3d v3 = m_surf.get_position(third_vertex_1);
    
    Vec3d nv0 = conversion_matrix*v0;
    Vec3d nv1 = conversion_matrix*v1;
    Vec3d nv2 = conversion_matrix*v2;
    Vec3d nv3 = conversion_matrix*v3;
    
    //only bother looking at the 2D coordinates in the tangent plane
    Vec2d n0_2d(nv0[0], nv0[1]);
    Vec2d n1_2d(nv1[0], nv1[1]);
    Vec2d n2_2d(nv2[0], nv2[1]);
    Vec2d n3_2d(nv3[0], nv3[1]);
    
    
    ////warp the vertices into the normalized space to account for anisotropy, via M
    n0_2d = M * n0_2d;
    n1_2d = M * n1_2d;
    n2_2d = M * n2_2d;
    n3_2d = M * n3_2d;
    
    //check the Delaunay criterion (sum of opposite angles < 180) in the modified space
    Vec2d off0 = n0_2d - n2_2d, off1 = n1_2d - n2_2d;
    double angle0 = acos(dot(off0,off1) / mag(off0) / mag(off1));
    
    Vec2d off2 = n0_2d - n3_2d, off3 = n1_2d - n3_2d;
    double angle1 = acos(dot(off2, off3) / mag(off2) / mag(off3));
    
    return angle0 + angle1 < M_PI;
    
}


// --------------------------------------------------------
///
/// Flip all non-delaunay edges
///
// --------------------------------------------------------

bool EdgeFlipper::flip_pass( )
{
    
    if ( m_surf.m_verbose )
    {
        std::cout << "---------------------- EdgeFlipper: flipping ----------------------" << std::endl;
    }
    
    //   if ( m_surf.m_collision_safety )
    //   {
    //      m_surf.check_continuous_broad_phase_is_up_to_date();
    //   }
    
    m_surf.m_dirty_triangles.clear();
    
    bool flip_occurred_ever = false;          // A flip occurred in this function call
    bool flip_occurred = true;                // A flip occurred in the current loop iteration
    
    static unsigned int MAX_NUM_FLIP_PASSES = 5;
    unsigned int num_flip_passes = 0;
    
    NonDestructiveTriMesh& m_mesh = m_surf.m_mesh;
    const std::vector<Vec3d>& xs = m_surf.get_positions();
    
    //
    // Each "pass" is once over the entire set of edges (ignoring edges created during the current pass)
    //
    
    while ( flip_occurred && num_flip_passes++ < MAX_NUM_FLIP_PASSES )
    {
        if ( m_surf.m_verbose )
        {
            std::cout << "---------------------- Los Topos: flipping ";
            std::cout << "pass " << num_flip_passes << "/" << MAX_NUM_FLIP_PASSES;
            std::cout << "----------------------" << std::endl;
        }
        
        flip_occurred = false;
        
        size_t number_of_edges = m_mesh.m_edges.size();      // don't work on newly created edges
        
        for( size_t i = 0; i < number_of_edges; i++ )
        {
            if ( m_mesh.m_edges[i][0] == m_mesh.m_edges[i][1] )   { continue; }
            if ( m_mesh.m_edge_to_triangle_map[i].size() > 4 || m_mesh.m_edge_to_triangle_map[i].size() < 2 )   { continue; }
            
            if ( m_mesh.m_edge_to_triangle_map[i].size() == 3) continue; //don't try flipping Y-junction non-manifold edges.
            
            //if ( m_mesh.m_is_boundary_vertex[ m_mesh.m_edges[i][0] ] || m_mesh.m_is_boundary_vertex[ m_mesh.m_edges[i][1] ] )  { continue; }  // skip boundary vertices
            //NOTE: This check disables flipping on edges where either endpoint is on the boundary.
            //Future work: extend to the cases where meshes have open boundaries
            
            size_t triangle_a = (size_t)~0, triangle_b =(size_t)~0;
            
            if ( m_mesh.m_edge_to_triangle_map[i].size() == 2 )
            {
                triangle_a = m_mesh.m_edge_to_triangle_map[i][0];
                triangle_b = m_mesh.m_edge_to_triangle_map[i][1];
            }
            else if ( m_mesh.m_edge_to_triangle_map[i].size() == 4 )
            {
                // non manifold edge: disable flipping
                continue;
                
                triangle_a = m_mesh.m_edge_to_triangle_map[i][0];
                
                // Find first triangle with orientation opposite triangle_a's orientation
                unsigned int j = 1;
                for ( ; j < 4; ++j )
                {
                    triangle_b = m_mesh.m_edge_to_triangle_map[i][j];
                    if (    m_mesh.oriented( m_mesh.m_edges[i][0], m_mesh.m_edges[i][1], m_mesh.get_triangle(triangle_a) )
                        != m_mesh.oriented( m_mesh.m_edges[i][0], m_mesh.m_edges[i][1], m_mesh.get_triangle(triangle_b) ) )
                    {
                        break;
                    }
                }
                assert ( j < 4 );
            }
            else
            {
                std::cout << m_mesh.m_edge_to_triangle_map[i].size() << " triangles incident to an edge" << std::endl;
                assert(0);
            }
            
            //don't flip if one of the faces is all solid
            if(m_surf.triangle_is_all_solid(triangle_a) || m_surf.triangle_is_all_solid(triangle_b)) {
                continue;
            }
            
            // Don't flip edge on a degenerate/deleted triangles
            if ( m_mesh.triangle_is_deleted(triangle_a) || m_mesh.triangle_is_deleted(triangle_b) )
                continue;
            
            const Vec3st& tri_a = m_mesh.get_triangle( triangle_a );
            const Vec3st& tri_b = m_mesh.get_triangle( triangle_b );
            
            size_t third_vertex_0 = m_mesh.get_third_vertex( m_mesh.m_edges[i][0], m_mesh.m_edges[i][1], tri_a );
            size_t third_vertex_1 = m_mesh.get_third_vertex( m_mesh.m_edges[i][0], m_mesh.m_edges[i][1], tri_b );
            
            if ( third_vertex_0 == third_vertex_1 ) {
                continue;
            }
            
            size_t vert_0 = m_mesh.m_edges[i][0];
            size_t vert_1 = m_mesh.m_edges[i][1];
            
            bool flipped = false;
            
            bool flip_required = false;
            
            if(m_use_Delaunay_criterion) {
                
                //compute the angles that oppose the edge
                Vec3d pos_3rd_0 = m_surf.get_position(third_vertex_0);
                Vec3d pos_3rd_1 = m_surf.get_position(third_vertex_1);
                Vec3d pos_vert_0 = m_surf.get_position(vert_0);
                Vec3d pos_vert_1 = m_surf.get_position(vert_1);
                
                Vec3d off0 = pos_vert_0 - pos_3rd_0;
                Vec3d off1 = pos_vert_1 - pos_3rd_0;
                double m0 = mag(off0), m1 = mag(off1);
                if(m0 == 0 || m1 == 0) continue;
                double angle0 = acos( dot(off0,off1) / (m0*m1) );
                
                Vec3d off2 = pos_vert_0 - pos_3rd_1;
                Vec3d off3 = pos_vert_1 - pos_3rd_1;
                double m2 = mag(off2), m3 = mag(off3);
                if(m2 == 0 || m3 == 0) continue;
                double angle1 = acos( dot(off2, off3) / (m2*m3) );
                
                if(m_surf.m_aggressive_mode) {
                    //skip any triangles that don't have fairly bad angles.
                    Vec2d angles_cos0, angles_cos1;
                    min_and_max_triangle_angle_cosines(pos_vert_0, pos_vert_1, pos_3rd_0, angles_cos0);
                    min_and_max_triangle_angle_cosines(pos_vert_0, pos_vert_1, pos_3rd_1, angles_cos1);
                    double min_cos, max_cos;
                    min_cos = min(angles_cos0[0], angles_cos1[0]);
                    max_cos = max(angles_cos0[1], angles_cos1[1]);

                    bool angles_are_fine_cos = min_cos > m_surf.m_min_angle_cosine || max_cos < m_surf.m_max_angle_cosine;

                    if (angles_are_fine_cos)
                        continue;
                }
                
                //if the sum of the opposing angles exceeds 180, then we should flip (according to the Delaunay criterion)
                //Delaunay apparently maximizes the minimum angle in the triangulation
                flip_required = angle0 + angle1 > M_PI;
            }
            else {
                //Flip based on valences instead.
                //per e.g. "A Remeshing Approach to Multiresolution Modeling"
                
                //Here we treat non-manifold vertices as being on boundaries, and boundaries as boundaries.
                //so their optimal valence is 4 instead of 6.
                //See e.g. https://code.google.com/p/stacker/source/browse/trunk/GraphicsLibrary/Remeshing/LaplacianRemesher.h
                
                
                int opt_val_a = m_mesh.is_vertex_nonmanifold(third_vertex_0)?4:(m_mesh.m_is_boundary_vertex[third_vertex_0]?4:6), 
                opt_val_b = m_mesh.is_vertex_nonmanifold(third_vertex_1)?4:(m_mesh.m_is_boundary_vertex[third_vertex_1]?4:6),
                opt_val_0 = m_mesh.is_vertex_nonmanifold(vert_0)?4:(m_mesh.m_is_boundary_vertex[vert_0]?4:6), 
                opt_val_1 = m_mesh.is_vertex_nonmanifold(vert_1)?4:(m_mesh.m_is_boundary_vertex[vert_1]?4:6);
                
                int val_a, val_b, val_0, val_1;
                Vec2i region_pair = m_mesh.get_triangle_label(triangle_a); //doesn't matter which triangle we consider.
                val_0 = edge_count_bordering_region_pair(vert_0, region_pair);
                val_1 = edge_count_bordering_region_pair(vert_1, region_pair);
                val_a = edge_count_bordering_region_pair(third_vertex_0, region_pair);
                val_b = edge_count_bordering_region_pair(third_vertex_1, region_pair);
                
                int score_before = sqr(val_a-opt_val_a) + sqr(val_b-opt_val_b)  + sqr(val_0-opt_val_0) + sqr(val_1-opt_val_1);
                
                //now work out the valences after
                val_a++; val_b++;
                val_0--; val_1--;
                
                int score_after = sqr(val_a-opt_val_a) + sqr(val_b-opt_val_b)  + sqr(val_0-opt_val_0) + sqr(val_1-opt_val_1);
                
                flip_required = score_before > score_after;
                
                double current_length = mag( xs[m_mesh.m_edges[i][1]] - xs[m_mesh.m_edges[i][0]] );        
                double potential_length = mag( xs[third_vertex_1] - xs[third_vertex_0] );     
            }
            
            if ( flip_required )
            {
                flipped = flip_edge( i, triangle_a, triangle_b, third_vertex_0, third_vertex_1 );            
            }
            
            flip_occurred |= flipped;            
        }
        
        flip_occurred_ever |= flip_occurred;
    }
    
    
    return flip_occurred_ever;
    
}

int EdgeFlipper::edge_count_bordering_region_pair(size_t vertex, Vec2i region_pair) {
    int count = 0;
    
    Vec2i flipped_pair(region_pair[1], region_pair[0]);
    
    //consider all incident edges
    for(size_t i = 0; i < m_surf.m_mesh.m_vertex_to_edge_map[vertex].size(); ++i) {
        size_t edge = m_surf.m_mesh.m_vertex_to_edge_map[vertex][i];
        
        //consider all triangles on the edge
        for(size_t j = 0; j < m_surf.m_mesh.m_edge_to_triangle_map[edge].size(); ++j) {
            size_t tri = m_surf.m_mesh.m_edge_to_triangle_map[edge][j];
            const Vec2i& labels = m_surf.m_mesh.m_triangle_labels[tri];
            //if one of the triangles matches the requested manifold region (label pair), we're done.
            if(labels == region_pair || labels == flipped_pair) {
                ++count;
                break;
            }
        }
        
    }
    return count;
}
    
}
