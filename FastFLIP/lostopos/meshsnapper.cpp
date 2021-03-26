// ---------------------------------------------------------
//
//  meshsnapper.cpp
//  Christopher Batty, Fang Da 2014
//
//  Functions supporting the "vertex snap" operation: merging nearby vertices.
//  The opposite of a pinch operation. This was adapted from the edge collapse code.
//
// ---------------------------------------------------------

#include <meshsnapper.h>

#include <broadphase.h>
#include <collisionpipeline.h>
#include <collisionqueries.h>
#include <nondestructivetrimesh.h>
#include <runstats.h>
#include <subdivisionscheme.h>
#include <surftrack.h>
#include <trianglequality.h>
#include <edgesplitter.h>
#include <facesplitter.h>

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
/// Mesh Snapper constructor.  Takes a SurfTrack object.
///
// --------------------------------------------------------

MeshSnapper::MeshSnapper( SurfTrack& surf, bool use_curvature, bool remesh_boundaries, double max_curvature_multiplier ) :
m_surf( surf ),
m_edge_threshold(0.12),
m_face_threshold(0.1),
m_facesplitter(surf),
m_edgesplitter(surf, use_curvature, remesh_boundaries, max_curvature_multiplier)
{}


// --------------------------------------------------------
///
/// Get all triangles which are incident on either vertex.
///
// --------------------------------------------------------

void MeshSnapper::get_moving_triangles(size_t source_vertex,
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
/// Get all edges which are incident on either vertex.
///
// --------------------------------------------------------

void MeshSnapper::get_moving_edges( size_t source_vertex,
                                   size_t destination_vertex,
                                   std::vector<size_t>& moving_edges )
{
    
    moving_edges = m_surf.m_mesh.m_vertex_to_edge_map[ source_vertex ];
    moving_edges.insert( moving_edges.end(), m_surf.m_mesh.m_vertex_to_edge_map[ destination_vertex ].begin(), m_surf.m_mesh.m_vertex_to_edge_map[ destination_vertex ].end() );
    
}


// --------------------------------------------------------
///
/// Determine if the edge collapse operation would invert the normal of any incident triangles.
///
// --------------------------------------------------------

bool MeshSnapper::snap_introduces_normal_inversion( size_t source_vertex,
                                                   size_t destination_vertex,
                                                   const Vec3d& vertex_new_position )
{
    
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
    
    //
    // check for normal inversion
    //
    
    for ( size_t i = 0; i < moving_triangles.size(); ++i )
    {
        
        const Vec3st& current_triangle = m_surf.m_mesh.get_triangle( moving_triangles[i] );
        Vec3d old_normal = m_surf.get_triangle_normal( current_triangle );
        
        Vec3d new_normal;
        
        //  the new triangle always has the same orientation as the
        //  old triangle so no change is needed
        
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
            if ( m_surf.m_verbose ) { std::cout << "collapse edge introduces normal inversion" << std::endl; }
            
            g_stats.add_to_int( "MeshSnapper:collapse_normal_inversion", 1 );
            
            return true;
        }
        
        if ( new_area < m_surf.m_min_triangle_area && new_area < mag2(m_surf.get_position(source_vertex) - m_surf.get_position(destination_vertex)) * 0.1 )
        {
            if ( m_surf.m_verbose ) { std::cout << "collapse edge introduces tiny triangle area" << std::endl; }
            
            g_stats.add_to_int( "MeshSnapper:collapse_degenerate_triangle", 1 );
            
            return true;
        }
        
    }
    
    return false;
    
}



// --------------------------------------------------------
///
/// Check the "pseudo motion" introduced by snapping vertices for collision
///
// --------------------------------------------------------

bool MeshSnapper::snap_pseudo_motion_introduces_collision( size_t source_vertex,
                                                          size_t destination_vertex,
                                                          const Vec3d& )
{
    assert( m_surf.m_collision_safety );
    
    // Get the set of triangles which move because of this motion
    std::vector<size_t> moving_triangles;
    get_moving_triangles( source_vertex, destination_vertex, moving_triangles );
    
    // And the set of edges
    std::vector<size_t> moving_edges;
    get_moving_edges( source_vertex, destination_vertex, moving_edges );
    
    
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
    
    return false;
    
}

// --------------------------------------------------------
///
/// Snap a pair of vertices by moving the source vertex to the destination vertex
///
// --------------------------------------------------------

bool MeshSnapper::snap_vertex_pair( size_t vertex_to_keep, size_t vertex_to_delete)
{
    
    assert(m_surf.m_allow_non_manifold);
    
    bool keep_vert_is_boundary = m_surf.m_mesh.m_is_boundary_vertex[vertex_to_keep];
    bool del_vert_is_boundary = m_surf.m_mesh.m_is_boundary_vertex[vertex_to_delete];
    
    //TODO Figure out how to deal with boundaries here.
    
    if ( m_surf.m_verbose ) {
        std::cout << "Snapping vertices.  Doomed vertex: " << vertex_to_delete
        << " --- Vertex to keep: " << vertex_to_keep << std::endl;
    }
    
//    // --------------
//    // decide on new vertex position: just take the mass-weighted average of the two points to be snapped, so that constraints marked by infinite masses are respected.
//    
//    Vec3d m0 = m_surf.m_masses[vertex_to_delete];
//    Vec3d m1 = m_surf.m_masses[vertex_to_keep];
//    
//    Vec3d vertex_new_position;
//    for (int i = 0; i < 3; i++)
//    {
//        if (m0[i] == DynamicSurface::solid_mass() && m1[i] == DynamicSurface::solid_mass())
//        {
//            // both vertices are constrained in this direction: this requires the two cooresponding coordindates are equal
//            assert(m_surf.get_position(vertex_to_delete)[i] == m_surf.get_position(vertex_to_keep)[i]);
//            vertex_new_position[i] = m_surf.get_position(vertex_to_delete)[i];
//        } else if (m0[i] == DynamicSurface::solid_mass())
//        {
//            vertex_new_position[i] = m_surf.get_position(vertex_to_delete)[i];
//        } else if (m1[i] == DynamicSurface::solid_mass())
//        {
//            vertex_new_position[i] = m_surf.get_position(vertex_to_keep)[i];
//        } else
//        {
//            vertex_new_position[i] = 0.5*(m_surf.get_position(vertex_to_delete)[i] + m_surf.get_position(vertex_to_keep)[i]);
//        }
//    }
    
    Vec3d vertex_new_position = 0.5 * (m_surf.get_position(vertex_to_delete) + m_surf.get_position((vertex_to_keep)));
    Vec3c vertex_new_solid_label(0, 0, 0);
    
    bool keep_vert_is_any_solid = m_surf.vertex_is_any_solid(vertex_to_keep);
    bool delete_vert_is_any_solid = m_surf.vertex_is_any_solid(vertex_to_delete);
    if (keep_vert_is_any_solid || delete_vert_is_any_solid)
    {
        assert(m_surf.m_solid_vertices_callback);
        
        if (!keep_vert_is_any_solid)
            std::swap(vertex_to_keep, vertex_to_delete);
        
        Vec3d newpos = vertex_new_position;
        if (!m_surf.m_solid_vertices_callback->generate_snapped_position(m_surf, vertex_to_keep, vertex_to_delete, newpos))
        {
            // the callback decides this vertex pair should not be snapped
            if (m_surf.m_verbose)
                std::cout << "Constraint callback vetoed collapsing." << std::endl;
            return false;
        }
        
        vertex_new_position = newpos;
        
        vertex_new_solid_label = m_surf.m_solid_vertices_callback->generate_snapped_solid_label(m_surf, vertex_to_keep, vertex_to_delete, m_surf.vertex_is_solid_3(vertex_to_keep), m_surf.vertex_is_solid_3(vertex_to_delete));
    }
    
    if ( m_surf.m_verbose ) {
        std::cout << "Snapping vertex pair.  Doomed vertex: " << vertex_to_delete
        << " --- Vertex to keep: " << vertex_to_keep << std::endl;
    }
    
    // --------------
    
    // Check vertex pseudo motion for collisions
    
    if ( mag ( m_surf.get_position(vertex_to_delete) - m_surf.get_position(vertex_to_keep) ) > 0 )
    {
        
        // Change source vertex predicted position to superimpose onto destination vertex
        m_surf.set_newposition( vertex_to_keep, vertex_new_position );
        m_surf.set_newposition( vertex_to_delete, vertex_new_position );
        
        bool normal_inversion = snap_introduces_normal_inversion(  vertex_to_delete, vertex_to_keep, vertex_new_position );
        
        if ( normal_inversion )
        {
            // Restore saved positions which were changed by the function we just called.
            m_surf.set_newposition( vertex_to_keep, m_surf.get_position(vertex_to_keep) );
            m_surf.set_newposition( vertex_to_delete, m_surf.get_position(vertex_to_delete) );
            
            if ( m_surf.m_verbose ) { std::cout << "normal_inversion" << std::endl; }
            return false;
        }
        
        bool collision = false;
        
        if ( m_surf.m_collision_safety )
        {
            collision = snap_pseudo_motion_introduces_collision( vertex_to_delete, vertex_to_keep, vertex_new_position );
        }
        
        if ( collision )
        {
            if ( m_surf.m_verbose ) { std::cout << "collision" << std::endl; }
            g_stats.add_to_int( "MeshSnapper:collapse_collisions", 1 );
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
    
//    if (!m_surf.vertex_is_any_solid(vertex_to_keep) && m_surf.vertex_is_any_solid(vertex_to_delete))
//        std::swap(vertex_to_delete, vertex_to_keep);
    
    
    // --------------
    // all clear, now perform the snap
    
    void * data = NULL;
    if (m_surf.m_mesheventcallback)
        m_surf.m_mesheventcallback->pre_snap(m_surf, vertex_to_delete, vertex_to_keep, &data);

    // start building history data
    MeshUpdateEvent collapse(MeshUpdateEvent::SNAP);
    collapse.m_vert_position = vertex_new_position;
    collapse.m_v0 = vertex_to_keep;
    collapse.m_v1 = vertex_to_delete;
    
    // move the vertex we decided to keep
    
    m_surf.set_position( vertex_to_keep, vertex_new_position );
    m_surf.set_newposition( vertex_to_keep, vertex_new_position );

    for (int i = 0; i < 3; i++)
    {
        if (vertex_new_solid_label[i])
            m_surf.m_masses[vertex_to_keep][i] = DynamicSurface::solid_mass();
        else
            m_surf.m_masses[vertex_to_keep][i] = 1;
    }
    
    // Find anything pointing to the doomed vertex and change it
    
    // copy the list of triangles, don't take a reference to it
    std::vector< size_t > triangles_incident_to_vertex = m_surf.m_mesh.m_vertex_to_triangle_map[vertex_to_delete];
    
    for ( size_t i=0; i < triangles_incident_to_vertex.size(); ++i )
    {
        //don't bother copying over dead tris. (Can this ever happen?)
        if(m_surf.m_mesh.triangle_is_deleted(triangles_incident_to_vertex[i])) continue;
        
        Vec3st new_triangle = m_surf.m_mesh.get_triangle( triangles_incident_to_vertex[i] );
        
        if ( new_triangle[0] == vertex_to_delete )   { new_triangle[0] = vertex_to_keep; }
        if ( new_triangle[1] == vertex_to_delete )   { new_triangle[1] = vertex_to_keep; }
        if ( new_triangle[2] == vertex_to_delete )   { new_triangle[2] = vertex_to_keep; }
        
        if ( m_surf.m_verbose ) { std::cout << "adding updated triangle: " << new_triangle << std::endl; }
        
        // the old label carries over to the new triangle.
        // no need to test for orientation because the new triangle
        // generation code above does not change orientation.
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
    
    // update the remeshing velocity
    m_surf.pm_velocities[vertex_to_keep] = (m_surf.pm_velocities[vertex_to_keep] + m_surf.pm_velocities[vertex_to_delete]) / 2;

    // update target edge lengths
    m_surf.m_target_edge_lengths[vertex_to_keep] = std::min(m_surf.m_target_edge_lengths[vertex_to_keep], m_surf.m_target_edge_lengths[vertex_to_delete]);
    
    // Store the history
    m_surf.m_mesh_change_history.push_back(collapse);
    
    // clean up degenerate triangles and tets
    m_surf.trim_degeneracies( m_surf.m_dirty_triangles );

    if (m_surf.m_mesheventcallback)
        m_surf.m_mesheventcallback->post_snap(m_surf, vertex_to_keep, vertex_to_delete, data);
    
    return true;
}


// --------------------------------------------------------
///
/// Snap an edge-edge pair by splitting each, and snapping the vertices
///
// --------------------------------------------------------

bool MeshSnapper::snap_edge_pair( size_t edge0, size_t edge1)
{
    
    assert(m_surf.m_allow_non_manifold);
    
    //TODO Figure out how to deal with boundaries here.
    //TODO Check relative velocities or CCD test.
    
    double distance, s0, s2;
    Vec3d normal;
    const Vec2st& edge_data0 = m_surf.m_mesh.m_edges[edge0];
    const Vec2st& edge_data1 = m_surf.m_mesh.m_edges[edge1];
    
    const Vec3d& v0 = m_surf.get_position(edge_data0[0]);
    const Vec3d& v1 = m_surf.get_position(edge_data0[1]);
    const Vec3d& v2 = m_surf.get_position(edge_data1[0]);
    const Vec3d& v3 = m_surf.get_position(edge_data1[1]);
    
    check_edge_edge_proximity( v0, v1, v2, v3,
                              distance, s0, s2, normal );
    
    //compute the resulting closest points
    Vec3d midpoint0 = s0*v0 + (1-s0)*v1;
    Vec3d midpoint1 = s2*v2 + (1-s2)*v3;
    
    //Check if we're fairly close to an end vertex; if so just use the vertex directly for snapping.
    //otherwise, do a split to create a new point which will then be snapped.
    
    if (m_surf.m_mesheventcallback) {
        m_surf.m_mesheventcallback->log() << "ee snap: edge0 = " << edge0 << ": " << edge_data0[0] << " (" << v0 << ") -> " << edge_data0[1] << " (" << v1 << ") edge1 = " << edge1 << ": " << edge_data1[0] << " (" << v2 << ") -> " << edge_data1[1] << " (" << v3 << ")" << std::endl;
        m_surf.m_mesheventcallback->log() << "s0 = " << s0 << " s2 = " << s2 << " dist = " << distance << " midpoint0 = " << midpoint0 << " midpoint1 = " << midpoint1 << std::endl;
    }

    size_t snapping_vert0 = m_surf.m_mesh.nv();
    size_t snapping_vert1 = m_surf.m_mesh.nv();

    // droplets:
    // special handling to preserve the triple junction: all-solid edge vs non-all-solid edge, in which case the two edges are on two sides of the triple junction and snapping them is a bad idea
    // unlike with vf snapping, we don't even need to check if snapping can happen on endpoints which are on the triple junction; that case should be more easily handled by collapsing the triple junction edges.
    bool v0s = m_surf.vertex_is_any_solid(edge_data0[0]);
    bool v1s = m_surf.vertex_is_any_solid(edge_data0[1]);
    bool v2s = m_surf.vertex_is_any_solid(edge_data1[0]);
    bool v3s = m_surf.vertex_is_any_solid(edge_data1[1]);
    if ((((v0s && v1s) && (!v2s || !v3s)) ||
         ((v2s && v3s) && (!v0s || !v1s))) && distance > 0.01 * m_surf.m_merge_proximity_epsilon)   // added a distance criteria, so that extremely close proximities will still be resolved regardless of the triple junction. Not resolving these will be dangerous because near coincidence meshes won't be effectively separated by the dynamics.
    {
        // we can't allow this case to go through. snapping in the middle of a non-triple-junction edge or to a non-triple-junction end point can perturb the triple junction geometry
        if (m_surf.m_mesheventcallback)
            m_surf.m_mesheventcallback->log() << "canceling snapping to preserve triple junction" << std::endl;
        return false;

    } else  // the general case, which is the original code
    {

    if(s0 > 1 - m_edge_threshold) {
        snapping_vert0 = edge_data0[0];
        if (m_surf.m_mesheventcallback)
            m_surf.m_mesheventcallback->log() << "snap to v0" << std::endl;
    }
    else if(s0 < m_edge_threshold) {
        snapping_vert0 = edge_data0[1];
        if (m_surf.m_mesheventcallback)
            m_surf.m_mesheventcallback->log() << "snap to v1" << std::endl;
    }
    else {
        size_t split_result;
        if (m_surf.m_mesheventcallback)
            m_surf.m_mesheventcallback->log() << "attemping to split edge0 at " << midpoint0 << std::endl;
        
        if(!m_edgesplitter.edge_is_splittable(edge0) || !m_edgesplitter.split_edge(edge0, split_result, false, true, &midpoint0, std::vector<size_t>(), true))
            return false;
        snapping_vert0 = split_result;
    }
    
    if(s2 > 1 - m_edge_threshold) {
        snapping_vert1 = edge_data1[0];
        if (m_surf.m_mesheventcallback)
            m_surf.m_mesheventcallback->log() << "snap to v2" << std::endl;
    }
    else if(s2 < m_edge_threshold) {
        snapping_vert1 = edge_data1[1];
        if (m_surf.m_mesheventcallback)
            m_surf.m_mesheventcallback->log() << "snap to v3" << std::endl;
    }
    else {
        size_t split_result;
        if (m_surf.m_mesheventcallback)
            m_surf.m_mesheventcallback->log() << "attempting to split edge1 at " << midpoint1 << std::endl;
        
        if(!m_edgesplitter.edge_is_splittable(edge1) || !m_edgesplitter.split_edge(edge1, split_result, false, true, &midpoint1, std::vector<size_t>(1, snapping_vert0), true))
            return false;
        snapping_vert1 = split_result;
    }
        
    }
    
    if (m_surf.m_mesheventcallback)
        m_surf.m_mesheventcallback->log() << "attempting to snap vertex " << snapping_vert0 << " to " << snapping_vert1 << std::endl;
    
    //finally, if the splitting succeeds and we have a good vertex pair, try to snap them.
//    bool success = vert_pair_is_snappable(snapping_vert0, snapping_vert1) && snap_vertex_pair(snapping_vert0, snapping_vert1);
    bool success = true;
    if (!vert_pair_is_snappable(snapping_vert0, snapping_vert1))
    {
        success = false;
    } else
    {
        if (m_surf.m_mesh.get_edge_index(snapping_vert0, snapping_vert1) != m_surf.m_mesh.ne())
        {
            // there's an edge connecting the two vertices. it's the responsibility of the collapser, which does extra checks.
            // instead of calling it a day, we should bringin in the collapser to get the job done right here.
            size_t edge_to_collapse = m_surf.m_mesh.get_edge_index(snapping_vert0, snapping_vert1);
            double dummy;
            if (m_surf.m_collapser.edge_is_collapsible(edge_to_collapse, dummy))
                success = m_surf.m_collapser.collapse_edge(edge_to_collapse);
        } else
        {
            success = snap_vertex_pair(snapping_vert0, snapping_vert1);
        }
    }
    
    if (m_surf.m_mesheventcallback)
        m_surf.m_mesheventcallback->log() << "snap " << (success ? "succeeded" : "failed") << std::endl;
    
    return success;
}

// --------------------------------------------------------
///
/// Snap an edge-edge pair by splitting each, and snapping the vertices
///
// --------------------------------------------------------

bool MeshSnapper::snap_face_vertex_pair( size_t face, size_t vertex)
{
    
    assert(m_surf.m_allow_non_manifold);
    
    //TODO Figure out how to deal with boundaries here.
    //TODO Check relative velocities or CCD test.
    
    double s0, s1, s2;
    Vec3d normal;
    Vec3st face_data = m_surf.m_mesh.m_tris[face];
    double dist;
    
    const Vec3d& v_pos = m_surf.get_position(vertex);
    const Vec3d& t0_pos = m_surf.get_position(face_data[0]);
    const Vec3d& t1_pos = m_surf.get_position(face_data[1]);
    const Vec3d& t2_pos = m_surf.get_position(face_data[2]);
    
    //determine the distance, and barycentric coordinates of the closest point
    check_point_triangle_proximity(v_pos,
                                   t0_pos, t1_pos, t2_pos,
                                   dist, s0, s1, s2, normal );
    
    //Depending on the barycentric coordinates, either snap to one of the face vertices,
    //split an edge and snap to it, or split the face and snap to it.
    if (m_surf.m_mesheventcallback) {
        m_surf.m_mesheventcallback->log() << "vf snap: v = " << vertex << " (" << v_pos << ") f = " << face_data[0] << " (" << t0_pos << ") " << face_data[1] << " (" << t1_pos << ") " << face_data[2] << " (" << t2_pos << ")" << std::endl;
        m_surf.m_mesheventcallback->log() << "s0 = " << s0 << " s1 = " << s1 << " s2 = " << s2 << " dist = " << dist << " nearest = " << (t0_pos * s0 + t1_pos * s1 + t2_pos * s2) << std::endl;
    }
    size_t snapping_vertex;
    
    // droplets:
    // special handling to preserve the triple junction: free vertex vs full-solid face and solid vertex vs half-solid face
    // the strategy is to snap the vertex to some point on the triple junction (either a triple junction vertex or a triple junction edge), and prohibit snapping to the middle of a face on the other side of the triple junction.
    // the case of solid vertex vs full-solid face (a solid vertex approaching a neighboring face in a same plane) is also treated here, although it can also be resolved by the general case below, or handled by edge splitting, flipping and collapsing
    bool vs = m_surf.vertex_is_any_solid(vertex);
    bool t0s = m_surf.vertex_is_any_solid(face_data[0]);
    bool t1s = m_surf.vertex_is_any_solid(face_data[1]);
    bool t2s = m_surf.vertex_is_any_solid(face_data[2]);
    if (((vs && (t0s || t1s || t2s)) ||
        (!vs && (t0s && t1s && t2s))) && dist > 0.01 * m_surf.m_merge_proximity_epsilon)   // added a distance criteria, so that extremely close proximities will still be resolved regardless of the triple junction. Not resolving these will be dangerous because near coincidence meshes won't be effectively separated by the dynamics.
    {
        bool t0_on_triple_junction = false;
        for (size_t i = 0; i < m_surf.m_mesh.m_vertex_to_triangle_map[face_data[0]].size(); i++)
        {
            Vec3st t = m_surf.m_mesh.m_tris[m_surf.m_mesh.m_vertex_to_triangle_map[face_data[0]][i]];
            if (!(m_surf.vertex_is_any_solid(t[0]) && m_surf.vertex_is_any_solid(t[1]) && m_surf.vertex_is_any_solid(t[2])))
                t0_on_triple_junction = true;
        }
        t0_on_triple_junction &= m_surf.vertex_is_any_solid(face_data[0]);  // the vertex is on triple junction if it is a solid vertex and incident to at least one non-fully-solid face
        bool t1_on_triple_junction = false;
        for (size_t i = 0; i < m_surf.m_mesh.m_vertex_to_triangle_map[face_data[1]].size(); i++)
        {
            Vec3st t = m_surf.m_mesh.m_tris[m_surf.m_mesh.m_vertex_to_triangle_map[face_data[1]][i]];
            if (!(m_surf.vertex_is_any_solid(t[0]) && m_surf.vertex_is_any_solid(t[1]) && m_surf.vertex_is_any_solid(t[2])))
                t1_on_triple_junction = true;
        }
        t1_on_triple_junction &= m_surf.vertex_is_any_solid(face_data[1]);  // the vertex is on triple junction if it is a solid vertex and incident to at least one non-fully-solid face
        bool t2_on_triple_junction = false;
        for (size_t i = 0; i < m_surf.m_mesh.m_vertex_to_triangle_map[face_data[2]].size(); i++)
        {
            Vec3st t = m_surf.m_mesh.m_tris[m_surf.m_mesh.m_vertex_to_triangle_map[face_data[2]][i]];
            if (!(m_surf.vertex_is_any_solid(t[0]) && m_surf.vertex_is_any_solid(t[1]) && m_surf.vertex_is_any_solid(t[2])))
                t2_on_triple_junction = true;
        }
        t2_on_triple_junction &= m_surf.vertex_is_any_solid(face_data[2]);  // the vertex is on triple junction if it is a solid vertex and incident to at least one non-fully-solid face
        
        if (t0_on_triple_junction && s1 < m_face_threshold && s2 < m_face_threshold)
        {
            snapping_vertex = face_data[0];
            if (m_surf.m_mesheventcallback)
                m_surf.m_mesheventcallback->log() << "snap to triple junction vertex v0" << std::endl;
        } else if (t1_on_triple_junction && s2 < m_face_threshold && s0 < m_face_threshold)
        {
            snapping_vertex = face_data[1];
            if (m_surf.m_mesheventcallback)
                m_surf.m_mesheventcallback->log() << "snap to triple junction vertex v1" << std::endl;
        } else if (t2_on_triple_junction && s0 < m_face_threshold && s1 < m_face_threshold)
        {
            snapping_vertex = face_data[2];
            if (m_surf.m_mesheventcallback)
                m_surf.m_mesheventcallback->log() << "snap to triple junction vertex v2" << std::endl;
        } else if (t0_on_triple_junction && t1_on_triple_junction && s2 < m_face_threshold)
        {
            size_t edge_to_split = m_surf.m_mesh.m_triangle_to_edge_map[face][0];
            assert (edge_to_split == m_surf.m_mesh.get_edge_index(face_data[0], face_data[1]));
            
            Vec3d split_point = (s0 * t0_pos + s1 * t1_pos) / (s0 + s1);
            
            if (m_surf.m_mesheventcallback)
                m_surf.m_mesheventcallback->log() << "attempting to snap to triple junction edge e01: " << split_point << std::endl;
            
            size_t result_vertex;
            if (!m_edgesplitter.edge_is_splittable(edge_to_split) || !m_edgesplitter.split_edge(edge_to_split, result_vertex, false, true, &split_point, std::vector<size_t>(), true))
                return false;
            
            snapping_vertex = result_vertex;
        } else if (t1_on_triple_junction && t2_on_triple_junction && s0 < m_face_threshold)
        {
            size_t edge_to_split = m_surf.m_mesh.m_triangle_to_edge_map[face][1];
            assert (edge_to_split == m_surf.m_mesh.get_edge_index(face_data[1], face_data[2]));
            
            Vec3d split_point = (s1 * t1_pos + s2 * t2_pos) / (s1 + s2);
            
            if (m_surf.m_mesheventcallback)
                m_surf.m_mesheventcallback->log() << "attempting to snap to triple junction edge e12: " << split_point << std::endl;
            
            size_t result_vertex;
            if (!m_edgesplitter.edge_is_splittable(edge_to_split) || !m_edgesplitter.split_edge(edge_to_split, result_vertex, false, true, &split_point, std::vector<size_t>(), true))
                return false;
            
            snapping_vertex = result_vertex;
        } else if (t2_on_triple_junction && t0_on_triple_junction && s1 < m_face_threshold)
        {
            size_t edge_to_split = m_surf.m_mesh.m_triangle_to_edge_map[face][2];
            assert (edge_to_split == m_surf.m_mesh.get_edge_index(face_data[2], face_data[0]));
            
            Vec3d split_point = (s2 * t2_pos + s0 * t0_pos) / (s2 + s0);
            
            if (m_surf.m_mesheventcallback)
                m_surf.m_mesheventcallback->log() << "attempting to snap to triple junction edge e20: " << split_point << std::endl;
            
            size_t result_vertex;
            if (!m_edgesplitter.edge_is_splittable(edge_to_split) || !m_edgesplitter.split_edge(edge_to_split, result_vertex, false, true, &split_point, std::vector<size_t>(), true))
                return false;
            
            snapping_vertex = result_vertex;
        } else
        {
            // we can't allow this case to go through. subdividing this face and snapping there will either create a flap next to the triple junction (which will end up perturbing the triple junction too much when the flap is removed), or create non-manifoldness in the triple junction.
            if (m_surf.m_mesheventcallback)
                m_surf.m_mesheventcallback->log() << "can't find a point to snap to on the face" << std::endl;
            return false;
        }
        
    } else  // the general case below, which is the original code
    if (s0 < m_face_threshold) {
        if(s1 < m_face_threshold) {
            snapping_vertex = face_data[2];
            if (m_surf.m_mesheventcallback)
                m_surf.m_mesheventcallback->log() << "snap to v2" << std::endl;
        }
        else if(s2 < m_face_threshold) {
            snapping_vertex = face_data[1];
            if (m_surf.m_mesheventcallback)
                m_surf.m_mesheventcallback->log() << "snap to v1" << std::endl;
        }
        else {
            size_t result_vertex;
            size_t edge_to_split = m_surf.m_mesh.get_edge_index(face_data[1], face_data[2]);
            
            double edge_frac = s1 / (s1+s2);
            Vec3d split_point = edge_frac * t1_pos + (1-edge_frac) * t2_pos;
            
            if (m_surf.m_mesheventcallback)
                m_surf.m_mesheventcallback->log() << "attempting to snap to e12: " << split_point << std::endl;
            
            if(!m_edgesplitter.edge_is_splittable(edge_to_split) || !m_edgesplitter.split_edge(edge_to_split, result_vertex, false, true, &split_point, std::vector<size_t>(), true))
                return false;
            
            snapping_vertex = result_vertex;
        }
    }
    else if(s1 < m_face_threshold) {
        if(s2 < m_face_threshold) {
            snapping_vertex = face_data[0];
            if (m_surf.m_mesheventcallback)
                m_surf.m_mesheventcallback->log() << "snap to v0" << std::endl;
        }
        else {
            size_t result_vertex;
            size_t edge_to_split = m_surf.m_mesh.get_edge_index(face_data[0], face_data[2]);
            
            double edge_frac = s0 / (s0+s2);
            Vec3d split_point = edge_frac * t0_pos + (1-edge_frac) * t2_pos;
            
            if (m_surf.m_mesheventcallback)
                m_surf.m_mesheventcallback->log() << "attempting to snap to e02: " << split_point << std::endl;
            
            if(!m_edgesplitter.edge_is_splittable(edge_to_split) || !m_edgesplitter.split_edge(edge_to_split, result_vertex, false, true, &split_point, std::vector<size_t>(), true))
                return false;
            
            snapping_vertex = result_vertex;
        }
    }
    else if(s2 < m_face_threshold) {
        size_t result_vertex;
        size_t edge_to_split = m_surf.m_mesh.get_edge_index(face_data[0], face_data[1]);
        
        double edge_frac = s0 / (s0+s1);
        Vec3d split_point = edge_frac * t0_pos + (1-edge_frac) * t1_pos;
        
        if (m_surf.m_mesheventcallback)
            m_surf.m_mesheventcallback->log() << "attempting to snap to e01: " << split_point << std::endl;
        
        if(!m_edgesplitter.edge_is_splittable(edge_to_split) || !m_edgesplitter.split_edge(edge_to_split, result_vertex, false, true, &split_point, std::vector<size_t>(), true))
            return false;
        
        snapping_vertex = result_vertex;
    }
    else{
        size_t result_vertex;
        Vec3d split_point = s0*t0_pos + s1*t1_pos + s2*t2_pos;
        if (m_surf.m_mesheventcallback)
            m_surf.m_mesheventcallback->log() << "split face: " << split_point << std::endl;
        
        if(!m_facesplitter.face_is_splittable(face) || !m_facesplitter.split_face(face, result_vertex, true, &split_point))
            return false;
        
        snapping_vertex = result_vertex;
    }
    
    if (m_surf.m_mesheventcallback)
        m_surf.m_mesheventcallback->log() << "attempting to snap vertex " << snapping_vertex << " to " << vertex << std::endl;
    
    //finally, if the splitting succeeds and we have a good vertex pair, try to snap them.
//    bool success = vert_pair_is_snappable(snapping_vertex, vertex) && snap_vertex_pair(snapping_vertex, vertex);
    bool success = true;
    if (!vert_pair_is_snappable(snapping_vertex, vertex))
    {
        success = false;
    } else
    {
        if (m_surf.m_mesh.get_edge_index(snapping_vertex, vertex) != m_surf.m_mesh.ne())
        {
            // there's an edge connecting the two vertices. it's the responsibility of the collapser, which does extra checks.
            // instead of calling it a day, we should bringin in the collapser to get the job done right here.
            size_t edge_to_collapse = m_surf.m_mesh.get_edge_index(snapping_vertex, vertex);
            double dummy;
            if (m_surf.m_collapser.edge_is_collapsible(edge_to_collapse, dummy))
                success = m_surf.m_collapser.collapse_edge(edge_to_collapse);
        } else
        {
            success = snap_vertex_pair(snapping_vertex, vertex);
        }
    }
    
    if (m_surf.m_mesheventcallback)
        m_surf.m_mesheventcallback->log() << "snap " << (success ? "succeeded" : "failed") << std::endl;
    
    return success;
}

// --------------------------------------------------------
///
/// Determine if the vertex pair should be allowed to snap
///
// --------------------------------------------------------

bool MeshSnapper::vert_pair_is_snappable( size_t vert0, size_t vert1 )
{
    
    // skip deleted vertices
    if(m_surf.m_mesh.vertex_is_deleted(vert0) || m_surf.m_mesh.vertex_is_deleted(vert1) )
        return false;
    
    Vec3d x0, x1;
    x0 = m_surf.get_position(vert0);
    x1 = m_surf.get_position(vert1);
    Vec3d vel0, vel1;
    vel0 = m_surf.get_remesh_velocity(vert0);
    vel1 = m_surf.get_remesh_velocity(vert1);

    if (dot(x1 - x0, vel1 - vel0) >=0 ) {
        return false;
    }
    //shouldn't be calling this on duplicate vertices
    assert(vert0 != vert1);
    
//    //Check if there is an edge connecting the two vertices - if so, we can't do the snap
//    size_t edge_id = m_surf.m_mesh.get_edge_index(vert0, vert1);
//    if(edge_id != m_surf.m_mesh.m_edges.size())
//        return false;
    
    // New policy: If there's an edge connecting the two vertices, don't just bail out saying "snap failed". Instead, call the collapser to process it right away.
    // This is important when there's adaptive remeshing. For example, in the case of a tight fold (which itself could result from collapsing in a relatively flat and uninteresting surface), snapping could create a very short edge, which, if not collapsed right away, will cause the splitter to go crazy around it to the point that collapser may not be able to bring it back down to a coarse mesh.
    
    return true;
    
}

// --------------------------------------------------------
///
/// Determine if the edge edge pair should be snapped
///
// --------------------------------------------------------

bool MeshSnapper::edge_pair_is_snappable( size_t edge0, size_t edge1, double& current_length )
{
    
    // skip deleted vertices
    if(m_surf.m_mesh.edge_is_deleted(edge0) || m_surf.m_mesh.edge_is_deleted(edge1) )
        return false;
    
    //skip duplicate edges
    if(edge0 == edge1)
        return false;
    
    //edges shouldn't share a vertex
    const Vec2st& edge_data0 = m_surf.m_mesh.m_edges[edge0];
    const Vec2st& edge_data1 = m_surf.m_mesh.m_edges[edge1];
    if(edge_data0[0] == edge_data1[0] || edge_data0[0] == edge_data1[1] ||
       edge_data0[1] == edge_data1[0] || edge_data0[1] == edge_data1[1] )
        return false;
    
    //check if the edges are on two triangles sharing an edge, and if so
    //require that the faces be at less than 90 degrees to continue.
    //(otherwise, we're essentially snapping triangle to be degenerate/zero area in its plane.)
    
    //check all triangles incident on the first edge
    for(int i = 0; i < 2; ++i) {
        size_t v0 = edge_data0[i];
        for(int j = 0; j < 2; ++j) {
            size_t v1 = edge_data1[j];
            size_t edge_index = m_surf.m_mesh.get_edge_index(v0, v1);
            if(edge_index != m_surf.m_mesh.m_edges.size()) {
                //now find the two relevant triangles
                size_t tri0, tri1;
                bool found0 = false, found1 = false;
                for(size_t k = 0; k < m_surf.m_mesh.m_edge_to_triangle_map[edge_index].size(); ++k) {
                    size_t cur_tri = m_surf.m_mesh.m_edge_to_triangle_map[edge_index][k];
                    Vec3st tri_data = m_surf.m_mesh.m_tris[cur_tri];
                    if(m_surf.m_mesh.triangle_contains_edge(tri_data, edge_data0)) {
                        tri0 = cur_tri;
                        found0 = true;
                    }
                    if(m_surf.m_mesh.triangle_contains_edge(tri_data, edge_data1)) {
                        tri1 = cur_tri;
                        found1 = true;
                    }
                }
                
                //they shared an edge but we couldn't find a potentially offending triangle pair
                if(!found0 || !found1) continue;
                
                //get the data
                Vec3st tri0_data = m_surf.m_mesh.m_tris[tri0];
                Vec3st tri1_data = m_surf.m_mesh.m_tris[tri1];
                
                //and their normals
                Vec3d normal0 = m_surf.get_triangle_normal(tri0);
                Vec3d normal1 = m_surf.get_triangle_normal(tri1);
                
                //if the tris are _not_ oriented consistently, flip the normal to make the comparison valid.
                Vec2st shared_edge_data = m_surf.m_mesh.m_edges[edge_index];
                if(m_surf.m_mesh.oriented(shared_edge_data[0], shared_edge_data[1], tri0_data) !=
                   m_surf.m_mesh.oriented(shared_edge_data[1], shared_edge_data[0], tri1_data))
                    normal1 = -normal1;
                
                if(dot(normal0, normal1) > 0) //if normals match, don't do it.
                    return false;
            }
        }
    }
    
    
    //TODO extend to handle constraints, solids, and boundaries
    
    double s0, s2;
    Vec3d normal;
    
    check_edge_edge_proximity( m_surf.get_position(edge_data0[0]),
                              m_surf.get_position(edge_data0[1]),
                              m_surf.get_position(edge_data1[0]),
                              m_surf.get_position(edge_data1[1]),
                              current_length, s0, s2, normal );
    
    // for Droplets: use a different (smaller) merging proximity epsilon for liquid sheet puncture, if enabled (on the other hand, merging of two liquid volumes uses the original epsilon)
    Vec3d n0 = Vec3d(0, 0, 0);  // liquid outward normal
    for (size_t i = 0; i < m_surf.m_mesh.m_edge_to_triangle_map[edge0].size(); i++)
    {
        size_t t = m_surf.m_mesh.m_edge_to_triangle_map[edge0][i];
        n0 += m_surf.get_triangle_normal(t) * (m_surf.m_mesh.m_triangle_labels[t][0] < m_surf.m_mesh.m_triangle_labels[t][1] ? -1 : 1);
    }
    n0 /= mag(n0);
    Vec3d n1 = Vec3d(0, 0, 0);  // liquid outward normal
    for (size_t i = 0; i < m_surf.m_mesh.m_edge_to_triangle_map[edge1].size(); i++)
    {
        size_t t = m_surf.m_mesh.m_edge_to_triangle_map[edge1][i];
        n1 += m_surf.get_triangle_normal(t) * (m_surf.m_mesh.m_triangle_labels[t][0] < m_surf.m_mesh.m_triangle_labels[t][1] ? -1 : 1);
    }
    n1 /= mag(n1);
    bool snap_is_liquid_sheet_puncture = (dot(normal, n0) > 0.5 && dot(normal, n1) < -0.5);
    
    double merge_proximity_epsilon = (snap_is_liquid_sheet_puncture ? m_surf.m_merge_proximity_epsilon_for_liquid_sheet_puncture : m_surf.m_merge_proximity_epsilon);
    
    
    

    if(current_length < merge_proximity_epsilon) {
        ////velocity criteria libo
        //Vec3d vel0, vel1, vel2, vel3;
        //vel0 = m_surf.get_remesh_velocity(edge_data0[0]);
        //vel1 = m_surf.get_remesh_velocity(edge_data1[1]);
        //vel2 = m_surf.get_remesh_velocity(edge_data1[0]);
        //vel3 = m_surf.get_remesh_velocity(edge_data1[1]);

        //Vec3d vel_div = (s0 * vel0 + (1 - s0) * vel1) - (s2 * vel2 + (1 - s2) * vel3);
        //if (dot(vel_div, normal) > 0) {
        //    if (m_surf.m_verbose) {
        //        std::cout << "edge0 " << edge0 << " edge1 " << edge1 << " dist= " << current_length << " , <eps=" << merge_proximity_epsilon << "but <vel,n>=" << dot(vel_div, normal) << ">0 moving apart, do not snap\n";
        //    }
        //    return false;
        //}
       
        //check for "dimensional drop-down" cases which would lead to snapping two vertices that already share an edge.
        if(s0 > 1 - m_edge_threshold) {
            if(s2 > 1 - m_edge_threshold) {
                if(m_surf.m_mesh.get_edge_index(edge_data0[0], edge_data1[0]) != m_surf.m_mesh.m_edges.size())
                    return false;
            }
            else if(s2 < m_edge_threshold) {
                if(m_surf.m_mesh.get_edge_index(edge_data0[0], edge_data1[1]) != m_surf.m_mesh.m_edges.size())
                    return false;
            }
        }
        else if(s0 < m_edge_threshold) {
            if(s2 > 1 - m_edge_threshold) {
                if(m_surf.m_mesh.get_edge_index(edge_data0[1], edge_data1[0]) != m_surf.m_mesh.m_edges.size())
                    return false;
            }
            else if(s2 < m_edge_threshold) {
                if(m_surf.m_mesh.get_edge_index(edge_data0[1], edge_data1[1]) != m_surf.m_mesh.m_edges.size())
                    return false;
            }
        }
        return true;
    }
    else
        return false;
    
}


// --------------------------------------------------------
///
/// Determine if the face vertex pair should be snapped
///
// --------------------------------------------------------

bool MeshSnapper::face_vertex_pair_is_snappable( size_t face, size_t vertex, double& current_length )
{
    
    // skip deleted pairs
    if(m_surf.m_mesh.triangle_is_deleted(face) || m_surf.m_mesh.vertex_is_deleted(vertex) )
        return false;
    
    //face shouldn't contain the vertex to be snapped
    const Vec3st& face_data = m_surf.m_mesh.m_tris[face];
    if(m_surf.m_mesh.triangle_contains_vertex(face_data, vertex))
        return false;
    
    //check if the vertex is contained in a face adjacent to the one it's merging with.
    //if so, require that the angle between the faces be less than 90 degrees.
    //Vec3d tri_normal = m_surf.get_triangle_normal(face);
    for(int i = 0; i < 3; ++i) {
        size_t edge = m_surf.m_mesh.m_triangle_to_edge_map[face][i];
        for(size_t j = 0; j < m_surf.m_mesh.m_edge_to_triangle_map[edge].size(); ++j) {
            size_t other_tri = m_surf.m_mesh.m_edge_to_triangle_map[edge][j];
            if(other_tri == face) continue;
            Vec3st other_tri_data = m_surf.m_mesh.m_tris[other_tri];
            if(m_surf.m_mesh.triangle_contains_vertex(other_tri_data, vertex)) {
                //check the angle between the faces
                Vec3d other_normal = m_surf.get_triangle_normal(other_tri);
                
                //if the tris are _not_ oriented consistently, flip the normal to make the comparison valid.
                if(m_surf.m_mesh.oriented(m_surf.m_mesh.m_edges[edge][0], m_surf.m_mesh.m_edges[edge][1], face_data) !=
                   m_surf.m_mesh.oriented(m_surf.m_mesh.m_edges[edge][1], m_surf.m_mesh.m_edges[edge][0], other_tri_data)) {
                    other_normal = -other_normal;
                }
                Vec3d tri_normal = m_surf.get_triangle_normal(face);
                if(dot(other_normal, tri_normal) > 0) {
                    return false;
                }
            }
        }
    }
    
    //TODO extend to handle constraints, solids, and boundaries
    
    double s0, s1, s2;
    Vec3d normal;
    
    check_point_triangle_proximity( m_surf.get_position(vertex),
                                   m_surf.get_position(face_data[0]),
                                   m_surf.get_position(face_data[1]),
                                   m_surf.get_position(face_data[2]),
                                   current_length, s0, s1, s2, normal );
    
    // for Droplets: use a different (smaller) merging proximity epsilon for liquid sheet puncture, if enabled (on the other hand, merging of two liquid volumes uses the original epsilon)
    Vec3d n0 = Vec3d(0, 0, 0);  // liquid outward normal
    for (size_t i = 0; i < m_surf.m_mesh.m_vertex_to_triangle_map[vertex].size(); i++)
    {
        size_t t = m_surf.m_mesh.m_vertex_to_triangle_map[vertex][i];
        n0 += m_surf.get_triangle_normal(t) * (m_surf.m_mesh.m_triangle_labels[t][0] < m_surf.m_mesh.m_triangle_labels[t][1] ? -1 : 1);
    }
    n0 /= mag(n0);
    Vec3d n1 = m_surf.get_triangle_normal(face_data);
    bool snap_is_liquid_sheet_puncture = (dot(normal, n0) > 0.5 && dot(normal, n1) < -0.5);
    
    double merge_proximity_epsilon = (snap_is_liquid_sheet_puncture ? m_surf.m_merge_proximity_epsilon_for_liquid_sheet_puncture : m_surf.m_merge_proximity_epsilon);
    
    if(current_length < merge_proximity_epsilon) {
        
        //anticipate the case where we would drop down to vertex snapping
        //but there is already a connecting edge. That should be an edge collapse, not a snap.
        if(s1 < m_face_threshold && s2 < m_face_threshold &&
           m_surf.m_mesh.get_edge_index(face_data[0], vertex) != m_surf.m_mesh.m_edges.size())
            return false;
        
        if(s0 < m_face_threshold && s2 < m_face_threshold && 
           m_surf.m_mesh.get_edge_index(face_data[1], vertex) != m_surf.m_mesh.m_edges.size())
            return false;
        
        if(s0 < m_face_threshold && s1 < m_face_threshold &&
           m_surf.m_mesh.get_edge_index(face_data[2], vertex) != m_surf.m_mesh.m_edges.size())
            return false;
        
        ////if two vertices are moving apart, do not collapse
        //Vec3d face_proximity_velocity = s0 * m_surf.get_remesh_velocity(face_data[0]) + s1 * m_surf.get_remesh_velocity(face_data[1]) + s2 * m_surf.get_remesh_velocity(face_data[2]);
        //double rel_div = dot(m_surf.get_remesh_velocity(vertex) - face_proximity_velocity, normal);
        //if (rel_div>0) {
        //    if (m_surf.m_verbose) {
        //        std::cout << "v " << vertex << " and f " << face << " dist= " << current_length << " , <eps=" << merge_proximity_epsilon << "but <vel,n>=" << rel_div << ">0 moving apart, do not snap\n";
        //    }
        //    return false;  
        //}


        return true;
    }
    else
        return false;
    
}


bool MeshSnapper::snap_pass()
{
    
    if ( m_surf.m_verbose )
    {
        std::cout << "\n\n\n---------------------- MeshSnapper: collapsing ----------------------" << std::endl;
        std::cout << "m_merge_proximity_epsilon: " << m_surf.m_merge_proximity_epsilon << ", " << m_surf.m_merge_proximity_epsilon_for_liquid_sheet_puncture << std::endl;
        
    }
    
    bool snap_occurred = false;
    
    assert( m_surf.m_dirty_triangles.size() == 0 );
    
    std::vector<SortableProximity> sortable_pairs_to_try;
    
    
    //
    // get sets of geometry pairs to try snapping!
    //
    
    // first the face-vertex pairs
    for(size_t vertex = 0; vertex < m_surf.get_num_vertices(); ++vertex) {
        if(m_surf.m_mesh.vertex_is_deleted(vertex)) continue;
        
        Vec3d vmin, vmax;
        m_surf.vertex_static_bounds(vertex, vmin, vmax);
        vmin -= std::max(m_surf.m_merge_proximity_epsilon, m_surf.m_merge_proximity_epsilon_for_liquid_sheet_puncture) * Vec3d(1,1,1);
        vmax += std::max(m_surf.m_merge_proximity_epsilon, m_surf.m_merge_proximity_epsilon_for_liquid_sheet_puncture) * Vec3d(1,1,1);
        
        std::vector<size_t> overlapping_tris;
        m_surf.m_broad_phase->get_potential_triangle_collisions(vmin, vmax, false, true, overlapping_tris);
        
        for(size_t i = 0; i < overlapping_tris.size(); ++i) {
            size_t face = overlapping_tris[i];
            Vec3st tri_data = m_surf.m_mesh.m_tris[face];
            
            double len;
            if(face_vertex_pair_is_snappable(face, vertex, len))
            {
                SortableProximity prox(face, vertex, len, true);
                sortable_pairs_to_try.push_back(prox);
            }
        }
    }
    
    //now the edge-edge pairs
    for(size_t edge0 = 0; edge0 < m_surf.m_mesh.m_edges.size(); ++edge0) {
        if(m_surf.m_mesh.edge_is_deleted(edge0)) continue;
        
        Vec3d vmin, vmax;
        m_surf.edge_static_bounds(edge0, vmin, vmax);
        vmin -= std::max(m_surf.m_merge_proximity_epsilon, m_surf.m_merge_proximity_epsilon_for_liquid_sheet_puncture) * Vec3d(1,1,1);
        vmax += std::max(m_surf.m_merge_proximity_epsilon, m_surf.m_merge_proximity_epsilon_for_liquid_sheet_puncture) * Vec3d(1,1,1);
        
        std::vector<size_t> overlapping_edges;
        m_surf.m_broad_phase->get_potential_edge_collisions(vmin, vmax, false, true, overlapping_edges);
        
        for(size_t ind = 0; ind < overlapping_edges.size(); ++ind) {
            size_t edge1 = overlapping_edges[ind];
            
            //always use the lower numbered edge, to avoid duplicates
            if(edge0 >= edge1)
                continue;
            
            double len;
            if (edge_pair_is_snappable(edge0, edge1, len))
            {
                SortableProximity prox(edge0, edge1, len, false);
                sortable_pairs_to_try.push_back(prox);
            }
            
        }
    }
    
    //
    // sort in ascending order by distance (prefer to merge nearby geometry first)
    //
    
    // TODO: can we update the local geometry after merges to find newly formed proximities
    // and then using a priority queue, make sure to test them immediately if new ones are closer?
    
    std::sort( sortable_pairs_to_try.begin(), sortable_pairs_to_try.end() );
    
    if ( m_surf.m_verbose )
    {
        std::cout << sortable_pairs_to_try.size() << " candidate pairs sorted" << std::endl;
    }
    
    //
    // attempt to split and snap each pair in the sorted list
    //
    
    if(m_surf.m_mesheventcallback)
        for (size_t si = 0; si < sortable_pairs_to_try.size(); si++)
            m_surf.m_mesheventcallback->log() << "Snap pair: " << sortable_pairs_to_try[si].m_length << ", " << (sortable_pairs_to_try[si].m_face_vert_proximity ? "vf" : "ee") << " pair: " << sortable_pairs_to_try[si].m_index0 << " and " << sortable_pairs_to_try[si].m_index1 << std::endl;
    
    for ( size_t si = 0; si < sortable_pairs_to_try.size(); ++si )
    {
        size_t ind0 = sortable_pairs_to_try[si].m_index0;
        size_t ind1 = sortable_pairs_to_try[si].m_index1;
        
        if(m_surf.m_mesheventcallback)
            m_surf.m_mesheventcallback->log() << "Snap pair to try: " << (sortable_pairs_to_try[si].m_face_vert_proximity ? "vf" : "ee") << " pair: " << ind0 << " and " << ind1 << " with distance " << sortable_pairs_to_try[si].m_length << std::endl;
        
        bool result = false;
        bool attempted = false;
        if(sortable_pairs_to_try[si].m_face_vert_proximity) {
            
            //perform face-vertex split-n-snap
            size_t face = ind0;
            size_t vertex = ind1;
            
            double cur_len;
            if(face_vertex_pair_is_snappable(face, vertex, cur_len)) {
                result = snap_face_vertex_pair(face, vertex);
                attempted = true;
            }
        }
        else {
            //perform edge-edge split-n-snap
            size_t edge0 = ind0;
            size_t edge1 = ind1;
            
            double cur_len;
            if(edge_pair_is_snappable(edge0, edge1, cur_len)) {
                result = snap_edge_pair(edge0, edge1);
                attempted = true;
            }
        }
        
        if ( result )
        { 
            if (m_surf.m_mesheventcallback)
                m_surf.m_mesheventcallback->log() << "snap successful" << std::endl;
        }
        else if(attempted) {
            if (m_surf.m_mesheventcallback)
                m_surf.m_mesheventcallback->log() << "snap failed" << std::endl;
            //Snapping attempted and failed
        }
        else {
            if (m_surf.m_mesheventcallback)
                m_surf.m_mesheventcallback->log() << "snap not attempted" << std::endl;
            //Snapping not attempted because the situation changed.
        }
        
        snap_occurred |= result;
        
    }
    
    
    return snap_occurred;
    
}
    
    
}
