// ---------------------------------------------------------
//
//  meshmerger.cpp
//  Tyson Brochu 2011
//  Christopher Batty, Fang Da 2014
//
//  Search for mesh edges which are near to each other, zipper their neighbouring triangles together.
//
// ---------------------------------------------------------


#include <meshmerger.h>

#include <broadphase.h>
#include <collisionqueries.h>
#include <queue>
#include <runstats.h>
#include <surftrack.h>

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
/// Move vertices around so v[0] and v[4] are closest together
///
// --------------------------------------------------------

void MeshMerger::twist_vertices( size_t *zipper_vertices )
{
    double min_dist = 1e+30, dist;
    Vec2st min_pair((size_t)~0, (size_t)~0);
    
    // find the closest pair among the 8 vertices
    for (int i=0; i<4; ++i)
    {
        for (int j=4; j<8; ++j)
        {
            dist = mag( m_surf.get_position(zipper_vertices[i]) - m_surf.get_position(zipper_vertices[j]) );
            if (dist < min_dist)
            {
                min_dist = dist;
                min_pair[0] = i;
                min_pair[1] = j;
            }
        }
    }
    
    size_t new_vertices[8];
    for (int i=0; i<4; ++i)
    {
        new_vertices[i]   = zipper_vertices[(min_pair[0] + i) % 4];
        new_vertices[i+4] = zipper_vertices[(min_pair[1] + i - 4) % 4 + 4];
    }
    
    memcpy( zipper_vertices, new_vertices, 8 * sizeof(size_t) );
    
}

bool MeshMerger::get_zipper_triangles( size_t edge_index_a, size_t edge_index_b,
                                      std::vector<Vec3st>& output_triangles, int& shared_label )
{
    assert( output_triangles.size() == 8 );
    
    const Vec2st& edge_a = m_surf.m_mesh.m_edges[edge_index_a];
    const Vec2st& edge_b = m_surf.m_mesh.m_edges[edge_index_b];
    
    size_t zipper_vertices[8];
    
    //put endpoints of zipper edges into zipper list
    zipper_vertices[0] = edge_a[0];
    zipper_vertices[2] = edge_a[1];
    zipper_vertices[4] = edge_b[0];
    zipper_vertices[6] = edge_b[1];
    
    //grab the two incident tris
    const std::vector<size_t>& incident_triangles_a = m_surf.m_mesh.m_edge_to_triangle_map[edge_index_a];
    const std::vector<size_t>& incident_triangles_b = m_surf.m_mesh.m_edge_to_triangle_map[edge_index_b];
    
    assert( incident_triangles_b.size() == 2 );       // should be checked before calling this function
    assert( incident_triangles_a.size() == 2 );
    
    //get wings of the first edge
    const Vec3st& inc_tri_a0 = m_surf.m_mesh.get_triangle( incident_triangles_a[0] );
    const Vec3st& inc_tri_a1 = m_surf.m_mesh.get_triangle( incident_triangles_a[1] );
    
    //get wings of the second edge
    const Vec3st& inc_tri_b0 = m_surf.m_mesh.get_triangle( incident_triangles_b[0] );
    const Vec3st& inc_tri_b1 = m_surf.m_mesh.get_triangle( incident_triangles_b[1] );
    
    //work out which will be the "middle" or shared region, i.e. the one that the tunnel will penetrate.
    //it must be one where there is a duplicate region label.
    //(There may be *two* duplicate regions in the re-merging / 2-phase case. Doesn't matter as long
    //as we pick one shared region and be consistent.)
    
    //get two labels that we'll use to do the ordering decisions.
    const Vec2i& inc_tri_a0_label = m_surf.m_mesh.get_triangle_label(incident_triangles_a[0]);
    const Vec2i& inc_tri_b0_label = m_surf.m_mesh.get_triangle_label(incident_triangles_b[0]);
    
    //work out which of the labels is shared between the two triangles
    shared_label = -1;
    if(inc_tri_a0_label[0] == inc_tri_b0_label[0] || inc_tri_a0_label[0] == inc_tri_b0_label[1]) {
        shared_label = inc_tri_a0_label[0];
    }
    else if(inc_tri_a0_label[1] == inc_tri_b0_label[0] || inc_tri_a0_label[1] == inc_tri_b0_label[1]) {
        shared_label = inc_tri_a0_label[1];
    }
    else {
        //Note: this case can happen if a triangle comes close to another triangle, when the two are also separated
        //by yet ANOTHER interface. Collision detection would cull it, but lets kill it here instead.
        return false;
    }
    
    //get the other two vertices comprising the first patch
    size_t third_vertices[2];
    third_vertices[0] = m_surf.m_mesh.get_third_vertex( zipper_vertices[0], zipper_vertices[2], inc_tri_a0 );
    third_vertices[1] = m_surf.m_mesh.get_third_vertex( zipper_vertices[0], zipper_vertices[2], inc_tri_a1 );
    
    //assign the vertex ordering so that the loop is consistently oriented with *triangle a0*.
    if ( m_surf.m_mesh.oriented(zipper_vertices[0], zipper_vertices[2], inc_tri_a0 ) )
    {
        zipper_vertices[1] = third_vertices[1];
        zipper_vertices[3] = third_vertices[0];
    }
    else
    {
        zipper_vertices[1] = third_vertices[0];
        zipper_vertices[3] = third_vertices[1];
    }
    
    //get the other two vertices comprising the second patch
    third_vertices[0] = m_surf.m_mesh.get_third_vertex( zipper_vertices[4], zipper_vertices[6], inc_tri_b0 );
    third_vertices[1] = m_surf.m_mesh.get_third_vertex( zipper_vertices[4], zipper_vertices[6], inc_tri_b1 );
    
    //assign the vertex ordering to be consistent with *triangle b0*.
    if ( m_surf.m_mesh.oriented(zipper_vertices[4], zipper_vertices[6], inc_tri_b0))
    {
        zipper_vertices[5] = third_vertices[1];
        zipper_vertices[7] = third_vertices[0];
    }
    else
    {
        zipper_vertices[5] = third_vertices[0];
        zipper_vertices[7] = third_vertices[1];
    }
    
    ////////////////////////////////////////////////////////////
    
    // Check for degenerate case
    for ( unsigned int i = 0; i < 8; ++i)
    {
        for ( unsigned int j = i+1; j < 8; ++j)
        {
            
            if ( zipper_vertices[i] == zipper_vertices[j] )         // vertices not distinct
            {
                return false;
            }
            
            // Check if an edge already exists between two vertices in opposite edge neighbourhoods
            // (i.e. look for an edge which would be created by zippering)
            
            if ( (i < 4) && (j > 3) )
            {
                
                for ( size_t ii = 0; ii < m_surf.m_mesh.m_vertex_to_edge_map[ zipper_vertices[i] ].size(); ++ii )
                {
                    for ( size_t jj = 0; jj < m_surf.m_mesh.m_vertex_to_edge_map[ zipper_vertices[j] ].size(); ++jj )
                    {
                        if ( m_surf.m_mesh.m_vertex_to_edge_map[ zipper_vertices[i] ][ii] == m_surf.m_mesh.m_vertex_to_edge_map[ zipper_vertices[j] ][jj] )
                        {
                            return false;
                        }
                    }
                }
            }
            
        }
    }
    
    // Twist so that vertices 0 and 4 are the pair closest together
    twist_vertices( zipper_vertices );
    
    // There are two options for triangle connectivities
    // based on the labels of a0 and b0.
    int shared_label_pos_a = (shared_label == inc_tri_a0_label[0] ? 0 : 1);
    int shared_label_pos_b = (shared_label == inc_tri_b0_label[0] ? 0 : 1);
    
    if(shared_label_pos_a == shared_label_pos_b) {
        //normal orientation
        output_triangles[0] = Vec3st( zipper_vertices[0], zipper_vertices[1], zipper_vertices[7] );
        output_triangles[1] = Vec3st( zipper_vertices[0], zipper_vertices[7], zipper_vertices[4] );
        output_triangles[2] = Vec3st( zipper_vertices[1], zipper_vertices[2], zipper_vertices[6] );
        output_triangles[3] = Vec3st( zipper_vertices[1], zipper_vertices[6], zipper_vertices[7] );
        output_triangles[4] = Vec3st( zipper_vertices[2], zipper_vertices[3], zipper_vertices[5] );
        output_triangles[5] = Vec3st( zipper_vertices[2], zipper_vertices[5], zipper_vertices[6] );
        output_triangles[6] = Vec3st( zipper_vertices[3], zipper_vertices[0], zipper_vertices[4] );
        output_triangles[7] = Vec3st( zipper_vertices[3], zipper_vertices[4], zipper_vertices[5] );
    }
    else {
        //switched orientation
        output_triangles[0] = Vec3st( zipper_vertices[0], zipper_vertices[1], zipper_vertices[5] );
        output_triangles[1] = Vec3st( zipper_vertices[0], zipper_vertices[5], zipper_vertices[4] );
        output_triangles[2] = Vec3st( zipper_vertices[1], zipper_vertices[2], zipper_vertices[6] );
        output_triangles[3] = Vec3st( zipper_vertices[1], zipper_vertices[6], zipper_vertices[5] );
        output_triangles[4] = Vec3st( zipper_vertices[2], zipper_vertices[3], zipper_vertices[7] );
        output_triangles[5] = Vec3st( zipper_vertices[2], zipper_vertices[7], zipper_vertices[6] );
        output_triangles[6] = Vec3st( zipper_vertices[3], zipper_vertices[0], zipper_vertices[4] );
        output_triangles[7] = Vec3st( zipper_vertices[3], zipper_vertices[4], zipper_vertices[7] );
    }
    
    //now we have a tunnel, whose triangles match the orientation of triangle a0.
    //we can simply copy the label over.
    //both triangle b labels needs to be adjusted so that the shared_region is replaced
    //with a's interior (region 0, the non-shared region).
    
    ////////////////////////////////////////////////////////////
    
    return true;
}


// --------------------------------------------------------
///
/// Check whether the introduction of the new zippering triangles causes a collision
///
// --------------------------------------------------------

bool MeshMerger::zippering_introduces_collision( const std::vector<Vec3st>& new_triangles,
                                                const std::vector<size_t>& deleted_triangles )
{
    for ( size_t i = 0; i < new_triangles.size(); ++i )
    {
        // Check all existing edges vs new triangles
        Vec3d low, high;
        minmax(m_surf.get_position(new_triangles[i][0]), m_surf.get_position(new_triangles[i][1]), m_surf.get_position(new_triangles[i][2]), low, high);
        
        std::vector<size_t> overlapping_triangles;
        m_surf.m_broad_phase->get_potential_triangle_collisions( low, high, true, true, overlapping_triangles );
        
        const Vec3st& current_triangle = new_triangles[i];
        
        // Check to make sure there doesn't already exist triangles with the same vertices
        for ( size_t t = 0; t < overlapping_triangles.size(); ++t )
        {
            const Vec3st& other_triangle = m_surf.m_mesh.get_triangle(overlapping_triangles[t]);
            
            if (    ((current_triangle[0] == other_triangle[0]) || (current_triangle[0] == other_triangle[1]) || (current_triangle[0] == other_triangle[2]))
                && ((current_triangle[1] == other_triangle[0]) || (current_triangle[1] == other_triangle[1]) || (current_triangle[1] == other_triangle[2]))
                && ((current_triangle[2] == other_triangle[0]) || (current_triangle[2] == other_triangle[1]) || (current_triangle[2] == other_triangle[2])) )
            {
                return true;
            }
        }
        
        // Check all existing triangles vs new triangles
        for ( size_t t = 0; t < overlapping_triangles.size(); ++t )
        {
            bool go_to_next_triangle = false;
            for ( size_t d = 0; d < deleted_triangles.size(); ++d )
            {
                if ( overlapping_triangles[t] == deleted_triangles[d] )
                {
                    go_to_next_triangle = true;
                    break;
                }
            }
            if ( go_to_next_triangle )
            {
                continue;
            }
            
            if ( check_triangle_triangle_intersection( new_triangles[i],
                                                      m_surf.m_mesh.get_triangle(overlapping_triangles[t]),
                                                      m_surf.get_positions() ) )
            {
                return true;
            }
        }
        
        // Check new triangles vs each other
        for ( size_t j = 0; j < new_triangles.size(); ++j )
        {
            if ( i == j )  { continue; }
            
            if ( check_triangle_triangle_intersection( new_triangles[i],
                                                      new_triangles[j],
                                                      m_surf.get_positions() ) )
            {
                return true;
            }
        }
    }
    
    // For real collision safety, we need to check for vertices inside the new, joined volume.
    // Checking edges vs triangles is technically not enough.
    
    return false;
}

// --------------------------------------------------------
///
/// Attempt to merge between two edges
///
// --------------------------------------------------------

bool MeshMerger::zipper_edges( size_t edge_index_a, size_t edge_index_b )
{
    // For now we'll only zipper edges which are incident on 2 triangles
    if ( m_surf.m_mesh.m_edge_to_triangle_map[edge_index_a].size() != 2 || m_surf.m_mesh.m_edge_to_triangle_map[edge_index_b].size() != 2 )
    {
        g_stats.add_to_int( "merge_non_manifold_edges", 1 );
        return false;
    }
    
    //
    // Get the set of 8 new triangles which will join the two holes in the mesh
    //
    
    std::vector<Vec3st> new_triangles;
    new_triangles.resize(8);
    int shared_label = -1;
    if ( false == get_zipper_triangles( edge_index_a, edge_index_b, new_triangles, shared_label ) )
    {
        g_stats.add_to_int( "merge_no_set_of_triangles", 1 );
        return false;
    }
    
    //get the two edges
    Vec2st edge_a = m_surf.m_mesh.m_edges[edge_index_a];
    Vec2st edge_b = m_surf.m_mesh.m_edges[edge_index_b];
    
    //get the triangles
    size_t triangle_a_0 = m_surf.m_mesh.m_edge_to_triangle_map[edge_index_a][0];
    size_t triangle_a_1 = m_surf.m_mesh.m_edge_to_triangle_map[edge_index_a][1];
    size_t triangle_b_0 = m_surf.m_mesh.m_edge_to_triangle_map[edge_index_b][0];
    size_t triangle_b_1 = m_surf.m_mesh.m_edge_to_triangle_map[edge_index_b][1];
    
    //get the labels
    Vec2i label_a_0 = m_surf.m_mesh.get_triangle_label(triangle_a_0);
    Vec2i label_a_1 = m_surf.m_mesh.get_triangle_label(triangle_a_1);
    Vec2i label_b_0 = m_surf.m_mesh.get_triangle_label(triangle_b_0);
    Vec2i label_b_1 = m_surf.m_mesh.get_triangle_label(triangle_b_1);
    
    //The triangles should not have the same material on both sides.
    assert(label_a_0[0] != label_a_0[1]);
    assert(label_a_1[0] != label_a_1[1]);
    assert(label_b_0[0] != label_b_0[1]);
    assert(label_b_1[0] != label_b_1[1]);
    
    // record the vertices involved
    size_t v0 = m_surf.m_mesh.m_edges[edge_index_a][0];
    size_t v1 = m_surf.m_mesh.m_edges[edge_index_a][1];
    size_t v2 = m_surf.m_mesh.m_edges[edge_index_b][0];
    size_t v3 = m_surf.m_mesh.m_edges[edge_index_b][1];
    
    //each patch should have labels consistent within themselves and their orientations.
    if (m_surf.m_mesh.oriented(v0, v1, m_surf.m_mesh.get_triangle(triangle_a_0)) == m_surf.m_mesh.oriented(v1, v0, m_surf.m_mesh.get_triangle(triangle_a_1)))
    {
        assert(label_a_0 == label_a_1);
    }
    else
    {
        assert(label_a_0[0] == label_a_1[1] && label_a_0[1] == label_a_1[0]);
    }
    if (m_surf.m_mesh.oriented(v2, v3, m_surf.m_mesh.get_triangle(triangle_b_0)) == m_surf.m_mesh.oriented(v3, v2, m_surf.m_mesh.get_triangle(triangle_b_1)))
    {
        assert(label_b_0 == label_b_1);
    }
    else
    {
        assert(label_b_0[0] == label_b_1[1] && label_b_0[1] == label_b_1[0]);
    }
    
    Vec2i label_a = label_a_0;
    Vec2i label_b = label_b_0;
    
    int region_0;   // region behind surface a
    int region_1;   // region behind surface b
    int region_2;   // region between the two surfaces
    
    region_2 = shared_label; //already decided in the zippering choice above
    
    //choose other two labels accordingly.
    region_0 = label_a[0] == shared_label? label_a[1] : label_a[0];
    region_1 = label_b[0] == shared_label? label_b[1] : label_b[0];
    
    
    // Keep a list of triangles to delete
    std::vector<size_t> deleted_triangles;
    deleted_triangles.push_back( m_surf.m_mesh.m_edge_to_triangle_map[edge_index_a][0] );
    deleted_triangles.push_back( m_surf.m_mesh.m_edge_to_triangle_map[edge_index_a][1] );
    // if the labels don't match, keep the next two triangles as the separating film
    if(region_0 == region_1) {
        //the labels match, so this is two of the same region re-combining
        //do a standard two-phase merge. (i.e. don't add the separating film.)
        deleted_triangles.push_back( m_surf.m_mesh.m_edge_to_triangle_map[edge_index_b][0] );
        deleted_triangles.push_back( m_surf.m_mesh.m_edge_to_triangle_map[edge_index_b][1] );
    }
    
    
    //
    // Check the new triangles for collision safety, ignoring the triangles which will be deleted
    //
    
    bool saved_verbose = m_surf.m_verbose;
    m_surf.m_verbose = false;
    
    if ( m_surf.m_collision_safety && zippering_introduces_collision( new_triangles, deleted_triangles ) )
    {
        m_surf.m_verbose = saved_verbose;
        g_stats.add_to_int( "merge_not_intersection_safe", 1 );
        return false;
    }
    
    m_surf.m_verbose = saved_verbose;
    
    //we built our tunnel to be consistent with triangle a0
    Vec2i label_tunnel =  label_a;
    
    //on the original b triangles, just replace the shared region label (where it occurs) with a's other region
    //which now comprises the interior of the tunnel.
    Vec2i label_new_b_0 = label_b_0;
    if(label_new_b_0[0] == shared_label) label_new_b_0[0] = region_0;
    if(label_new_b_0[1] == shared_label) label_new_b_0[1] = region_0;
    
    Vec2i label_new_b_1 = label_b_1;
    if(label_new_b_1[0] == region_2) label_new_b_1[0] = region_0;
    if(label_new_b_1[1] == region_2) label_new_b_1[1] = region_0;
    
    Vec3st tri_b_0 = m_surf.m_mesh.get_triangle(triangle_b_0);
    Vec3st tri_b_1 = m_surf.m_mesh.get_triangle(triangle_b_1);
    
    std::vector<Vec2i> created_triangle_labels;
    
    //
    // Add the new triangles
    //
    
    std::vector<size_t> created_triangles;
    size_t new_index = m_surf.add_triangle( new_triangles[0], label_tunnel );
    
    m_surf.m_dirty_triangles.push_back( new_index );
    created_triangles.push_back(new_index);
    created_triangle_labels.push_back(label_tunnel);
    
    new_index = m_surf.add_triangle( new_triangles[1], label_tunnel );
    m_surf.m_dirty_triangles.push_back( new_index );
    created_triangles.push_back(new_index);
    created_triangle_labels.push_back(label_tunnel);
    
    new_index = m_surf.add_triangle( new_triangles[2], label_tunnel );
    m_surf.m_dirty_triangles.push_back( new_index );
    created_triangles.push_back(new_index);
    created_triangle_labels.push_back(label_tunnel);
    
    new_index = m_surf.add_triangle( new_triangles[3], label_tunnel );
    m_surf.m_dirty_triangles.push_back( new_index );
    created_triangles.push_back(new_index);
    created_triangle_labels.push_back(label_tunnel);
    
    new_index = m_surf.add_triangle( new_triangles[4], label_tunnel );
    m_surf.m_dirty_triangles.push_back( new_index );
    created_triangles.push_back(new_index);
    created_triangle_labels.push_back(label_tunnel);
    
    new_index = m_surf.add_triangle( new_triangles[5], label_tunnel );
    m_surf.m_dirty_triangles.push_back( new_index );
    created_triangles.push_back(new_index);
    created_triangle_labels.push_back(label_tunnel);
    
    new_index = m_surf.add_triangle( new_triangles[6], label_tunnel );
    m_surf.m_dirty_triangles.push_back( new_index );
    created_triangles.push_back(new_index);
    created_triangle_labels.push_back(label_tunnel);
    
    new_index = m_surf.add_triangle( new_triangles[7], label_tunnel );
    m_surf.m_dirty_triangles.push_back( new_index );
    created_triangles.push_back(new_index);
    created_triangle_labels.push_back(label_tunnel);
    
    //
    // Remove the old triangles as needed
    //
    
    m_surf.remove_triangle( m_surf.m_mesh.m_edge_to_triangle_map[edge_index_a][0] );
    m_surf.remove_triangle( m_surf.m_mesh.m_edge_to_triangle_map[edge_index_a][0] );
    if(region_0 == region_1) {
        //the original labels matched, so this is two of the same region combining together.
        //Delete the wall/film.
        m_surf.remove_triangle( m_surf.m_mesh.m_edge_to_triangle_map[edge_index_b][0] );
        m_surf.remove_triangle( m_surf.m_mesh.m_edge_to_triangle_map[edge_index_b][0] );
    } else {
        //correct b's labels, since we're keeping it
        m_surf.m_mesh.set_triangle_label(triangle_b_0, label_new_b_0);
        m_surf.m_mesh.set_triangle_label(triangle_b_1, label_new_b_1);
    }
    
    
    
    
    //Record the event for posterity
    MeshUpdateEvent update(MeshUpdateEvent::MERGE);
    update.m_v0 = v0;
    update.m_v1 = v1;
    update.m_v2 = v2;
    update.m_v3 = v3;
    update.m_deleted_tris = deleted_triangles;
    update.m_created_tris = created_triangles;
    update.m_created_tri_data = new_triangles;
    update.m_created_tri_labels = created_triangle_labels;
    if(region_0 != region_1) { //account for relabeling of retained film
        update.m_dirty_tris.push_back(std::pair<size_t, Vec2i>(triangle_b_0, label_new_b_0));
        update.m_dirty_tris.push_back(std::pair<size_t, Vec2i>(triangle_b_1, label_new_b_1));
    }
    m_surf.m_mesh_change_history.push_back(update);
    
    ////////////////////////////////////////////////////////////
    
    return true;
    
}

// ---------------------------------------------------------
///
/// Look for pairs of edges close to each other, attempting to merge when close edges are found.
///
// ---------------------------------------------------------

bool MeshMerger::merge_pass( )
{
    
    std::queue<Vec2st> edge_edge_candidates;
    
    //
    // Check edge-edge proximities for zippering candidates
    //
    
    bool merge_occurred = false;
    
    for(size_t i = 0; i < m_surf.m_mesh.m_edges.size(); i++)
    {
        const Vec2st& e0 = m_surf.m_mesh.m_edges[i];
        
        if ( e0[0] == e0[1] ) { continue; }
        if ( m_surf.edge_is_any_solid(i) ) { continue; }
        
        if ( m_surf.m_mesh.m_is_boundary_vertex[ e0[0] ] || m_surf.m_mesh.m_is_boundary_vertex[ e0[1] ] )  { continue; }  // skip boundary vertices
        
        Vec3d emin, emax;
        m_surf.edge_static_bounds(i, emin, emax);
        emin -= std::max(m_surf.m_merge_proximity_epsilon, m_surf.m_merge_proximity_epsilon_for_liquid_sheet_puncture) * Vec3d(1,1,1);
        emax += std::max(m_surf.m_merge_proximity_epsilon, m_surf.m_merge_proximity_epsilon_for_liquid_sheet_puncture) * Vec3d(1,1,1);
        
        std::vector<size_t> edge_candidates;
        m_surf.m_broad_phase->get_potential_edge_collisions( emin, emax, false, true, edge_candidates );
        
        for(size_t j = 0; j < edge_candidates.size(); j++)
        {
            size_t proximal_edge_index = edge_candidates[j];
            const Vec2st& e1 = m_surf.m_mesh.m_edges[proximal_edge_index];
            
            if ( proximal_edge_index <= i )
            {
                continue;
            }
            
            if ( m_surf.m_mesh.m_is_boundary_vertex[ e1[0] ] || m_surf.m_mesh.m_is_boundary_vertex[ e1[1] ] )  { continue; }  // skip boundary vertices
            
            if(e0[0] != e1[0] && e0[0] != e1[1] && e0[1] != e1[0] && e0[1] != e1[1])
            {
                double distance, s0, s2;
                Vec3d normal;
                
                check_edge_edge_proximity( m_surf.get_position(e0[0]), 
                                          m_surf.get_position(e0[1]), 
                                          m_surf.get_position(e1[0]), 
                                          m_surf.get_position(e1[1]), 
                                          distance, s0, s2, normal );
                
                if (distance < m_surf.m_merge_proximity_epsilon)
                {
                    if ( m_surf.m_verbose ) 
                    { 
                        std::cout << "proximity: " << distance << " / " << m_surf.m_merge_proximity_epsilon << std::endl; //proximities[i].distance << std::endl; 
                    }
                    
                    if ( zipper_edges( i, proximal_edge_index ) )
                    {
                        m_surf.trim_degeneracies( m_surf.m_dirty_triangles );
                        
                        if ( m_surf.m_verbose ) 
                        { 
                            std::cout << "zippered" << std::endl; 
                        }
                        
                        merge_occurred = true;
                        g_stats.add_to_int( "merge_success", 1 );
                    }
                    else
                    {
                        g_stats.add_to_int( "merge_failed", 1 );
                    }
                    
                }
            }
        }
    }
    
    
    if ( merge_occurred )
    {
        m_surf.assert_no_degenerate_triangles();
    }
    
    return merge_occurred;
    
    
}

}