// ---------------------------------------------------------
//
//  meshpincher.cpp
//  Tyson Brochu 2011
//  Christopher Batty, Fang Da 2014
//
//  Identifies "singular vertices", defined as having more than one connected triangle neighbourhoods, and
//  splits the mesh surface at these vertices.
//
// ---------------------------------------------------------

#include <meshcutter.h>

#include <broadphase.h>
#include <surftrack.h>
#include <set>

// ---------------------------------------------------------
// Member function definitions
// ---------------------------------------------------------

// --------------------------------------------------------
///
/// Partition the triangles incident to a vertex into connected components
///
// --------------------------------------------------------

namespace LosTopos {
    
void MeshCutter::partition_edge_neighbourhood( size_t edge_index, std::vector< TriangleSet >& connected_components )
{
    
    Vec2st edge_data = m_surf.m_mesh.m_edges[edge_index];
    size_t vtx0 = edge_data[0];
    size_t vtx1 = edge_data[1];
    
    //ensure this is an interior edge, with a splitting vertex on the boundary
    assert(m_surf.m_mesh.m_is_boundary_vertex[vtx0] || m_surf.m_mesh.m_is_boundary_vertex[vtx1]);
    assert(!m_surf.m_mesh.m_is_boundary_edge[edge_index]);  // require the edge to be interior, otherwise splitting makes no sense
    if(m_surf.m_mesh.m_edge_to_triangle_map[edge_index].size() != 2)
        std::cout << "Assert will fail, size is: " << m_surf.m_mesh.m_edge_to_triangle_map[edge_index].size() << std::endl;
    assert(m_surf.m_mesh.m_edge_to_triangle_map[edge_index].size() == 2); // require manifold geometry for now.
    assert(!m_surf.m_mesh.edge_is_deleted(edge_index)); //don't process dead edges
    
    // find triangles incident to boundary vertices
    
    TriangleSet incident_triangles;
    if(m_surf.m_mesh.m_is_boundary_vertex[vtx0]) {
        TriangleSet set0 = m_surf.m_mesh.m_vertex_to_triangle_map[vtx0];
        incident_triangles.insert(incident_triangles.end(), set0.begin(), set0.end());
    }
    if(m_surf.m_mesh.m_is_boundary_vertex[vtx1]) {
        TriangleSet set1 = m_surf.m_mesh.m_vertex_to_triangle_map[vtx1];
        incident_triangles.insert(incident_triangles.end(), set1.begin(), set1.end());
    }
    
    //TODO Faster to use an ACTUAL std::set when building the list?
    sort( incident_triangles.begin(), incident_triangles.end() );
    incident_triangles.erase( unique( incident_triangles.begin(), incident_triangles.end() ), incident_triangles.end() );
    
    
    // unvisited triangles which are adjacent to some visited ones and incident to vt
    TriangleSet unvisited_triangles, visited_triangles;
    
    std::vector<size_t> edges; edges.push_back(edge_index);
    perform_partitioning(incident_triangles, edges, connected_components);
    
}



void MeshCutter::partition_edge_neighbourhood_internal( size_t edge0, size_t edge1, std::vector< TriangleSet >& connected_components )
{
    
    Vec2st edge_data0 = m_surf.m_mesh.m_edges[edge0];
    Vec2st edge_data1 = m_surf.m_mesh.m_edges[edge1];
    
    size_t shared_vert = m_surf.m_mesh.get_common_vertex(edge0, edge1);
    std::cout << "Shared: " << shared_vert << std::endl;
    size_t vtx0 = edge_data0[0] == shared_vert? edge_data0[1] : edge_data0[0];
    size_t vtx1 = edge_data1[0] == shared_vert? edge_data1[1] : edge_data1[0];
    
    std::cout << "Others: " << vtx0 << " and " << vtx1 << std::endl;
    std::cout << "Boundary?\n" << m_surf.m_mesh.m_is_boundary_vertex[vtx0] << " "
    << m_surf.m_mesh.m_is_boundary_vertex[vtx1] << " "
    << m_surf.m_mesh.m_is_boundary_vertex[shared_vert] << std::endl;
    
    //ensure this is a pair of interior edges, with no boundary vertices
    assert(!m_surf.m_mesh.m_is_boundary_vertex[vtx0] &&
           !m_surf.m_mesh.m_is_boundary_vertex[vtx1] &&
           !m_surf.m_mesh.m_is_boundary_vertex[shared_vert]);
    
    // require the edges to be interior, otherwise splitting makes no sense
    assert(!m_surf.m_mesh.m_is_boundary_edge[edge0]);
    assert(!m_surf.m_mesh.m_is_boundary_edge[edge1]);
    
    // require manifold geometry for now.
    assert(m_surf.m_mesh.m_edge_to_triangle_map[edge0].size() == 2);
    assert(m_surf.m_mesh.m_edge_to_triangle_map[edge1].size() == 2);
    
    // find triangles incident to central shared vertex
    TriangleSet incident_triangles = m_surf.m_mesh.m_vertex_to_triangle_map[shared_vert];
    
    // unvisited triangles which are adjacent to some visited ones and incident to vt
    std::vector<size_t> edges;
    edges.push_back(edge0);
    edges.push_back(edge1);
    
    perform_partitioning(incident_triangles, edges, connected_components);
}

// --------------------------------------------------------
///
/// Actually do the partitioning, given a triangle set and separating edges
///
// --------------------------------------------------------

void MeshCutter::perform_partitioning(const std::vector<size_t>& incident_tris, const std::vector<size_t>& separating_edges, std::vector< TriangleSet >& connected_components) {
    
    std::vector<size_t> incident_triangles = incident_tris; //copy, since we're being destructive here
    TriangleSet unvisited_triangles, visited_triangles;
    
    while ( incident_triangles.size() > 0 )
    {
        unvisited_triangles.clear();
        visited_triangles.clear();
        unvisited_triangles.push_back( incident_triangles.back() );
        
        while ( unvisited_triangles.size() > 0 )
        {
            // get an unvisited triangle
            size_t curr_tri = unvisited_triangles.back();
            unvisited_triangles.pop_back();
            
            // delete it from triangles_incident_to_vertex
            incident_triangles.erase( find(incident_triangles.begin(), incident_triangles.end(), curr_tri) );
            
            // put it on closed
            visited_triangles.push_back(curr_tri);
            
            // find a triangle which is incident to vertex and adjacent to curr_tri
            for ( size_t i = 0; i < incident_triangles.size(); ++i )
            {
                size_t incident_triangle_index =  incident_triangles[i];
                
                if ( curr_tri == incident_triangle_index )
                {
                    continue;
                }
                
                //consider triangles adjacent if they touch, and if their shared edge isn't among the separating edges
                if ( m_surf.m_mesh.triangles_are_adjacent( curr_tri, incident_triangle_index ) )
                {
                    
                    bool found_separator = false;
                    for(size_t k = 0; k < separating_edges.size(); ++k) {
                        if(m_surf.m_mesh.get_common_edge(curr_tri, incident_triangle_index) == separating_edges[k]) {
                            found_separator = true;
                            break;
                        }
                    }
                    if(found_separator) continue;
                    
                    // if not in visited_triangles or unvisited_triangles, put them on unvisited_triangles
                    if ( ( find(unvisited_triangles.begin(), unvisited_triangles.end(), incident_triangle_index) == unvisited_triangles.end() ) &&
                        ( find(visited_triangles.begin(), visited_triangles.end(), incident_triangle_index) == visited_triangles.end() ) )
                    {
                        unvisited_triangles.push_back( incident_triangle_index );
                    }
                }
            }
        }
        
        // one connected component = visited triangles
        connected_components.push_back(visited_triangles);
    }
}


// --------------------------------------------------------
///
/// Duplicate boundary vertex (or vertices) and move the copies away from each other slightly
///
// --------------------------------------------------------


bool MeshCutter::pull_apart_edge( size_t edge_index, const std::vector< TriangleSet >& connected_components )
{
    
    Vec2st edge_data = m_surf.m_mesh.m_edges[edge_index];
    size_t vtx0 = edge_data[0];
    size_t vtx1 = edge_data[1];
    bool bnd0 = m_surf.m_mesh.m_is_boundary_vertex[vtx0];
    bool bnd1 = m_surf.m_mesh.m_is_boundary_vertex[vtx1];
    
    //ensure this is an interior edge, with one or more splitting vertex on the boundary
    assert(bnd0 || bnd1);
    assert(!m_surf.m_mesh.m_is_boundary_edge[edge_index]);  // require the edge to be interior, otherwise splitting makes no sense
    assert(m_surf.m_mesh.m_edge_to_triangle_map[edge_index].size() == 2); // require manifold geometry for now.
    
    double dx = 10.0 * m_surf.m_proximity_epsilon;
    
    TriangleSet triangles_to_delete;
    std::vector< Vec3st > triangles_to_add;
    std::vector< size_t > vertices_added;
    
    std::vector<size_t> boundary_verts;
    std::vector<size_t> dup_verts;
    std::vector< std::set<size_t> > surrounding_vert_sets;
    
    if(bnd0) boundary_verts.push_back(vtx0);
    if(bnd1) boundary_verts.push_back(vtx1);
    
    // Start preparing the history data
    MeshUpdateEvent cut_event(MeshUpdateEvent::EDGE_CUT);
    cut_event.m_v0 = vtx0;
    cut_event.m_v1 = vtx1;
    cut_event.m_v2 = static_cast<unsigned int>(~0); //flag as unused
    
    bool success = perform_pull_apart(boundary_verts, connected_components, cut_event);
    if(success)
        m_surf.m_mesh_change_history.push_back(cut_event);
    
    return success;
    
}

///--------------------------------------------------------


bool MeshCutter::pull_apart_edge_internal( size_t edge0, size_t edge1, const std::vector< TriangleSet >& connected_components, size_t& new_vert )
{
    
    Vec2st edge_data0 = m_surf.m_mesh.m_edges[edge0];
    Vec2st edge_data1 = m_surf.m_mesh.m_edges[edge1];
    
    size_t shared_vert = m_surf.m_mesh.get_common_vertex(edge0, edge1);
    size_t vtx0 = edge_data0[0] == shared_vert? edge_data0[1] : edge_data0[0];
    size_t vtx1 = edge_data1[0] == shared_vert? edge_data1[1] : edge_data1[0];
    
    //ensure this is a pair of interior edges, with no boundary vertices
    assert(!m_surf.m_mesh.m_is_boundary_vertex[vtx0] &&
           !m_surf.m_mesh.m_is_boundary_vertex[vtx1] &&
           !m_surf.m_mesh.m_is_boundary_vertex[shared_vert]);
    
    // require the edges to be interior, otherwise splitting makes no sense
    assert(!m_surf.m_mesh.m_is_boundary_edge[edge0]);
    assert(!m_surf.m_mesh.m_is_boundary_edge[edge1]);
    
    // require manifold geometry for now.
    assert(m_surf.m_mesh.m_edge_to_triangle_map[edge0].size() == 2);
    assert(m_surf.m_mesh.m_edge_to_triangle_map[edge1].size() == 2);
    
    std::vector<size_t> boundary_verts;
    boundary_verts.push_back(shared_vert);
    
    MeshUpdateEvent cut_event(MeshUpdateEvent::EDGE_CUT);
    cut_event.m_v0 = vtx0;
    cut_event.m_v1 = vtx1;
    cut_event.m_v2 = shared_vert;
    
    bool success = perform_pull_apart(boundary_verts, connected_components, cut_event);
    if(success) {
        m_surf.m_mesh_change_history.push_back(cut_event);
        new_vert = cut_event.m_created_verts[0];
    }
    
    return success;
}



// --------------------------------------------------------
///
/// Actually do the separating of the geometry, if possible
///
// --------------------------------------------------------
bool MeshCutter::perform_pull_apart(const std::vector<size_t>& boundary_verts, const std::vector< TriangleSet >& connected_components, MeshUpdateEvent& history) {
    
    double dx = 10.0 * m_surf.m_proximity_epsilon;
    
    TriangleSet triangles_to_delete;
    std::vector< Vec3st > triangles_to_add;
    std::vector< size_t > vertices_added;
    std::vector<Vec2i> triangle_labels;
    
    std::vector<size_t> dup_verts;
    std::vector< std::set<size_t> > surrounding_vert_sets;
    
    // for each connected component except the last one, create a duplicate vertex
    for ( int i = 0; i < (int)connected_components.size() - 1; ++i )
    {
        // duplicate the vertices
        
        for(size_t bv = 0; bv < boundary_verts.size(); ++bv) {
            size_t duplicate_vertex = m_surf.add_vertex( m_surf.get_position(boundary_verts[bv]), m_surf.m_masses[boundary_verts[bv]] );
            dup_verts.push_back(duplicate_vertex);
            vertices_added.push_back( duplicate_vertex );
            surrounding_vert_sets.push_back( std::set<size_t>() );
            
            history.m_created_verts.push_back(duplicate_vertex);
        }
        
        // map component triangles to the duplicated vertices
        for ( size_t t = 0; t < connected_components[i].size(); ++t )
        {
            // create a new triangle with any boundary vertices replaced with the new duplicate vertex
            Vec3st new_triangle = m_surf.m_mesh.get_triangle( connected_components[i][t] );
            
            for ( unsigned short v = 0; v < 3; ++v )
            {
                
                //replace the boundary vertices with their duplicates
                bool is_boundary_vert = false;
                for(size_t bv = 0; bv < boundary_verts.size(); ++bv) {
                    if ( new_triangle[v] == boundary_verts[bv])
                    {
                        new_triangle[v] = dup_verts[bv];
                    }
                    else {
                        surrounding_vert_sets[bv].insert(new_triangle[v]);
                    }
                }
                
            }
            
            triangles_to_add.push_back( new_triangle );
            triangles_to_delete.push_back( connected_components[i][t] );
            triangle_labels.push_back(m_surf.m_mesh.get_triangle_label(connected_components[i][t]));
        }
        
        for(size_t bv = 0; bv < boundary_verts.size(); ++bv) {
            
            // compute the centroid
            Vec3d centroid( 0, 0, 0 );
            for(std::set<size_t>::iterator it = surrounding_vert_sets[bv].begin(); it != surrounding_vert_sets[bv].end(); ++it) {
                centroid += m_surf.get_position( *it );
            }
            centroid /= (double)(surrounding_vert_sets[bv].size());
            
            // move the duplicate vertex towards the centroid by dx
            
            Vec3d added_vertex_position = (1.0 - dx) * m_surf.get_position(dup_verts[bv]) + dx * centroid;
            
            m_surf.set_position( dup_verts[bv], added_vertex_position );
            m_surf.set_newposition( dup_verts[bv], added_vertex_position );
            
            history.m_created_vert_data.push_back(added_vertex_position);
        }
    }
    
    // check new triangles for collision safety
    
    bool collision_occurs = false;
    
    if ( m_surf.m_collision_safety )
    {
        
        for ( size_t i = 0; i < triangles_to_add.size(); ++i )
        {
            const Vec3st& current_triangle = triangles_to_add[i];
            Vec3d low, high;
            
            minmax( m_surf.get_position(current_triangle[0]), m_surf.get_position(current_triangle[1]), m_surf.get_position(current_triangle[2]), low, high );
            
            std::vector<size_t> overlapping_triangles;
            m_surf.m_broad_phase->get_potential_triangle_collisions( low, high, true, true, overlapping_triangles );
            
            for ( size_t j=0; j < overlapping_triangles.size(); ++j )
            {
                
                //prevent checking against to-be-deleted triangles
                bool go_to_next_triangle = false;
                for ( size_t d = 0; d < triangles_to_delete.size(); ++d )
                {
                    if ( overlapping_triangles[j] == triangles_to_delete[d] )
                    {
                        go_to_next_triangle = true;
                        break;
                    }
                }
                if ( go_to_next_triangle )
                {
                    continue;
                }
                
                const Vec3st& tri_j = m_surf.m_mesh.get_triangle(overlapping_triangles[j]);
                
                assert( tri_j[0] != tri_j[1] );
                
                if ( check_triangle_triangle_intersection( current_triangle, tri_j, m_surf.get_positions() ) )
                {
                    // collision occurs - abort separation
                    collision_occurs = true;
                    break;
                }
            }
            
            if(collision_occurs)
                break;
        }
        
        if(!collision_occurs) {
            // check new triangles vs each other as well
            for ( size_t i = 0; i < triangles_to_add.size(); ++i )
            {
                for ( size_t j = i+1; j < triangles_to_add.size(); ++j )
                {
                    if ( check_triangle_triangle_intersection( triangles_to_add[i], triangles_to_add[j], m_surf.get_positions() ) )
                    {
                        // collision occurs - abort separation
                        collision_occurs = true;
                        break;
                    }
                }
                if(collision_occurs)
                    break;
            }
        }
    }
    
    // abort separation, remove added vertices and return
    
    if ( collision_occurs )
    {
        for ( size_t i = 0; i < vertices_added.size(); ++i )
        {
            m_surf.remove_vertex( vertices_added[i] );
        }
        return false;
    }
    
    // all new triangles check out okay for collision safety. Add them to the data structure.
    
    assert(triangles_to_add.size() == triangle_labels.size());
    
    for ( size_t i = 0; i < triangles_to_add.size(); ++i )
    {
        size_t new_tri = m_surf.add_triangle( triangles_to_add[i], triangle_labels[i] );
        
        history.m_created_tris.push_back(new_tri);
        history.m_created_tri_data.push_back(triangles_to_add[i]);
    }
    
    for ( size_t i = 0; i < triangles_to_delete.size(); ++i )
    {
        m_surf.remove_triangle( triangles_to_delete[i] );
        
        history.m_deleted_tris.push_back( triangles_to_delete[i] );
    }
    
    //m_surf.m_mesh_change_history.push_back(history);
    
    if ( m_surf.m_collision_safety )
    {
        m_surf.assert_mesh_is_intersection_free(false);
    }
    
    if ( m_surf.m_verbose ) { std::cout << "pulled apart a vertex" << std::endl; }
    
    return true;
    
}



// --------------------------------------------------------
///
/// Find vertices with disconnected neighbourhoods, and pull them apart
///
// --------------------------------------------------------

void MeshCutter::separate_edges(const std::vector<std::pair<size_t,size_t> >& edge_set)
{
    
    std::vector< std::pair<size_t,size_t> > boundary_edges;
    
    std::set<size_t> boundary_edges_to_split;
    
    for ( size_t i = 0; i < edge_set.size(); ++i )
    {
        // Grab the two end vertices, find the edge
        std::pair<size_t,size_t> vert_pair = edge_set[i];
        size_t vtx0 = vert_pair.first;
        size_t vtx1 = vert_pair.second;
        size_t edge = m_surf.m_mesh.get_edge_index(vtx0, vtx1);
        
        //if at least one of the two end vertices is a boundary vertex, and this isn't a boundary edge,
        //we will want to process the cut
        if((m_surf.m_mesh.m_is_boundary_vertex[vtx0] || m_surf.m_mesh.m_is_boundary_vertex[vtx1]) & !m_surf.m_mesh.m_is_boundary_edge[edge]) {
            boundary_edges.push_back(vert_pair);
        }
    }
    
    
    for ( size_t i = 0; i < boundary_edges.size(); ++i )
    {
        
        // Extract the data
        std::pair<size_t,size_t> verts = boundary_edges[i];
        size_t vtx0 = verts.first;
        size_t vtx1 = verts.second;
        size_t edge = m_surf.m_mesh.get_edge_index(vtx0, vtx1);
        
        //skip already-boundary edges that somehow got added.
        if(m_surf.m_mesh.m_is_boundary_edge[edge]) continue;
        
        int bdry_count = 0;
        bdry_count += m_surf.m_mesh.m_is_boundary_vertex[vtx0]?1:0;
        bdry_count += m_surf.m_mesh.m_is_boundary_vertex[vtx1]?1:0;
        
        // Partition the set of triangles adjacent to this vertex into connected components
        std::vector< TriangleSet > connected_components;
        
        std::cout << "Partitioning edges.\n";
        partition_edge_neighbourhood( edge, connected_components );
        std::cout << "Partitioned successfully.\n";
        std::cout << "Connected components:";
        for(unsigned int j = 0; j < connected_components.size(); ++j) {
            std::cout << "Component " << j << ": ";
            for(unsigned int k = 0; k < connected_components[j].size(); ++k) {
                std::cout << connected_components[j][k] << " ";
            }
            std::cout << std::endl;
        }
        
        if ( connected_components.size() > 1 )
        {
            std::cout << "Doing the cutting itself.\n";
            
            bool cut = pull_apart_edge( edge, connected_components );
            if ( cut )
            {
                // TODO: Shouldn't need this.
                m_surf.rebuild_continuous_broad_phase();
            }
        }
    }
}


void MeshCutter::separate_edges_new(const std::vector<std::pair<size_t,size_t> >& edge_set)
{
    
    std::vector< std::pair<size_t,size_t> > boundary_edges;
    
    std::vector<size_t> edges_to_split;
    std::set<size_t> edges_to_split_set;
    for ( size_t i = 0; i < edge_set.size(); ++i )
    {
        // Grab the two end vertices, find the edge
        std::pair<size_t,size_t> vert_pair = edge_set[i];
        size_t vtx0 = vert_pair.first;
        size_t vtx1 = vert_pair.second;
        size_t edge = m_surf.m_mesh.get_edge_index(vtx0, vtx1);
        
        assert(!m_surf.m_mesh.m_is_boundary_edge[edge]); //user shouldn't pass in boundary edges for splitting!
        
        edges_to_split.push_back(edge);
        edges_to_split_set.insert(edge);
    }
    
    //std::cout << "Edge set contains:" << edges_to_split.size() << std::endl;
    
    std::set<size_t> boundary_edges_to_split;
    std::set< std::pair<size_t,size_t> > internal_edge_pairs_to_split;
    
    for ( std::vector<size_t>::iterator it = edges_to_split.begin(); it != edges_to_split.end(); ++it )
    {
        // Grab the two end vertices
        size_t edge = *it;
        size_t vtx0 = m_surf.m_mesh.m_edges[edge][0];
        size_t vtx1 = m_surf.m_mesh.m_edges[edge][1];
        
        //skip already-boundary edges
        if(m_surf.m_mesh.m_is_boundary_edge[edge])
            continue;
        
        //if at least one of the two end vertices is a boundary vertex, this is an easy edge cut to handle
        if((m_surf.m_mesh.m_is_boundary_vertex[vtx0] || m_surf.m_mesh.m_is_boundary_vertex[vtx1])) {
            boundary_edges_to_split.insert(edge);
        }
        else {
            //Search for another splitting edge that shares a vertex with this one.
            
            //This is n^2 for now. For massive amounts of tearing, it could perhaps be a bottleneck,
            //in which case, better to split as many easy boundary edges as possible before wasting time
            //searching for internal cuts, since internal cuts often become boundary cuts during the process
            
            for(std::vector<size_t>::iterator it2 = it+1; it2 != edges_to_split.end(); ++it2) {
                size_t other_edge = *it2;
                size_t vtx2 = m_surf.m_mesh.m_edges[other_edge][0];
                size_t vtx3 = m_surf.m_mesh.m_edges[other_edge][1];
                
                //don't bother with edges who themselves have boundary verts, since they're easy.
                if(m_surf.m_mesh.m_is_boundary_vertex[vtx2] || m_surf.m_mesh.m_is_boundary_vertex[vtx3])
                    continue;
                
                assert(!m_surf.m_mesh.m_is_boundary_edge[other_edge]);
                
                if(vtx2 == vtx0 || vtx2 == vtx1 || vtx3 == vtx0 || vtx3 == vtx1) {
                    internal_edge_pairs_to_split.insert(std::make_pair(edge, other_edge));
                }
            }
            
            
        }
    }
    
    while(boundary_edges_to_split.size() > 0 || internal_edge_pairs_to_split.size() > 0) {
        
        //prefer boundary-incident edges, since they generate more of themselves
        
        if(boundary_edges_to_split.size() > 0) {
            
            //process a boundary edge
            size_t edge = *(boundary_edges_to_split.begin());
            boundary_edges_to_split.erase(edge);
            
            //don't process if this is already a boundary edge or dead edge - they can't be split
            if(m_surf.m_mesh.edge_is_deleted(edge)) continue;
            if(m_surf.m_mesh.m_is_boundary_edge[edge]) continue;
            
            size_t vtx0 = m_surf.m_mesh.m_edges[edge][0];
            size_t vtx1 = m_surf.m_mesh.m_edges[edge][1];
            
            bool bound0 = m_surf.m_mesh.m_is_boundary_vertex[vtx0];
            bool bound1 = m_surf.m_mesh.m_is_boundary_vertex[vtx1];
            
            int bdry_count = 0;
            bdry_count += bound0?1:0;
            bdry_count += bound1?1:0;
            
            std::vector<size_t> potential_new_boundary_edges;
            if(!bound0)
                potential_new_boundary_edges.insert(potential_new_boundary_edges.end(),
                                                    m_surf.m_mesh.m_vertex_to_edge_map[vtx0].begin(),
                                                    m_surf.m_mesh.m_vertex_to_edge_map[vtx0].end());
            if(!bound1)
                potential_new_boundary_edges.insert(potential_new_boundary_edges.end(),
                                                    m_surf.m_mesh.m_vertex_to_edge_map[vtx1].begin(),
                                                    m_surf.m_mesh.m_vertex_to_edge_map[vtx1].end());
            
            assert(bdry_count > 0);
            
            // Partition the set of triangles adjacent to this vertex into connected components
            std::vector< TriangleSet > connected_components;
            
            partition_edge_neighbourhood( edge, connected_components );
            
            if ( connected_components.size() > 1 ) 
            {
                
                bool cut = pull_apart_edge( edge, connected_components );
                if ( cut )
                {
                    
                    //check which incident edges are now more easily cut-able, and add them to the boundary list.
                    for(unsigned int q = 0; q < potential_new_boundary_edges.size(); ++q) {
                        size_t new_boundary_edge = potential_new_boundary_edges[q];
                        size_t vtxA = m_surf.m_mesh.m_edges[new_boundary_edge][0];
                        size_t vtxB = m_surf.m_mesh.m_edges[new_boundary_edge][1];
                        if((m_surf.m_mesh.m_is_boundary_vertex[vtxA] || m_surf.m_mesh.m_is_boundary_vertex[vtxB]) && 
                           edges_to_split_set.find(new_boundary_edge) != edges_to_split_set.end())
                            boundary_edges_to_split.insert(new_boundary_edge);
                    }
                    
                    // TODO: Shouldn't need this.
                    m_surf.rebuild_continuous_broad_phase();
                }
            }
            
        }
        else if(internal_edge_pairs_to_split.size() > 0) {
            
            //process an internal edge pair
            std::pair<size_t, size_t> edge_pair = *(internal_edge_pairs_to_split.begin());
            internal_edge_pairs_to_split.erase(edge_pair);
            
            size_t edge0 = edge_pair.first;
            size_t edge1 = edge_pair.second;
            
            if(m_surf.m_mesh.m_is_boundary_edge[edge0] || m_surf.m_mesh.m_is_boundary_edge[edge1])
                continue; //cannot process this scenario. (May have been introduced by other cuts)
            
            //skip dead edges
            if(m_surf.m_mesh.edge_is_deleted(edge0) || m_surf.m_mesh.edge_is_deleted(edge1)) continue;
            
            size_t shared_vert = m_surf.m_mesh.get_common_vertex(edge0, edge1);
            Vec2st edge_data0 = m_surf.m_mesh.m_edges[edge0];
            Vec2st edge_data1 = m_surf.m_mesh.m_edges[edge1];
            size_t vtx0 = edge_data0[0] == shared_vert? edge_data0[1] : edge_data0[0]; 
            size_t vtx1 = edge_data1[0] == shared_vert? edge_data1[1] : edge_data1[0]; 
            
            if(m_surf.m_mesh.m_is_boundary_vertex[vtx0] || 
               m_surf.m_mesh.m_is_boundary_vertex[vtx1] || 
               m_surf.m_mesh.m_is_boundary_vertex[shared_vert]) {
                continue; //cannot process this scenario. (May have been introduced by other cuts)
            }
            std::vector< TriangleSet > connected_components;
            
            partition_edge_neighbourhood_internal( edge0, edge1, connected_components );
            
            if ( connected_components.size() > 1 ) 
            {
                size_t new_vert;
                bool cut = pull_apart_edge_internal( edge0, edge1, connected_components, new_vert );
                if ( cut )
                {
                    
                    std::vector<size_t> incident_verts;
                    incident_verts.push_back(shared_vert);
                    incident_verts.push_back(vtx0);
                    incident_verts.push_back(vtx1);
                    incident_verts.push_back(new_vert);
                    
                    //Add all the other edges incident on any of this section's vertices
                    
                    for(unsigned int q = 0; q < incident_verts.size(); ++q) {
                        size_t vert = incident_verts[q];
                        for(unsigned int r = 0; r < m_surf.m_mesh.m_vertex_to_edge_map[vert].size(); ++r) {
                            size_t new_boundary_edge = m_surf.m_mesh.m_vertex_to_edge_map[vert][q];
                            
                            if(edges_to_split_set.find(new_boundary_edge) != edges_to_split_set.end() &&
                               !m_surf.m_mesh.m_is_boundary_edge[new_boundary_edge]) {
                                boundary_edges_to_split.insert(new_boundary_edge);
                            }
                        }
                    }
                    
                    // TODO: Shouldn't need this.
                    m_surf.rebuild_continuous_broad_phase();
                }
            }
            
        }
        else {
            //Should never get here.
            assert(false);
        }
        
    } 
    
}
    
}




