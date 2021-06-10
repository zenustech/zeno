// ---------------------------------------------------------
//
//  subdivisionscheme.cpp
//  Tyson Brochu 2008
//  Christopher Batty, Fang Da 2014
//
//  A collection of interpolation schemes for generating vertex locations.
//
// ---------------------------------------------------------

// ---------------------------------------------------------
// Includes
// ---------------------------------------------------------

#include <subdivisionscheme.h>

#include <mat.h>
#include <surftrack.h>

// ---------------------------------------------------------
// Global externs
// ---------------------------------------------------------

// ---------------------------------------------------------
// Local constants, typedefs, macros
// ---------------------------------------------------------

// ---------------------------------------------------------
// Static function definitions
// ---------------------------------------------------------

// ---------------------------------------------------------
// Member function definitions
// ---------------------------------------------------------

// --------------------------------------------------------
///
/// Midpoint scheme: simply places the new vertex at the midpoint of the edge
///
// --------------------------------------------------------

namespace LosTopos {

void MidpointScheme::generate_new_midpoint( size_t edge_index, const SurfTrack& surface, Vec3d& new_point )
{
    const NonDestructiveTriMesh& mesh = surface.m_mesh;
    const std::vector<Vec3d>& positions = surface.get_positions();
    size_t p1_index = mesh.m_edges[edge_index][0];
	size_t p2_index = mesh.m_edges[edge_index][1];   
    
    new_point = 0.5 * ( positions[ p1_index ] + positions[ p2_index ] );
}


// --------------------------------------------------------
///
/// Butterfly scheme: uses a defined weighting of nearby vertices to determine the new vertex location
///
// --------------------------------------------------------

void ButterflyScheme::generate_new_midpoint( size_t edge_index, const SurfTrack& surface, Vec3d& new_point )
{
    const NonDestructiveTriMesh& mesh = surface.m_mesh;
    const std::vector<Vec3d>& positions = surface.get_positions();
    
    size_t p1_index = mesh.m_edges[edge_index][0];
	size_t p2_index = mesh.m_edges[edge_index][1];
	
   //Butterfly doesn't support non-manifold scenarios, really.  We revert to midpoint below.
   //Better to straight-up use midpoint, or go with modified butterfly.

    size_t tri0 = mesh.m_edge_to_triangle_map[edge_index][0];
    size_t tri1 = mesh.m_edge_to_triangle_map[edge_index][1];
    
	size_t p3_index = mesh.get_third_vertex( mesh.m_edges[edge_index][0], mesh.m_edges[edge_index][1], mesh.get_triangle(tri0) );
	size_t p4_index = mesh.get_third_vertex( mesh.m_edges[edge_index][0], mesh.m_edges[edge_index][1], mesh.get_triangle(tri1) );
	
	size_t adj_edges[4] = { mesh.get_edge_index( p1_index, p3_index ),
        mesh.get_edge_index( p2_index, p3_index ),
        mesh.get_edge_index( p1_index, p4_index ),
        mesh.get_edge_index( p2_index, p4_index ) };
    
	size_t q_indices[4];
	
	for ( size_t i = 0; i < 4; ++i )
	{
		const std::vector<size_t>& adj_tris = mesh.m_edge_to_triangle_map[ adj_edges[i] ];
		if ( adj_tris.size() != 2 )
		{
            // abort! revert to midpoint here.
			new_point = 0.5 * ( positions[ p1_index ] + positions[ p2_index ] );
            return;
		}
		
		if ( adj_tris[0] == tri0 || adj_tris[0] == tri1 )
		{
			q_indices[i] = mesh.get_third_vertex( mesh.m_edges[ adj_edges[i] ][0], mesh.m_edges[ adj_edges[i] ][1], mesh.get_triangle( adj_tris[1] ) );
		}
		else
		{
			q_indices[i] = mesh.get_third_vertex( mesh.m_edges[ adj_edges[i] ][0], mesh.m_edges[ adj_edges[i] ][1], mesh.get_triangle( adj_tris[0] ) );
		}
	}
    
	new_point =   8. * positions[ p1_index ] + 8. * positions[ p2_index ] + 2. * positions[ p3_index ] + 2. * positions[ p4_index ]
    - positions[ q_indices[0] ] - positions[ q_indices[1] ] - positions[ q_indices[2] ] - positions[ q_indices[3] ];
    
	new_point *= 0.0625;
    
}



// --------------------------------------------------------
///
/// Quadric error minimization scheme: places the new vertex at the location that minimizes the change in the quadric metric tensor along the edge.
///
// --------------------------------------------------------

void QuadraticErrorMinScheme::generate_new_midpoint( size_t edge_index, const SurfTrack& surface, Vec3d& new_point )
{
    const NonDestructiveTriMesh& mesh = surface.m_mesh;
    const std::vector<Vec3d>& positions = surface.get_positions();
    
    size_t v0 = mesh.m_edges[edge_index][0];
    size_t v1 = mesh.m_edges[edge_index][1];
    
    Mat33d Q;
    zero(Q);
    Vec3d b;
    zero(b);
    
    std::vector<size_t> triangles_counted;
    
    Mat<1,1,double> constant_dist;
    constant_dist.a[0] = 0;
    
    for ( size_t i = 0; i < mesh.m_vertex_to_triangle_map[v0].size(); ++i )
    {
        size_t t = mesh.m_vertex_to_triangle_map[v0][i];
        const Vec3d& plane_normal = surface.get_triangle_normal( t );
        Q += outer( plane_normal, plane_normal );
        b += dot( positions[v0], plane_normal ) * plane_normal;
        constant_dist.a[0] += dot( plane_normal, positions[v0] ) * dot( plane_normal, positions[v0] );
        triangles_counted.push_back(t);
    }
    
    for ( size_t i = 0; i < mesh.m_vertex_to_triangle_map[v1].size(); ++i )
    {
        size_t t = mesh.m_vertex_to_triangle_map[v1][i];
        
        bool already_counted = false;
        for ( size_t j = 0; j < triangles_counted.size(); ++j ) 
        {
            if ( t == triangles_counted[j] )
            {
                already_counted = true;
            }
        }
        
        if ( !already_counted )
        {
            const Vec3d& plane_normal = surface.get_triangle_normal( t );
            Q += outer( plane_normal, plane_normal );
            b += dot( positions[v1], plane_normal ) * plane_normal;
            constant_dist.a[0] += dot( plane_normal, positions[v1] ) * dot( plane_normal, positions[v1] );
        }
    }
    
    // Compute normal direction
    Vec3d normal = 0.5 * (surface.get_vertex_normal(v0) + surface.get_vertex_normal(v1));
    normalize(normal);
    
    Mat<3,1,double> n;
    n(0,0) = normal[0];
    n(1,0) = normal[1];
    n(2,0) = normal[2];
    
    // Compute edge midpoint
    Vec3d midpoint = 0.5 * (positions[v0] + positions[v1]);   
    Mat<3,1,double> m;
    m(0,0) = midpoint[0];
    m(1,0) = midpoint[1];
    m(2,0) = midpoint[2]; 
    
    Mat<3,1,double> d;
    d(0,0) = b[0];
    d(1,0) = b[1];
    d(2,0) = b[2];
    
    double LHS = 2.0 * (n.transpose()*Q*n).a[0];              // result of multiplication is Mat<1,1,double>, hence the .a[0]
    double RHS = ( 2.0 * (n.transpose()*d) - (n.transpose()*Q*m) - (m.transpose()*Q*n) ).a[0];
    
    double a;
    if ( std::fabs(LHS) > 1e-10 )
    {
        a = RHS / LHS;
    }
    else
    {
        a = 0.0;
    }
    
    Mat<3,1,double> v = m + (a * n);
    
    double v_error = (v.transpose() * Q * v - 2.0 * (v.transpose() * d) + constant_dist).a[0];
    double m_error = (m.transpose() * Q * m - 2.0 * (m.transpose() * d) + constant_dist).a[0];
    
    //assert( v_error < m_error + 1e-8 );
    
    if ( surface.m_verbose )
    {
        std::cout << "a: " << a << std::endl;
        std::cout << "error at v: " << v_error << std::endl;
        std::cout << "error at midpoint: " << m_error << std::endl;
    }
    
    new_point = Vec3d( v.a[0], v.a[1], v.a[2] );
    
}

// --------------------------------------------------------
///
/// Modified Butterfly scheme: uses the method of Zorin et al. to generate a new vertex, for meshes with arbitrary topology
/// Modeled loosely after the implementation in OpenMesh.
/// Extended to treat non-manifold edges as boundary edges.
/// Also to handle sharp feature edges as boundary edges!
// --------------------------------------------------------

ModifiedButterflyScheme::ModifiedButterflyScheme() {
    const int MAX_VALENCE = 100; //precompute up to valence of 100 for good measure.
    weights.resize(100);
    
    //special case: K==3, K==4
    weights[3].resize(4);
    weights[3][0] = 5.0/12.0;
    weights[3][1] = -1.0/12.0;
    weights[3][2] = -1.0/12.0;
    weights[3][3] = 3.0/4.0;
    
    weights[4].resize(5);
    weights[4][0] = 3.0/8.0;
    weights[4][1] = 0;
    weights[4][2] = -1.0/8.0;
    weights[4][3] = 0;
    weights[4][4] = 3.0/4.0;
    
    for(unsigned int K = 5; K <MAX_VALENCE; ++K) {
        weights[K].resize(K+1);
        // s(j) = ( 1/4 + cos(2*pi*j/K) + 1/2 * cos(4*pi*j/K) )/K
        double invK  = 1.0/double(K);
        double sum = 0;
        for(unsigned int j=0; j<K; ++j)
        {
            weights[K][j] = (0.25 + cos(2.0*M_PI*j*invK) + 0.5*cos(4.0*M_PI*j*invK))*invK;
            sum += weights[K][j];
        }
        weights[K][K] = 1.0 - sum;
    }
}

void ModifiedButterflyScheme::generate_new_midpoint( size_t edge_index, const SurfTrack& surface, Vec3d& new_point )
{
   
    const NonDestructiveTriMesh& mesh = surface.m_mesh;
    const std::vector<Vec3d>& positions = surface.get_positions();

    size_t p1_index = mesh.m_edges[edge_index][0];
    size_t p2_index = mesh.m_edges[edge_index][1];

    new_point = Vec3d(0,0,0);
    Vec2st edge_data = mesh.m_edges[edge_index];
    
    //TODO Eliminate redundancy in feature/boundary/non-manifold curve subdivision. They all use the same cubic curve for subdivision,
    //but just choose the next/previous vertex sets differently.

    //require internal edge and not a triple-junction for standard subdivision
    if(!mesh.m_is_boundary_edge[edge_index] && !mesh.is_edge_nonmanifold(edge_index) && !surface.edge_is_feature(edge_index)) { 
        
        int valence_p1 = (int)mesh.m_vertex_to_edge_map[p1_index].size();
        int valence_p2 = (int)mesh.m_vertex_to_edge_map[p2_index].size();

        //TODO: Watch out for case where the valence is 6, but is non-manifold and consists 
        //of two distinct neighborhoods -> should probably be treated as irregular.

        if(   (valence_p1 == 6 || mesh.m_is_boundary_vertex[p1_index] || mesh.is_vertex_incident_on_nonmanifold_edge(p1_index) || surface.vertex_feature_edge_count(p1_index) > 0) 
           && (valence_p2 == 6 || mesh.m_is_boundary_vertex[p2_index] || mesh.is_vertex_incident_on_nonmanifold_edge(p2_index) || surface.vertex_feature_edge_count(p2_index) > 0)) {
            
            const double alpha    = 1.0/2.0;
            const double beta     = 1.0/8.0;
            const double gamma    = -1.0/16.0;

            size_t tri0 = mesh.m_edge_to_triangle_map[edge_index][0];
            size_t tri1 = mesh.m_edge_to_triangle_map[edge_index][1];

            size_t p3_index = mesh.get_third_vertex( p1_index, p2_index, mesh.get_triangle(tri0) );
            size_t p4_index = mesh.get_third_vertex( p1_index, p2_index, mesh.get_triangle(tri1) );

            size_t adj_edges[4] = { mesh.get_edge_index( p1_index, p3_index ),
                mesh.get_edge_index( p2_index, p3_index ),
                mesh.get_edge_index( p1_index, p4_index ),
                mesh.get_edge_index( p2_index, p4_index ) };

            Vec3d surround_vert_sum(0,0,0);

            for ( size_t i = 0; i < 4; ++i )
            {
                size_t cur_edge_ind = adj_edges[i];
                Vec2st cur_edge = mesh.m_edges[cur_edge_ind];
                const std::vector<size_t>& adj_tris = mesh.m_edge_to_triangle_map[ cur_edge_ind ];
                if(mesh.m_is_boundary_edge[cur_edge_ind] || mesh.is_edge_nonmanifold(cur_edge_ind) || surface.edge_is_feature(cur_edge_ind)) {
                    //create a vertex by reflection of the vertices in the adjacent triangle
                   
                   //for the non-manifold or feature case, make sure the tri we grab is one of the original edge wings
                    int which_tri = 0;
                    while(!mesh.triangle_contains_edge(mesh.m_tris[adj_tris[which_tri]], edge_data))
                       ++which_tri;
                    
                    size_t third_vert = mesh.get_third_vertex(cur_edge_ind, adj_tris[which_tri]);
                    Vec3d reflected_point = surface.get_position(cur_edge[0]) +
                                            surface.get_position(cur_edge[1]) - 
                                            surface.get_position(third_vert);
                    surround_vert_sum += reflected_point;
                }
                else {
                    //grab the appropriate vertex
                    Vec3st cur_tri = (adj_tris[0] == tri0 || adj_tris[0] == tri1) ? 
                        mesh.get_triangle( adj_tris[1] ) : 
                        mesh.get_triangle( adj_tris[0] );
                    size_t vert = mesh.get_third_vertex( cur_edge[0], cur_edge[1], cur_tri );
                    surround_vert_sum += surface.get_position(vert);
                }
            }
            //Note: this implements the standard w=0 case described in the paper, which avoids the need for the
            //9th and 10th vertices (i.e. the outer ones labeled "d" in figure 3).
            new_point = alpha * (surface.get_position(p1_index) + surface.get_position(p2_index)) + 
                        beta * (surface.get_position(p3_index) + surface.get_position(p4_index)) + 
                        gamma * (surround_vert_sum);
        }
        else { //both vertices are either: (a) not 6-valence or (b) not incident on boundary/non-manifold/feature edges
           //i.e. both vertices are manifold, not on the boundary, and have irregular valences

           //apply the irregular stencil
            double norm_factor = 0.0;
            
            //consider each vertex of the main edge
            for(int cur_vertex = 0; cur_vertex < 2; ++cur_vertex) {
                size_t cur_vert_index = mesh.m_edges[edge_index][cur_vertex];
                int cur_valence = (int)mesh.m_vertex_to_edge_map[cur_vert_index].size();
                
                const std::vector<double>& local_weights = weights[cur_valence];

                //if it's irregular and not a boundary vertex or a non-manifold vertex or a feature vertex, process it
                if(cur_valence != 6 && !mesh.m_is_boundary_vertex[cur_vert_index] 
                                    && !mesh.is_vertex_incident_on_nonmanifold_edge(cur_vert_index) 
                                    && surface.vertex_feature_edge_count(cur_vert_index) == 0) {
                    
                    //walk around the surrounding vertices in order (using the triangle connectivity to figure out that ordering)
                    size_t cur_edge = edge_index;
                    const std::vector<size_t>& edge_map = mesh.m_edge_to_triangle_map[edge_index];
                    size_t last_tri = edge_map[0]; //picked arbitrarily to start us in one direction
                    
                    for(int i = 0; i < cur_valence; ++i) {
                        Vec2st edge_data = mesh.m_edges[cur_edge];
                        size_t nbr_vert = edge_data[0] == cur_vert_index ? edge_data[1] : edge_data[0];
                        new_point += local_weights[i] * surface.get_position(nbr_vert);
                    
                        //advance to next edge around the vertex, by grabbing the next tri
                        const std::vector<size_t>& edge_to_tris = mesh.m_edge_to_triangle_map[cur_edge];
                        size_t tri = edge_to_tris[0] == last_tri ? edge_to_tris[1] : edge_to_tris[0];
                        
                        //find the edge of the new tri that shares the central vertex, but is not the same as the previous edge.
                        Vec3st tri_edges = mesh.m_triangle_to_edge_map[tri];
                        size_t next_edge = mesh.ne();
                        for(int j = 0; j < 3; ++j) {
                            size_t edge_candidate = tri_edges[j];
                            Vec2st edge_data = mesh.m_edges[edge_candidate];
                            if(edge_candidate != cur_edge && (edge_data[0] == cur_vert_index || edge_data[1] == cur_vert_index))  {
                                next_edge = edge_candidate;
                                break;
                            }
                        }
                        assert(next_edge < mesh.ne());
                        //advance
                        last_tri = tri;
                        cur_edge = next_edge;
                        
                        assert(!mesh.is_edge_nonmanifold(cur_edge)); //none of the edges we're on should be non-manifold
                        assert(!mesh.m_is_boundary_edge[cur_edge]); //none of the edges we're on should be boundaries
                        assert(!surface.edge_is_feature(cur_edge)); //none of the edges we're on should be features
                    }
                    
                    //add the central vertex too
                    new_point += local_weights[cur_valence] * surface.get_position(cur_vert_index);
                    norm_factor += 1;
                }
                else {
                   //This case is fine. If one is (irregular and not boundary) and one is (regular or boundary),
                   //we apply only the stencil of that side (the irregular side).
                }
            }
            assert(norm_factor > 0);

            //normalize (effectively averages the two if both vertices were irregular)
            new_point /= norm_factor;
            
        }
    }
    else if(surface.edge_is_feature(edge_index)) {
       //sharp feature edge - treat as a curve if possible, exactly like the boundary case

       new_point = 9.0 / 16.0 *(surface.get_position(p1_index) + surface.get_position(p2_index));

       //find the next and previous feature edges, fit a curve, and away you go.
       //if we can't find next/prev feature edges do midpoint subdivision instead.

       //now need to find the subsequent non-manifold edge. if there is more than one, fail out.
       bool found_next_feature = false;
       for(unsigned int i = 0; i < mesh.m_vertex_to_edge_map[p1_index].size(); ++i) {
          size_t nbr_edge = mesh.m_vertex_to_edge_map[p1_index][i];
          if(nbr_edge == edge_index) continue;
          if(surface.edge_is_feature(nbr_edge)) {
             if(!found_next_feature) {
                //grab the other vertex
                size_t other_vert = mesh.m_edges[nbr_edge][0] == p1_index? mesh.m_edges[nbr_edge][1] : mesh.m_edges[nbr_edge][0];
                new_point += -1.0/16.0 * surface.get_position(other_vert);
                found_next_feature = true;
             } else {
                found_next_feature = false;
                break;
             }
          }
       }

       //TODO Treat it like a straight line if there isn't another one?
       if(!found_next_feature) {
          //couldn't find another feature-edge, so fail back to midpoint.
          new_point = 0.5*(surface.get_position(p1_index) + surface.get_position(p2_index));
          return;
       }

       //then find the previous one and do the same
       found_next_feature = false;
       for(unsigned int i = 0; i < mesh.m_vertex_to_edge_map[p2_index].size(); ++i) {
          size_t nbr_edge = mesh.m_vertex_to_edge_map[p2_index][i];
          if(nbr_edge == edge_index) continue;
          if(surface.edge_is_feature(nbr_edge)){
             size_t other_vert = mesh.m_edges[nbr_edge][0] == p2_index? mesh.m_edges[nbr_edge][1] : mesh.m_edges[nbr_edge][0];
             new_point += -1.0/16.0 * surface.get_position(other_vert);
             found_next_feature = true;
          } else {
             found_next_feature = false;
             break;
          }
       }

       //TODO Treat it like a straight line if there isn't another one?
       if(!found_next_feature) {
          //couldn't find another feature-edge, so fail back to midpoint.
          new_point = 0.5*(surface.get_position(p1_index) + surface.get_position(p2_index));
          return;
       }

    }
    else if(mesh.is_edge_nonmanifold(edge_index)) {
       //non-manifold edge - treat as a curve if possible, exactly like the boundary case

       new_point = 9.0 / 16.0 *(surface.get_position(p1_index) + surface.get_position(p2_index));

       //find the next and previous non-manifold edges, fit a curve, and away you go.
       //if this is not just a single smooth curve (i.e. there are > 2 non-manifold edges come out of the
       //adjacent vertex) do midpoint subdivision instead.

       //now need to find the subsequent non-manifold edge. if there is more than one, fail out.
       bool found_next_nmf = false;
       for(unsigned int i = 0; i < mesh.m_vertex_to_edge_map[p1_index].size(); ++i) {
          size_t nbr_edge = mesh.m_vertex_to_edge_map[p1_index][i];
          if(nbr_edge == edge_index) continue;
          if(mesh.is_edge_nonmanifold(nbr_edge)) {
             if(!found_next_nmf) {
                //grab the other vertex
                size_t other_vert = mesh.m_edges[nbr_edge][0] == p1_index? mesh.m_edges[nbr_edge][1] : mesh.m_edges[nbr_edge][0];
                new_point += -1.0/16.0 * surface.get_position(other_vert);
                found_next_nmf = true;
             } else {
                found_next_nmf = false;
                break;
             }
          }
       }

       //TODO Treat it like a straight line if there isn't another one?
       if(!found_next_nmf) {
          //couldn't find another nmf-edge, so fail back to midpoint.
          new_point = 0.5*(surface.get_position(p1_index) + surface.get_position(p2_index));
          return;
       }
       
       //then find the previous one and do the same
       found_next_nmf = false;
       for(unsigned int i = 0; i < mesh.m_vertex_to_edge_map[p2_index].size(); ++i) {
          size_t nbr_edge = mesh.m_vertex_to_edge_map[p2_index][i];
          if(nbr_edge == edge_index) continue;
          if(mesh.is_edge_nonmanifold(nbr_edge)){
            size_t other_vert = mesh.m_edges[nbr_edge][0] == p2_index? mesh.m_edges[nbr_edge][1] : mesh.m_edges[nbr_edge][0];
            new_point += -1.0/16.0 * surface.get_position(other_vert);
            found_next_nmf = true;
          } else {
            found_next_nmf = false;
            break;
          }
       }

       //TODO Treat it like a straight line if there isn't another one?
       if(!found_next_nmf) {
          //couldn't find another nmf-edge, so fail back to midpoint.
          new_point = 0.5*(surface.get_position(p1_index) + surface.get_position(p2_index));
          return;
       }

    }
    else if(mesh.m_is_boundary_edge[edge_index]) { //use the 4 point boundary scheme
        //boundary edge - treat as a curve if possible
        
        new_point = 9.0 / 16.0 *(surface.get_position(p1_index) + surface.get_position(p2_index));
        
        //find the next and previous boundary edges, fit a curve, and away you go.
        //if this is not just a single smooth curve (i.e. there are > 2 boundary edges come out of the
        //adjacent vertex) do midpoint subdivision instead.
        
        //now need to find the subsequent and preceding vertices along the boundary
        bool found_next_nmf = false;
        for(unsigned int i = 0; i < mesh.m_vertex_to_edge_map[p1_index].size(); ++i) {
            size_t nbr_edge = mesh.m_vertex_to_edge_map[p1_index][i];
            if(nbr_edge == edge_index) continue;
            if(mesh.m_is_boundary_edge[nbr_edge]) {
                if(!found_next_nmf) {
                    //grab the other vertex
                    size_t other_vert = mesh.m_edges[nbr_edge][0] == p1_index? mesh.m_edges[nbr_edge][1] : mesh.m_edges[nbr_edge][0];
                    new_point += -1.0/16.0 * surface.get_position(other_vert);
                    found_next_nmf = true;
                } else {
                    found_next_nmf = false;
                    break;
                }
            }
        }
        
        //TODO Treat it like a straight line if there isn't another one?
        if(!found_next_nmf) {
            //couldn't find another nmf-edge, so fail back to midpoint.
            new_point = 0.5*(surface.get_position(p1_index) + surface.get_position(p2_index));
            return;
        }
        
        //then find the previous one and do the same
        found_next_nmf = false;
        for(unsigned int i = 0; i < mesh.m_vertex_to_edge_map[p2_index].size(); ++i) {
            size_t nbr_edge = mesh.m_vertex_to_edge_map[p2_index][i];
            if(nbr_edge == edge_index) continue;
            if(mesh.m_is_boundary_edge[nbr_edge]) {
                size_t other_vert = mesh.m_edges[nbr_edge][0] == p2_index? mesh.m_edges[nbr_edge][1] : mesh.m_edges[nbr_edge][0];
                new_point += -1.0/16.0 * surface.get_position(other_vert);
                found_next_nmf = true;
            } else {
                found_next_nmf = false;
                break;
            }
        }
        
        //TODO Treat it like a straight line if there isn't another one?
        if(!found_next_nmf) {
            //couldn't find another nmf-edge, so fail back to midpoint.
            new_point = 0.5*(surface.get_position(p1_index) + surface.get_position(p2_index));
            return;
        }
        
    }
   

}


}
