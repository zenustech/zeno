// ---------------------------------------------------------
//
//  edgecollapser.h
//  Tyson Brochu 2011
//  Christopher Batty, Fang Da 2014
//
//  Functions supporting the "edge collapse" operation: removing short edges from the mesh.
//
// ---------------------------------------------------------

#ifndef LOSTOPOS_EDGECOLLAPSER_H
#define LOSTOPOS_EDGECOLLAPSER_H

// ---------------------------------------------------------
//  Nested includes
// ---------------------------------------------------------

#include <cstddef>
#include <vector>
#include <vec.h>

// ---------------------------------------------------------
//  Forwards and typedefs
// ---------------------------------------------------------

namespace LosTopos {

class SurfTrack;
template<unsigned int N, class T> struct Vec;
typedef Vec<3,double> Vec3d;
typedef Vec<3,size_t> Vec3st;

// ---------------------------------------------------------
//  Class definitions
// ---------------------------------------------------------

// ---------------------------------------------------------
///
/// Edge collapser object.  Removes edges smaller than the specified threshold, optionally scaling measures by mean curvature.
///
// ---------------------------------------------------------

class EdgeCollapser
{
    
public:
    
    /// Edge collapser constructor.  Takes a SurfTrack object and curvature-adaptive parameters.
    ///
    EdgeCollapser( SurfTrack& surf, bool use_curvature, bool remesh_boundaries, double min_curvature_multiplier );

    /// Collapse all short edges
    ///
    bool collapse_pass();
    
    
//    /// Minimum edge length.  Edges shorter than this will be collapsed.
//    ///
//    double m_min_edge_length;  
//
//    /// Maximum edge length.  Edges longer than this will not be collapsed.
//    ///
//    double m_max_edge_length; 
    
    /// Whether to scale by curvature when computing edge lengths, in order to coarsen low-curvature regions
    ///
    bool m_use_curvature;
    
    /// Whether to perform remeshing on mesh boundary edges (in the case of open surfaces, e.g. sheets)
    ///
    bool m_remesh_boundaries;

    /// The minimum curvature scaling allowed
    ///
    double m_min_curvature_multiplier;
    
    /// The region label to consider when computing ranks
    /// (default = -1, means consider all regions.)
    int m_rank_region;
  
    /// t1 pull apart distance
    double m_t1_pull_apart_distance;

private:
    
    friend class SurfTrack;
    
    /// The mesh this object operates on
    /// 
    SurfTrack& m_surf;

    /// Get all triangles which are incident on either edge end vertex.
    ///
    void get_moving_triangles(size_t source_vertex, 
                              size_t destination_vertex, 
                              std::vector<size_t>& moving_triangles );
    
    
    /// Get all edges which are incident on either edge end vertex.
    ///
    void get_moving_edges(size_t source_vertex, 
                          size_t destination_vertex, 
                          size_t edge_index,
                          std::vector<size_t>& moving_edges );
    
    /// Check the "pseudo motion" introduced by a collapsing edge for collision
    ///
    bool collapse_edge_pseudo_motion_introduces_collision( size_t source_vertex, 
                                                          size_t destination_vertex, 
                                                          size_t edge_index, 
                                                          const Vec3d& vertex_new_position );
    
    /// Determine if the edge collapse operation would invert the normal of any incident triangles.
    ///
    bool collapse_edge_introduces_normal_inversion( size_t source_vertex, 
                                                   size_t destination_vertex, 
                                                   size_t edge_index, 
                                                   const Vec3d& vertex_new_position );
    
    /// Determine whether collapsing an edge will introduce an unacceptable change in volume.
    ///
    bool collapse_edge_introduces_volume_change( size_t source_vertex, 
                                                size_t edge_index, 
                                                const Vec3d& vertex_new_position );   
    
    /// Returns true if the edge collapse would introduce a triangle with a min or max angle outside of the specified min or max.
    ///
    bool collapse_edge_introduces_bad_angle( size_t source_vertex, 
                                            size_t destination_vertex, 
                                            const Vec3d& vertex_new_position);

    /// Test if the result of a collapse will contain an irregular vertex that t1 may resolve
    ///
    bool collapse_will_produce_irregular_junction(size_t edge);
    
    /// Choose the vertex to keep and delete, and the remaining vertices' position.
    /// Return false if the edge turns out not to be collapsible
    ///
    bool get_new_vertex_position(Vec3d& new_vertex_position, size_t& vert_to_keep, size_t& vert_to_delete, const size_t& edge, bool& new_vert_solid_label);

    //Experimental edge collapser that decides features based on edge dihedral angles.
    bool get_new_vertex_position_dihedral(Vec3d& new_vertex_position, size_t& vert_to_keep, size_t& vert_to_delete, const size_t& edge, Vec3c& new_vert_solid_label);


public:
    /// Delete an edge by moving its source vertex to its destination vertex
    ///
    bool collapse_edge( size_t edge , bool force_verbose=false, bool no_really_collapse=false);
    
    /// Determine if the edge should be allowed to collapse
    ///
    bool edge_is_collapsible( size_t edge_index, double& cur_length );
    
};

}

#endif


