// ---------------------------------------------------------
//
//  meshcutter.h
//  Christopher Batty, Fang Da 2014
//  
//  Separates mesh sections along a given set of existing edges, so
//  long as it does not introduce new collisions.
//
// ---------------------------------------------------------

#ifndef LOSTOPOS_MESHCUTTER_H
#define LOSTOPOS_MESHCUTTER_H

// ---------------------------------------------------------
//  Nested includes
// ---------------------------------------------------------

#include <cstddef>
#include <vector>

// ---------------------------------------------------------
//  Forwards and typedefs
// ---------------------------------------------------------

namespace LosTopos {

class SurfTrack;
struct MeshUpdateEvent;
template<unsigned int N, class T> struct Vec;
typedef Vec<3,size_t> Vec3st;

// ---------------------------------------------------------
//  Class definitions
// ---------------------------------------------------------

// ---------------------------------------------------------
///
/// Mesh cutter object.  Splits the mesh along a desired set of edges, if possible.
///
// ---------------------------------------------------------

class MeshCutter
{
    
public:
    
    /// Save some typing when dealing with vertex neighbourhoods
    ///
    typedef std::vector<size_t> TriangleSet;
    
    /// Constructor
    /// 
    MeshCutter( SurfTrack& surf ) :
    m_surf( surf )
    {}
    
    /// Given a list of edges, try to separate them
    ///
    void separate_edges(const std::vector<std::pair<size_t,size_t> >& edge_set);
    void separate_edges_new(const std::vector<std::pair<size_t,size_t> >& edge_set);

private:

    /// The mesh this object operates on
    /// 
    SurfTrack& m_surf;
    
    /// Partition the triangles incident to an edge into connected components, 
    /// separated by the specified edge (which must have at least one vertex on the boundary).
    ///
    void partition_edge_neighbourhood( size_t edge_index, std::vector< TriangleSet >& connected_components );
    
    /// Partition the triangles incident to an internal vertex into connected components, 
    /// separated by two connected internal edges.
    ///
    void partition_edge_neighbourhood_internal( size_t edge0, size_t edge1, std::vector< TriangleSet >& connected_components );
    
    /// Helper function to partition a set of triangles into connected components separated
    /// by a given list of edges
    ///
    void perform_partitioning(const std::vector<size_t>& incident_tris, const std::vector<size_t>& separating_edges, std::vector< TriangleSet >& connected_components);

    /// Duplicate one or two boundary vertices of an edge, and move the copies away from each other slightly
    ///
    bool pull_apart_edge( size_t edge_index, const std::vector< TriangleSet >& connected_components );
    bool pull_apart_edge_internal( size_t edge0, size_t edge1, const std::vector< TriangleSet >& connected_components, size_t& new_vert );

    /// Helper function to do the separation, given the connected components
    ///
    bool perform_pull_apart(const std::vector<size_t>& separating_verts, const std::vector< TriangleSet >& connected_components, MeshUpdateEvent& history);

    

};

}

#endif //MESHCUTTER_H
