// ---------------------------------------------------------
//
//  meshpincher.h
//  Tyson Brochu 2011
//  Christopher Batty, Fang Da 2014
//
//  Identifies "singular vertices", defined as having more than one connected triangle neighbourhoods, and
//  splits the mesh surface at these vertices.
//
// ---------------------------------------------------------

#ifndef LOSTOPOS_MESHPINCHER_H
#define LOSTOPOS_MESHPINCHER_H

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
template<unsigned int N, class T> struct Vec;
typedef Vec<3,size_t> Vec3st;

// ---------------------------------------------------------
//  Class definitions
// ---------------------------------------------------------

// ---------------------------------------------------------
///
/// Mesh pincher object.  Identifies "singular vertices", defined as having more than one connected triangle neighbourhoods, and
/// splits the mesh surface at these vertices.
///
// ---------------------------------------------------------

class MeshPincher
{
    
public:
    
    /// Save some typing when dealing with vertex neighbourhoods
    ///
    typedef std::vector<size_t> TriangleSet;
    
    /// Constructor
    /// 
    MeshPincher( SurfTrack& surf ) :
    m_surf( surf )
    {}
    
    /// Find vertices with disconnected neighbourhoods, and pull them apart
    ///
    void separate_singular_vertices();
    
private:

    /// The mesh this object operates on
    /// 
    SurfTrack& m_surf;
    
    /// Partition the triangles incident to a vertex into connected components
    ///
    void partition_vertex_neighbourhood( size_t vertex_index, std::vector< TriangleSet >& connected_components );
    
    /// Duplicate a vertex and move the two copies away from each other slightly
    ///
    bool pull_apart_vertex( size_t vertex_index, const std::vector< TriangleSet >& connected_components );
    
};

}

#endif
