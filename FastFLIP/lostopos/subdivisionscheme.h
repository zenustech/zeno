// ---------------------------------------------------------
//
//  subdivisionscheme.h
//  Tyson Brochu 2008
//  Christopher Batty, Fang Da 2014
//
//  A collection of interpolation schemes for generating vertex locations.
//
// ---------------------------------------------------------

#ifndef LOSTOPOS_SUBDIVISIONSCHEME_H
#define LOSTOPOS_SUBDIVISIONSCHEME_H

// ---------------------------------------------------------
// Nested includes
// ---------------------------------------------------------

#include <vec.h>

// ---------------------------------------------------------
//  Forwards and typedefs
// ---------------------------------------------------------

namespace LosTopos {

class SurfTrack;
class NonDestructiveTriMesh;

// ---------------------------------------------------------
//  Class definitions
// ---------------------------------------------------------

// --------------------------------------------------------
///
/// Subdivision scheme interface.  Declares the function prototype for finding an interpolated vertex location.
///
// --------------------------------------------------------

class SubdivisionScheme
{
public:
    
    virtual ~SubdivisionScheme() {}
    
    /// Given an edge, compute the offset midpoint
    ///
    virtual void generate_new_midpoint( size_t edge_index, const SurfTrack& surface, Vec3d& new_point ) = 0;
    
};

// --------------------------------------------------------
///
/// Midpoint scheme: simply places the new vertex at the midpoint of the edge
///
// --------------------------------------------------------

class MidpointScheme : public SubdivisionScheme
{
public:
    
    /// Given an edge, compute the offset midpoint
    ///
    void generate_new_midpoint( size_t edge_index, const SurfTrack& surface, Vec3d& new_point );   
    
};

// --------------------------------------------------------
///
/// Butterfly scheme: uses a defined weighting of nearby vertices to determine the new vertex location
///
// --------------------------------------------------------

class ButterflyScheme : public SubdivisionScheme
{
public:  
    
    /// Given an edge, compute the offset midpoint
    ///
    void generate_new_midpoint( size_t edge_index, const SurfTrack& surface, Vec3d& new_point );
    
};


// --------------------------------------------------------
///
/// Quadric error minimization scheme: places the new vertex at the location that minimizes the change in the quadric metric tensor along the edge.
///
// --------------------------------------------------------

class QuadraticErrorMinScheme : public SubdivisionScheme
{
public:
    
    /// Given an edge, compute the offset midpoint
    ///
    void generate_new_midpoint( size_t edge_index, const SurfTrack& surface, Vec3d& new_point );
    
};


// --------------------------------------------------------
///
/// Modified Butterfly scheme: uses the method of Zorin et al. to generate a new vertex, for meshes with arbitrary topology
///
// --------------------------------------------------------

class ModifiedButterflyScheme : public SubdivisionScheme
{
    
public:  

    ModifiedButterflyScheme();

    /// Given an edge, compute the offset midpoint
    ///
    void generate_new_midpoint( size_t edge_index, const SurfTrack& surface, Vec3d& new_point );

    std::vector<std::vector<double> > weights;
};


}

#endif



