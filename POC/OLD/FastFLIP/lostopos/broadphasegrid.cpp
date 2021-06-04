// ---------------------------------------------------------
//
//  broadphasegrid.cpp
//  Tyson Brochu 2008
//  Christopher Batty, Fang Da 2014
//
//  Broad phase collision detection culling using three regular, volumetric grids.
//
// ---------------------------------------------------------

// ---------------------------------------------------------
// Includes
// ---------------------------------------------------------

#include <broadphasegrid.h>
#include <dynamicsurface.h>

// ---------------------------------------------------------
// Member function definitions
// ---------------------------------------------------------

// --------------------------------------------------------
///
/// Construct one grid from the given set of AABBs, using the given length scale as the cell size, with the given padding
///
// --------------------------------------------------------

namespace LosTopos {

void BroadPhaseGrid::build_acceleration_grid( AccelerationGrid& grid, 
                                             std::vector<Vec3d>& xmins, 
                                             std::vector<Vec3d>& xmaxs, 
                                             std::vector<size_t>& indices,
                                             double length_scale, 
                                             double grid_padding )
{
    
    assert( xmaxs.size() == xmins.size() );
    assert( xmins.size() == indices.size() );

    if ( indices.empty() )
    {
        grid.clear();
        return;
    }
    
    Vec3d xmax = xmaxs[0];
    Vec3d xmin = xmins[0];
    double maxdistance = 0;
    
    size_t n = xmins.size();
    for(size_t i = 0; i < n; i++)
    {
        update_minmax(xmins[i], xmin, xmax);
        update_minmax(xmaxs[i], xmin, xmax);
        maxdistance = std::max(maxdistance, mag(xmaxs[i] - xmins[i]));
    }
    
    for(unsigned int i = 0; i < 3; i++)
    {
        xmin[i] -= 2*maxdistance + grid_padding;
        xmax[i] += 2*maxdistance + grid_padding;
    }
    
    Vec3st dims(1,1,1);
    
    const size_t MAX_D = 2000;
    
    if(mag(xmax-xmin) > grid_padding)
    {
        for(unsigned int i = 0; i < 3; i++)
        {
            size_t d = (size_t)std::ceil((xmax[i] - xmin[i])/length_scale);
            
            if(d < 1) d = 1;
            
            if(d > MAX_D) 
            {
                d = MAX_D;
                printf("Warning: Acceleration structure using max dimensions! (MAX_D==%zd)\n",MAX_D);
            }
            
            dims[i] = d;
        }
    }
    
    grid.set(dims, xmin, xmax);
    
    // going backwards from n to 0, so hopefully the grid only has to allocate once
    for( int i = (int)n-1; i >= 0; i-- )
    {
        // don't add inside-out AABBs
        if ( xmins[i][0] > xmaxs[i][0] )  { continue; }
        grid.add_element( indices[i], xmins[i], xmaxs[i]);
    }
}


// --------------------------------------------------------
///
/// Rebuild acceleration grids according to the given triangle mesh
///
// --------------------------------------------------------

void BroadPhaseGrid::update_broad_phase( const DynamicSurface& surface, bool continuous )
{
    
    double grid_scale = surface.get_average_edge_length();
    //TODO Fix problems where the avg edge length drops close to zero
    //leading to huge dimensions for the box.
    //Where can we get a reasonable scale from otherwise?

    // 
    // vertices
    // 
    {
        size_t num_vertices = surface.get_num_vertices();
        
        std::vector<Vec3d> solid_vertex_xmins, solid_vertex_xmaxs;
        std::vector<size_t> solid_vertex_indices;
        std::vector<Vec3d> dynamic_vertex_xmins, dynamic_vertex_xmaxs;
        std::vector<size_t> dynamic_vertex_indices;
        
        for(size_t i = 0; i < num_vertices; i++)
        {
            Vec3d xmin, xmax;
            
            if ( continuous )
            {
                surface.vertex_continuous_bounds( i, xmin, xmax );
            }
            else
            {
                surface.vertex_static_bounds( i, xmin, xmax );
            }
            
            if ( surface.vertex_is_all_solid( i ) )
            {
                solid_vertex_xmins.push_back( xmin );
                solid_vertex_xmaxs.push_back( xmax );
                solid_vertex_indices.push_back( i );
            }
            else
            {
                dynamic_vertex_xmins.push_back( xmin );
                dynamic_vertex_xmaxs.push_back( xmax );
                dynamic_vertex_indices.push_back( i );
            }
        }
        
        build_acceleration_grid( m_solid_vertex_grid,
                                solid_vertex_xmins,
                                solid_vertex_xmaxs,
                                solid_vertex_indices,
                                grid_scale,
                                surface.m_aabb_padding );
        
        build_acceleration_grid( m_dynamic_vertex_grid,
                                dynamic_vertex_xmins,
                                dynamic_vertex_xmaxs,
                                dynamic_vertex_indices,
                                grid_scale,
                                surface.m_aabb_padding );
        
    }
    
    //
    // edges
    //
    {
        size_t num_edges = surface.m_mesh.m_edges.size();
        
        std::vector<Vec3d> solid_edge_xmins, solid_edge_xmaxs;
        std::vector<size_t> solid_edge_indices;
        std::vector<Vec3d> dynamic_edge_xmins, dynamic_edge_xmaxs;
        std::vector<size_t> dynamic_edge_indices;
        
        for(size_t i = 0; i < num_edges; i++)
        {
            Vec3d xmin, xmax;
            
            if ( continuous )
            {
                surface.edge_continuous_bounds( i, xmin, xmax );
            }
            else
            {
                surface.edge_static_bounds( i, xmin, xmax );
            }
            
            // if either vertex is solid, it has to go into the solid broad phase
            if ( surface.edge_is_all_solid(i) )
            {
                solid_edge_xmins.push_back( xmin );
                solid_edge_xmaxs.push_back( xmax );
                solid_edge_indices.push_back( i );
            }
            else
            {
                dynamic_edge_xmins.push_back( xmin );
                dynamic_edge_xmaxs.push_back( xmax );
                dynamic_edge_indices.push_back( i );
            }
        }      
        
        build_acceleration_grid( m_solid_edge_grid,
                                solid_edge_xmins,
                                solid_edge_xmaxs,
                                solid_edge_indices,
                                grid_scale,
                                surface.m_aabb_padding );
        
        build_acceleration_grid( m_dynamic_edge_grid,
                                dynamic_edge_xmins,
                                dynamic_edge_xmaxs,
                                dynamic_edge_indices,
                                grid_scale,
                                surface.m_aabb_padding );
        
    }
    
    //
    // triangles
    //
    {
        size_t num_triangles = surface.m_mesh.num_triangles();
        
        std::vector<Vec3d> solid_tri_xmins, solid_tri_xmaxs;
        std::vector<size_t> solid_tri_indices;
        std::vector<Vec3d> dynamic_tri_xmins, dynamic_tri_xmaxs;
        std::vector<size_t> dynamic_tri_indices;
        
        for(size_t i = 0; i < num_triangles; i++)
        {
            Vec3d xmin, xmax;
            
            if ( continuous )
            {
                surface.triangle_continuous_bounds(i, xmin, xmax);
            }
            else
            {
                surface.triangle_static_bounds(i, xmin, xmax);
            }
            
            if ( surface.triangle_is_all_solid( i ) )
            {
                solid_tri_xmins.push_back( xmin );
                solid_tri_xmaxs.push_back( xmax );
                solid_tri_indices.push_back( i );
            }
            else
            {
                dynamic_tri_xmins.push_back( xmin );
                dynamic_tri_xmaxs.push_back( xmax );
                dynamic_tri_indices.push_back( i );
            }
        }
        
        build_acceleration_grid( m_solid_triangle_grid,
                                solid_tri_xmins,
                                solid_tri_xmaxs,
                                solid_tri_indices,
                                grid_scale,
                                surface.m_aabb_padding );
        
        build_acceleration_grid( m_dynamic_triangle_grid,
                                dynamic_tri_xmins,
                                dynamic_tri_xmaxs,
                                dynamic_tri_indices,
                                grid_scale,
                                surface.m_aabb_padding );
        
    }
    
}

}
