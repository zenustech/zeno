//
//
//  Tyson Brochu 2011
//  Christopher Batty, Fang Da 2014
//

#ifndef TUNICATE_ROOTPARITYCOLLISIONTEST_H
#define TUNICATE_ROOTPARITYCOLLISIONTEST_H

#include <vec.h>
#include <interval.h>
#include <expansion.h>
namespace LosTopos {
namespace rootparity 
{
    
    typedef Vec<3,IntervalType> Vec3Interval;
    typedef Vec<2,expansion> Vec2e;
    typedef Vec<3,expansion> Vec3e;
    typedef Vec<3,bool> Vec3b;
    typedef Vec<4,bool> Vec4b;
    
    /// --------------------------------------------------------
    ///
    /// class RootParityCollisionTest: Encapsulates functions for continuous collision detection using the "root-parity" approach.
    ///
    /// Terms used in documentation:
    /// "Domain": The unit cube for edge-edge collision testing, and half the unit cube for point-triangle.
    /// "Domain boundary vertices": Vertices of the unit cube or half-cube.
    /// "Mapped domain boundary vertices": The set of points F(X), where X = the set of domain boundary vertices, F is either the 
    /// edge-edge collision function or the point-triangle collision function (i.e. F is zero where there is a collision).
    ///
    /// --------------------------------------------------------
    
    class RootParityCollisionTest
    {     
        
    public:
        
        /// Constructor, take a reference to the input vertex locations at t=0 and t=1.  User also specifies whether the vertices 
        /// should be interpreted as representing an edge-edge collision test, or point-triangle.
        ///
        inline RootParityCollisionTest(const Vec3d &x0old, const Vec3d &x1old, const Vec3d &x2old, const Vec3d &x3old,
                                       const Vec3d &x0new, const Vec3d &x1new, const Vec3d &x2new, const Vec3d &x3new,
                                       bool is_edge_edge );
        
        /// Returns true if there is an odd number of ray intersections (corresponding to an odd number of roots of F in the domain).
        ///
        inline bool run_test();
        
        /// Run edge-edge continuous collision detection
        ///
        bool edge_edge_collision();
        
        /// Run point-triangle continuous collision detection
        ///
        bool point_triangle_collision();
        
        
    private:
        
        /// Input vertex locations.
        /// If point-triangle collision test, the point is vertex 0, and the triangle is vertices (1,2,3).
        /// If edge-edge collision test, the edges are (0,1) and (2,3).
        ///
        const Vec3d &m_x0old, &m_x1old, &m_x2old, &m_x3old, &m_x0new, &m_x1new, &m_x2new, &m_x3new;
        
        /// Whether this is an edge-edge collision test
        ///
        const bool m_is_edge_edge;
        
        /// The root-parity test uses a ray from the origin. We will actually use a finite line segment with one end point at the 
        /// origin, and the other end point equal to this variable, "ray".  In the code, we will ensure that ray has magnitude 
        /// greater than the largest magnitude point in the object being tested.
        ///
        Vec3d m_ray;
        
        /// The mapped domain boundary vertices, computed in interval arithmetic.
        ///
        Vec3Interval m_interval_hex_vertices[8]; 
        
        /// The mapped domain boundary vertices, computed in fp-expansion arithmetic. We compute these only as needed, so for each 
        /// vertex, we track if it has been computed yet.
        ///
        Vec3e m_expansion_hex_vertices[8];
        
        /// Whether or not the mapping of each domain boundary vertex has been computed using floating-point expansions.
        ///
        bool m_expansion_hex_vertices_computed[8];
        
        //
        // Functions
        //
        
        /// Determine if the given point is inside the tetrahedron given by tet_indices
        ///
        bool point_tetrahedron_intersection( const Vec4ui& tet_indices, const Vec4b& ts, const Vec4b& us, const Vec4b& vs, const Vec3d& d_x );
        
        /// Determine if the given segment intersects the triangle
        ///
        bool edge_triangle_intersection(const Vec3ui& triangle,
                                        const Vec3b& ts, const Vec3b& us, const Vec3b& vs,
                                        const Vec3d& d_s0, const Vec3d& d_s1,
                                        double* alphas );
        
        /// Compute the sign of the implicit surface function which is zero on the bilinear patch defined by quad.
        ///
        int implicit_surface_function_sign( const Vec4ui& quad, const Vec4b& ts, const Vec4b& us, const Vec4b& vs, const Vec3d& d_x );
        
        /// Test the ray against the bilinear patch defined by a quad to determine whether there is an even or odd number of 
        /// intersections. Returns true if odd.
        ///        
        bool ray_quad_intersection_parity(const Vec4ui& quad, 
                                          const Vec4b& ts, 
                                          const Vec4b& us, 
                                          const Vec4b& vs, 
                                          const Vec3d& ray_origin, 
                                          const Vec3d& ray_direction, 
                                          bool& edge_hit, 
                                          bool& origin_on_surface );
        
        
        /// Determine the parity of the number of intersections between a ray from the origin and the generalized prism made up 
        /// of f(G) where G = the vertices of the domain boundary.
        ///
        bool ray_prism_parity_test();
        
        /// Determine the parity of the number of intersections between a ray from the origin and the generalized hexahedron made
        /// up of f(G) where G = the vertices of the domain boundary (corners of the unit cube).
        ///
        bool ray_hex_parity_test();
        
        /// For each triangle, form the plane it lies on, and determine if all interval_hex_vertices are on one side of the plane.
        ///
        bool plane_culling( const std::vector<Vec3ui>& triangles, const std::vector<Vec3d>& boundary_vertices );
        
        /// Take a set of planes defined by the mapped domain boundary, and determine if all interval_hex_vertices on one side of
        /// any plane.
        /// 
        bool edge_edge_interval_plane_culling();
        
        /// Take a set of planes defined by the mapped domain boundary, and determine if all interval_hex_vertices on one side of
        /// any plane.
        /// 
        bool point_triangle_interval_plane_culling();
        
        /// Take a fixed set of planes and determine if all interval_hex_vertices on one side of any plane.
        /// 
        bool fixed_plane_culling( unsigned int num_hex_vertices );
        
    };
    
    /// --------------------------------------------------------   
    ///
    /// Determine if the given AABB contains the origin.
    ///
    /// --------------------------------------------------------   
    
    inline bool aabb_contains_origin( const Vec3d& xmin, const Vec3d& xmax )
    {
        return (xmin[0] <= 0 && xmin[1] <= 0 && xmin[2] <= 0) && (xmax[0] >= 0 && xmax[1] >= 0 && xmax[2] >= 0 );
    }
    
    /// --------------------------------------------------------
    ///
    /// Determine if the given AABBs intersect each other.
    ///
    /// --------------------------------------------------------   
    
    inline bool aabb_test( const Vec3d& xmin, const Vec3d& xmax, const Vec3d& oxmin, const Vec3d& oxmax )
    {
        return ((xmin[0] <= oxmax[0] && xmin[1] <= oxmax[1] && xmin[2] <= oxmax[2]) &&
                (xmax[0] >= oxmin[0] && xmax[1] >= oxmin[1] && xmax[2] >= oxmin[2]) );
        
    }
    
    /// --------------------------------------------------------
    ///
    /// RootParityCollisionTest constructor.
    ///
    /// --------------------------------------------------------   
    
    inline RootParityCollisionTest::RootParityCollisionTest(const Vec3d &_x0old, const Vec3d &_x1old, const Vec3d &_x2old, const Vec3d &_x3old,
                                                            const Vec3d &_x0new, const Vec3d &_x1new, const Vec3d &_x2new, const Vec3d &_x3new,
                                                            bool _is_edge_edge ) : 
    m_x0old( _x0old ), m_x1old( _x1old ), m_x2old( _x2old ), m_x3old( _x3old ), 
    m_x0new( _x0new ), m_x1new( _x1new ), m_x2new( _x2new ), m_x3new( _x3new ), 
    m_is_edge_edge( _is_edge_edge ),
    m_ray()
    {
        for ( unsigned int i = 0; i < 8; ++i )
        {
            m_expansion_hex_vertices_computed[i] = false;
        }
    }
    
    /// --------------------------------------------------------
    ///
    /// Run the appropriate continuous collision detection test.
    ///
    /// --------------------------------------------------------   
    
    inline bool RootParityCollisionTest::run_test()
    {
        if ( m_is_edge_edge )
        {
            return edge_edge_collision();
        }
        else
        {
            return point_triangle_collision();
        }
    }
    
    
    
    
} // namespace rootparity
} //namespace LosTopos

#endif
