
// ---------------------------------------------------------
//
//  root_parity_ccd_wrapper.cpp
//  Tyson Brochu 2012
//  Christopher Batty, Fang Da 2014
//
//  Root parity collision queries.
//
// ---------------------------------------------------------


#include <ccd_defs.h>


#ifdef USE_ROOT_PARITY_CCD

#include <ccd_wrapper.h>
#include <collisionqueries.h>
//#include <rootparity2d.h>
#include <rootparitycollisiontest.h>
#include <tunicate.h>

namespace LosTopos {

namespace 
{
    /// Tolerance for trusting computed collision normal
    ///
    const double g_degen_normal_epsilon = 1e-6;

    //
    // Local function declarations
    //
    
    void degenerate_get_point_triangle_collision_normal(const Vec3d &x0, const Vec3d &x1, const Vec3d &x2, const Vec3d &x3,
                                                        double &s1, double &s2, double &s3,
                                                        Vec3d& normal );

    void get_point_triangle_collision_normal_no_time(const Vec3d &x0, const Vec3d &x1, const Vec3d &x2, const Vec3d &x3,
                                                     const Vec3d &xnew0, const Vec3d &xnew1, const Vec3d &xnew2, const Vec3d &xnew3,
                                                     double &s1, double &s2, double &s3, Vec3d& normal );

    void degenerate_get_edge_edge_collision_normal(const Vec3d &x0, const Vec3d &x1, const Vec3d &x2, const Vec3d &x3,
                                                   double s0, double s2, Vec3d& normal );

    void get_edge_edge_collision_normal_no_time(const Vec3d &x0, const Vec3d &x1, const Vec3d &x2, const Vec3d &x3,
                                                const Vec3d &xnew0, const Vec3d &xnew1, const Vec3d &xnew2, const Vec3d &xnew3,
                                                double &s0, double &s2, Vec3d& normal );

    
    //
    // Local function definitions
    //

    void degenerate_get_point_triangle_collision_normal(const Vec3d &x0, const Vec3d &x1, const Vec3d &x2, const Vec3d &x3,
                                                        double &s1, double &s2, double &s3,
                                                        Vec3d& normal )
    {
        // try triangle normal at start
        normal=cross(x2-x1, x3-x1);
        double m=mag(normal);
        if(m>sqr(g_degen_normal_epsilon))
        {
            normal/=m;
        }
        else
        {
            // if that didn't work, try vector between points at the start
            
            normal=(s1*x1+s2*x2+s3*x3)-x0;
            m=mag(normal);
            if(m>g_degen_normal_epsilon)
            {
                normal/=m;
            }
            else
            {
                // if that didn't work, boy are we in trouble; just get any non-parallel vector
                Vec3d dx=x2-x1;
                if(dx[0]!=0 || dx[1]!=0)
                {
                    normal=Vec3d(dx[1], -dx[0], 0);
                    normal/=mag(normal);
                }
                else
                {
                    dx=x3-x1;
                    if(dx[0]!=0 || dx[1]!=0)
                    {
                        normal=Vec3d(dx[1], -dx[0], 0);
                        normal/=mag(normal);
                    }
                    else
                    {
                        normal=Vec3d(0, 1, 0); // the last resort
                    }
                }
            }
        }
    }
    
    
    void get_point_triangle_collision_normal_no_time(const Vec3d &x0, const Vec3d &x1, const Vec3d &x2, const Vec3d &x3,
                                                     const Vec3d &xnew0, const Vec3d &xnew1, const Vec3d &xnew2, const Vec3d &xnew3,
                                                     double &s1, double &s2, double &s3, Vec3d& normal )
    {
        // First try at t=0.5
        double distance;
        
        Vec3d x0_half = 0.5 * ( x0 + xnew0 );
        Vec3d x1_half = 0.5 * ( x1 + xnew1 );
        Vec3d x2_half = 0.5 * ( x2 + xnew2 );
        Vec3d x3_half = 0.5 * ( x3 + xnew3 );
        check_point_triangle_proximity(x0_half, x1_half, x2_half, x3_half, distance, s1, s2, s3, normal);
        
        if(distance<1e-2*g_degen_normal_epsilon)
        { 
            // Degenerate normal, try at t=1
            check_point_triangle_proximity(xnew0, xnew1, xnew2, xnew3, distance, s1, s2, s3, normal);
            
            if(distance<1e-2*g_degen_normal_epsilon)
            { 
                // neither one works, go to degenerate normal finder
                degenerate_get_point_triangle_collision_normal( x0, x1, x2, x3, s1, s2, s3, normal );
            }
        }
        
        assert( mag(normal) > 0.0 );
    }
    
    void degenerate_get_edge_edge_collision_normal(const Vec3d &x0, const Vec3d &x1, const Vec3d &x2, const Vec3d &x3,
                                                   double s0, double s2, Vec3d& normal )
    {  
        // if that didn't work, try cross-product of edges at the start
        normal=cross(x1-x0, x3-x2);
        double m=mag(normal);
        if(m>sqr(g_degen_normal_epsilon)){
            normal/=m;
        }else{
            // if that didn't work, try vector between points at the start
            normal=(s2*x2+(1-s2)*x3)-(s0*x0+(1-s0)*x1);
            m=mag(normal);
            if(m>g_degen_normal_epsilon){
                normal/=m;
            }else{
                // if that didn't work, boy are we in trouble; just get any non-parallel vector
                Vec3d dx=x1-x0;
                if(dx[0]!=0 || dx[1]!=0){
                    normal=Vec3d(dx[1], -dx[0], 0);
                    normal/=mag(normal);
                }else{
                    dx=x3-x2;
                    if(dx[0]!=0 || dx[1]!=0){
                        normal=Vec3d(dx[1], -dx[0], 0);
                        normal/=mag(normal);
                    }else{
                        normal=Vec3d(0, 1, 0); // the last resort
                    }
                }
            }
        }
    }
    
    void get_edge_edge_collision_normal_no_time(const Vec3d &x0, const Vec3d &x1, const Vec3d &x2, const Vec3d &x3,
                                                const Vec3d &xnew0, const Vec3d &xnew1, const Vec3d &xnew2, const Vec3d &xnew3,
                                                double &s0, double &s2, Vec3d& normal )
    {
        // First try at t=0.5
        double distance;
        
        Vec3d x0_half = 0.5 * ( x0 + xnew0 );
        Vec3d x1_half = 0.5 * ( x1 + xnew1 );
        Vec3d x2_half = 0.5 * ( x2 + xnew2 );
        Vec3d x3_half = 0.5 * ( x3 + xnew3 );
        check_edge_edge_proximity(x0_half, x1_half, x2_half, x3_half, distance, s0, s2, normal);
        
        if(distance<1e-2*g_degen_normal_epsilon)
        { 
            // Degenerate normal, try at t=1
            check_edge_edge_proximity(xnew0, xnew1, xnew2, xnew3, distance, s0, s2, normal);
            
            if(distance<1e-2*g_degen_normal_epsilon)
            { 
                // neither one works, go to degenerate normal finder
                degenerate_get_edge_edge_collision_normal( x0, x1, x2, x3, s0, s2, normal );
            }
        }
        
        assert( mag(normal) > 0.0 );        
    }
    
}


//// --------------------------------------------------------------------------------------------------
//// 2D Continuous collision detection
//// --------------------------------------------------------------------------------------------------
//
//bool point_segment_collision(const Vec2d& x0, const Vec2d& xnew0, size_t /*index0*/,
//                             const Vec2d& x1, const Vec2d& xnew1, size_t /*index1*/,
//                             const Vec2d& x2, const Vec2d& xnew2, size_t /*index2*/,
//                             double& edge_alpha, Vec2d& normal, double& rel_disp)
//{
//    bool full_interval_result = root_parity_check_point_edge_collision( x0, x1, x2, xnew0, xnew1, xnew2 );
//    
//    if ( full_interval_result )
//    {
//        double distance;
//        check_point_edge_proximity( false, x0, x1, x2, distance, edge_alpha, normal, 1.0 );
//        
//        Vec2d dx0 = xnew0 - x0;
//        Vec2d dx1 = xnew1 - x1;
//        Vec2d dx2 = xnew2 - x2;
//        rel_disp = dot( normal, dx0 - edge_alpha*dx1 - (1-edge_alpha)*dx2 );
//    }
//    
//    return full_interval_result;
//}
//
//bool point_segment_collision(const Vec2d& x0, const Vec2d& xnew0, size_t /*index0*/,
//                             const Vec2d& x1, const Vec2d& xnew1, size_t /*index1*/,
//                             const Vec2d& x2, const Vec2d& xnew2, size_t /*index2*/ )
//{
//    bool full_interval_result = root_parity_check_point_edge_collision( x0, x1, x2, xnew0, xnew1, xnew2 );
//    return full_interval_result;
//}
//
//// --------------------------------------------------------------------------------------------------
//// 2D Static intersection detection
//// --------------------------------------------------------------------------------------------------
//
//bool segment_segment_intersection(const Vec2d& x0, size_t /*index0*/, 
//                                  const Vec2d& x1, size_t /*index1*/,
//                                  const Vec2d& x2, size_t /*index2*/,
//                                  const Vec2d& x3, size_t /*index3*/)
//{
//    double bary[4];   // not returned   
//    return simplex_intersection2d( 2, x0.v, x1.v, x2.v, x3.v, &bary[0], &bary[1], &bary[2], &bary[3] );
//}
//
//
//bool segment_segment_intersection(const Vec2d& x0, size_t /*index0*/, 
//                                  const Vec2d& x1, size_t /*index1*/,
//                                  const Vec2d& x2, size_t /*index2*/,
//                                  const Vec2d& x3, size_t /*index3*/,
//                                  double &s0, double& s2 )
//{
//    double s1, s3;    // not returned
//    return simplex_intersection2d( 2, x0.v, x1.v, x2.v, x3.v, &s0, &s1, &s2, &s3 );
//}


// --------------------------------------------------------------------------------------------------
// 3D Continuous collision detection
// --------------------------------------------------------------------------------------------------


bool point_triangle_collision(const Vec3d& x0, const Vec3d& xnew0, size_t /*index0*/,
                              const Vec3d& x1, const Vec3d& xnew1, size_t /*index1*/,
                              const Vec3d& x2, const Vec3d& xnew2, size_t /*index2*/,
                              const Vec3d& x3, const Vec3d& xnew3, size_t /*index3*/ )
{   
    rootparity::RootParityCollisionTest test( x0, x1, x2, x3, xnew0, xnew1, xnew2, xnew3, false );
    bool rayhex_result = test.run_test();
    return rayhex_result;
}


bool point_triangle_collision(const Vec3d& x0, const Vec3d& xnew0, size_t /*index0*/,
                              const Vec3d& x1, const Vec3d& xnew1, size_t /*index1*/,
                              const Vec3d& x2, const Vec3d& xnew2, size_t /*index2*/,
                              const Vec3d& x3, const Vec3d& xnew3, size_t /*index3*/,
                              double& bary1, double& bary2, double& bary3,
                              Vec3d& normal,
                              double& relative_normal_displacement )
{
    
    rootparity::RootParityCollisionTest test( x0, x1, x2, x3, xnew0, xnew1, xnew2, xnew3, false );
    bool rayhex_result = test.run_test();
    
    if ( rayhex_result )
    {
        get_point_triangle_collision_normal_no_time( x0, x1, x2, x3, xnew0, xnew1, xnew2, xnew3, bary1, bary2, bary3, normal );
        
        Vec3d dx0 = xnew0 - x0;
        Vec3d dx1 = xnew1 - x1;
        Vec3d dx2 = xnew2 - x2;
        Vec3d dx3 = xnew3 - x3;   
        relative_normal_displacement = dot( normal, dx0 - bary1*dx1 - bary2*dx2 - bary3*dx3 );
    }
    
    return rayhex_result;   
    
}


bool segment_segment_collision(const Vec3d& x0, const Vec3d& xnew0, size_t /*index0*/,
                               const Vec3d& x1, const Vec3d& xnew1, size_t /*index1*/,
                               const Vec3d& x2, const Vec3d& xnew2, size_t /*index2*/,
                               const Vec3d& x3, const Vec3d& xnew3, size_t /*index3*/)
{
    
    rootparity::RootParityCollisionTest test( x0, x1, x2, x3, xnew0, xnew1, xnew2, xnew3, true );
    bool rayhex_result = test.run_test();
    return rayhex_result;
}


bool segment_segment_collision(const Vec3d& x0, const Vec3d& xnew0, size_t /*index0*/,
                               const Vec3d& x1, const Vec3d& xnew1, size_t /*index1*/,
                               const Vec3d& x2, const Vec3d& xnew2, size_t /*index2*/,
                               const Vec3d& x3, const Vec3d& xnew3, size_t /*index3*/,
                               double& bary0, double& bary2,
                               Vec3d& normal,
                               double& relative_normal_displacement )
{
    rootparity::RootParityCollisionTest test( x0, x1, x2, x3, xnew0, xnew1, xnew2, xnew3, true );
    bool rayhex_result = test.edge_edge_collision();
    
    if ( rayhex_result )
    {
        get_edge_edge_collision_normal_no_time( x0, x1, x2, x3, xnew0, xnew1, xnew2, xnew3, bary0, bary2, normal );
        
        Vec3d dx0 = xnew0 - x0;
        Vec3d dx1 = xnew1 - x1;
        Vec3d dx2 = xnew2 - x2;
        Vec3d dx3 = xnew3 - x3;   
        
        relative_normal_displacement = dot( normal, bary0*dx0 + (1.0-bary0)*dx1 - bary2*dx2 - (1.0-bary2)*dx3 );
    }
    
    return rayhex_result;   
}


// --------------------------------------------------------------------------------------------------
// 3D Static intersection detection
// --------------------------------------------------------------------------------------------------

// x0-x1 is the segment and and x2-x3-x4 is the triangle.
bool segment_triangle_intersection(const Vec3d& x0, size_t /*index0*/,
                                   const Vec3d& x1, size_t /*index1*/,
                                   const Vec3d& x2, size_t /*index2*/,
                                   const Vec3d& x3, size_t /*index3*/,
                                   const Vec3d& x4, size_t /*index4*/,
                                   bool /*degenerate_counts_as_intersection*/,
                                   bool /*verbose*/ )
{
    double bary[5];
    return simplex_intersection3d( 2, x0.v, x1.v, x2.v, x3.v, x4.v, &bary[0], &bary[1], &bary[2], &bary[3], &bary[4] ) != 0;
}

bool segment_triangle_intersection(const Vec3d& x0, size_t /*index0*/,
                                   const Vec3d& x1, size_t /*index1*/,
                                   const Vec3d& x2, size_t /*index2*/,
                                   const Vec3d& x3, size_t /*index3*/,
                                   const Vec3d& x4, size_t /*index4*/,
                                   double& bary0, double& bary1, double& bary2, double& bary3, double& bary4,
                                   bool /*degenerate_counts_as_intersection*/,
                                   bool /*verbose*/ )
{
    return simplex_intersection3d( 2, x0.v, x1.v, x2.v, x3.v, x4.v, &bary0, &bary1, &bary2, &bary3, &bary4 ) != 0;
}

// x0 is the point and x1-x2-x3-x4 is the tetrahedron. Order is irrelevant.
bool point_tetrahedron_intersection(const Vec3d& x0, size_t /*index0*/,
                                    const Vec3d& x1, size_t /*index1*/,
                                    const Vec3d& x2, size_t /*index2*/,
                                    const Vec3d& x3, size_t /*index3*/,
                                    const Vec3d& x4, size_t /*index4*/)
{
    double bary[5];
    return simplex_intersection3d( 1, x0.v, x1.v, x2.v, x3.v, x4.v, &bary[0], &bary[1], &bary[2], &bary[3], &bary[4] ) != 0;
}

}

#endif



