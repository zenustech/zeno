// ---------------------------------------------------------
//
//  ccd_wrapper.cpp
//  Tyson Brochu 2009
//  Christopher Batty, Fang Da 2014
//
//  Tunicate-based implementation of collision and intersection queries.  (See Robert Bridson's "Tunicate" library.)
//
// ---------------------------------------------------------

//#include <bfstream.h>
#include <ccd_defs.h>
#include <ccd_wrapper.h>
#include <collisionqueries.h>
#include <tunicate.h>
#include <vec.h>

bool tunicate_verbose = false;

namespace LosTopos {


#ifdef USE_TUNICATE_CCD


// --------------------------------------------------------------------------------------------------
// Local functions
// --------------------------------------------------------------------------------------------------

namespace {
    
    bool tunicate_point_segment_collision( const Vec2d& x0, const Vec2d& xnew0, size_t index0,
                                          const Vec2d& x1, const Vec2d& xnew1, size_t index1,
                                          const Vec2d& x2, const Vec2d& xnew2, size_t index2 );
    
    bool tunicate_point_segment_collision(const Vec2d& x0, const Vec2d& xnew0, size_t index0,
                                          const Vec2d& x1, const Vec2d& xnew1, size_t index1,
                                          const Vec2d& x2, const Vec2d& xnew2, size_t index2,
                                          double& edge_alpha, Vec2d& normal, double& time, double& relative_normal_displacement );
    
    bool tunicate_point_triangle_collision(const Vec3d& x0, const Vec3d& xnew0, size_t index0,
                                           const Vec3d& x1, const Vec3d& xnew1, size_t index1,
                                           const Vec3d& x2, const Vec3d& xnew2, size_t index2,
                                           const Vec3d& x3, const Vec3d& xnew3, size_t index3,
                                           double& bary1, double& bary2, double& bary3,
                                           Vec3d& normal,
                                           double& t,
                                           double& relative_normal_displacement,
                                           bool verbose );
    
    bool tunicate_point_triangle_collision(const Vec3d& x0, const Vec3d& xnew0, size_t index0,
                                           const Vec3d& x1, const Vec3d& xnew1, size_t index1,
                                           const Vec3d& x2, const Vec3d& xnew2, size_t index2,
                                           const Vec3d& x3, const Vec3d& xnew3, size_t index3 );
    
    bool tunicate_segment_segment_collision( const Vec3d& x0, const Vec3d& xnew0, size_t index0,
                                            const Vec3d& x1, const Vec3d& xnew1,  size_t index1,
                                            const Vec3d& x2, const Vec3d& xnew2,  size_t index2,
                                            const Vec3d& x3, const Vec3d& xnew3,  size_t index3,
                                            double& bary0, double& bary2,
                                            Vec3d& normal,
                                            double& t,
                                            double& relative_normal_displacement,
                                            bool verbose );
    
    bool tunicate_segment_segment_collision(const Vec3d& x0, const Vec3d& xnew0, size_t index0,
                                            const Vec3d& x1, const Vec3d& xnew1, size_t index1,
                                            const Vec3d& x2, const Vec3d& xnew2, size_t index2,
                                            const Vec3d& x3, const Vec3d& xnew3, size_t index3);
    
    // --------------------------------------------------------------------------------------------------------------
    
    Vec2d get_normal( const Vec2d& v )
    {
        Vec2d p = perp( v );
        double m = mag(p);
        if ( m > 0.0 ) { return p / m; }
        
        // degenerate, pick a random unit vector:
        p[0] = random()%2 ? -0.707106781186548 : 0.707106781186548;
        p[1] = random()%2 ? -0.707106781186548 : 0.707106781186548;
        
        return p;
    }
    
    // --------------------------------------------------------------------------------------------------------------
    
    Vec3d get_normal(const Vec3d& u, const Vec3d& v)
    {
        Vec3d c=cross(u,v);
        double m=mag(c);
        if(m) return c/m;
        
        // degenerate case: either u and v are parallel, or at least one is zero; pick an arbitrary orthogonal vector
        if(mag2(u)>=mag2(v)){
            if(std::fabs(u[0])>=std::fabs(u[1]) && std::fabs(u[0])>=std::fabs(u[2]))
                c=Vec3d(-u[1]-u[2], u[0], u[0]);
            else if(std::fabs(u[1])>=std::fabs(u[2]))
                c=Vec3d(u[1], -u[0]-u[2], u[1]);
            else
                c=Vec3d(u[2], u[2], -u[0]-u[1]);
        }else{
            if(std::fabs(v[0])>=std::fabs(v[1]) && std::fabs(v[0])>=std::fabs(v[2]))
                c=Vec3d(-v[1]-v[2], v[0], v[0]);
            else if(std::fabs(v[1])>=std::fabs(v[2]))
                c=Vec3d(v[1], -v[0]-v[2], v[1]);
            else
                c=Vec3d(v[2], v[2], -v[0]-v[1]);
        }
        m=mag(c);
        if(m) return c/m;
        
        // really degenerate case: u and v are both zero vectors; pick a random unit-length vector
        c[0]=random()%2 ? -0.577350269189626 : 0.577350269189626;
        c[1]=random()%2 ? -0.577350269189626 : 0.577350269189626;
        c[2]=random()%2 ? -0.577350269189626 : 0.577350269189626;
        return c;
        
    }
    
    
    // --------------------------------------------------------------------------------------------------------------
    
    bool tunicate_point_segment_collision( const Vec2d& x0, const Vec2d& xnew0, size_t,
                                          const Vec2d& x1, const Vec2d& xnew1, size_t index1,
                                          const Vec2d& x2, const Vec2d& xnew2, size_t index2 )
    {
        assert( index1 < index2 );
        
        const int segment_triangle_test = 2;
        
        double p0[3] = { x0[0], x0[1], 0.0 };
        double pnew0[3] = { xnew0[0], xnew0[1], 1.0 };
        double p1[3] = { x1[0], x1[1], 0.0 };
        double pnew1[3] = { xnew1[0], xnew1[1], 1.0 };
        double p2[3] = { x2[0], x2[1], 0.0 };
        double pnew2[3] = { xnew2[0], xnew2[1], 1.0 };
        
        double bary[5];
        
        bool intersections[2] = { false, false };
        
        if ( simplex_intersection3d( segment_triangle_test,
                                    p0, pnew0, p1, p2, pnew2,
                                    &bary[0], &bary[1], &bary[2], &bary[3], &bary[4] ) )
        {
            intersections[0] = true;
        }
        
        if ( simplex_intersection3d( segment_triangle_test,
                                    p0, pnew0, p1, pnew1, pnew2,
                                    &bary[0], &bary[1], &bary[2], &bary[3], &bary[4] ) )
        {
            intersections[1] = true;
        }
        
        return ( intersections[0] ^ intersections[1] );  
        
    }
    
    // --------------------------------------------------------------------------------------------------------------
    
    bool tunicate_point_segment_collision(const Vec2d& x0, const Vec2d& xnew0, size_t,
                                          const Vec2d& x1, const Vec2d& xnew1, size_t index1,
                                          const Vec2d& x2, const Vec2d& xnew2, size_t index2,
                                          double& edge_alpha, Vec2d& normal, double& time, double& relative_normal_displacement )
    {
        
        assert( index1 < index2 );
        
        const int segment_triangle_test = 2;
        
        double p0[3] = { x0[0], x0[1], 0.0 };
        double pnew0[3] = { xnew0[0], xnew0[1], 1.0 };
        double p1[3] = { x1[0], x1[1], 0.0 };
        double pnew1[3] = { xnew1[0], xnew1[1], 1.0 };
        double p2[3] = { x2[0], x2[1], 0.0 };
        double pnew2[3] = { xnew2[0], xnew2[1], 1.0 };
        
        double bary[5];   
        bool intersections[2] = { false, false };
        
        if ( simplex_intersection3d( segment_triangle_test,
                                    p0, pnew0, p1, p2, pnew2,
                                    &bary[0], &bary[1], &bary[2], &bary[3], &bary[4] ) )
        {
            intersections[0] = true;      
            edge_alpha=0;     // bary1 = 0, bary2 = 1
            time=bary[1];
            normal = get_normal( x2-x1 );
            relative_normal_displacement = dot( normal, (xnew0-x0)-(xnew2-x2) );
        }
        
        if ( simplex_intersection3d( segment_triangle_test,
                                    p0, pnew0, p1, pnew1, pnew2,
                                    &bary[0], &bary[1], &bary[2], &bary[3], &bary[4] ) )
        {
            intersections[1] = true;
            edge_alpha=1;     // bary1 = 1, bary2 = 0
            time=bary[1];
            normal = get_normal( xnew2-xnew1 );
            relative_normal_displacement = dot( normal, (xnew0-x0)-(xnew1-x1) );
        }
        
        return ( intersections[0] ^ intersections[1] );  
        
    }
    
    
    // --------------------------------------------------------------------------------------------------------------
    
    bool tunicate_point_triangle_collision(const Vec3d& x0, const Vec3d& xnew0, size_t,
                                           const Vec3d& x1, const Vec3d& xnew1, size_t index1,
                                           const Vec3d& x2, const Vec3d& xnew2, size_t index2,
                                           const Vec3d& x3, const Vec3d& xnew3, size_t index3 )
    {
        
        assert( index1 < index2 && index2 < index3 );
        
        const int segment_tetrahedron_test = 2;
        
        double p0[4] = { x0[0], x0[1], x0[2], 0.0 };
        double pnew0[4] = { xnew0[0], xnew0[1], xnew0[2], 1.0 };
        double p1[4] = { x1[0], x1[1], x1[2], 0.0 };
        double pnew1[4] = { xnew1[0], xnew1[1], xnew1[2], 1.0 };
        double p2[4] = { x2[0], x2[1], x2[2], 0.0 };
        double pnew2[4] = { xnew2[0], xnew2[1], xnew2[2], 1.0 };
        double p3[4] = { x3[0], x3[1], x3[2], 0.0 };
        double pnew3[4] = { xnew3[0], xnew3[1], xnew3[2], 1.0 };
        
        
        size_t num_intersections = 0;
        
        double bary[6];
        
        if ( simplex_intersection4d( segment_tetrahedron_test,
                                    p0, pnew0, p1, p2, p3, pnew3,
                                    &bary[0], &bary[1], &bary[2], &bary[3], &bary[4], &bary[5] ) )
        {
            ++num_intersections;
        }
        
        if ( simplex_intersection4d( segment_tetrahedron_test,
                                    p0, pnew0, p1, p2, pnew2, pnew3,
                                    &bary[0], &bary[1], &bary[2], &bary[3], &bary[4], &bary[5] ) )
        {
            ++num_intersections;
        }
        
        if ( simplex_intersection4d( segment_tetrahedron_test,
                                    p0, pnew0, p1, pnew1, pnew2, pnew3,
                                    &bary[0], &bary[1], &bary[2], &bary[3], &bary[4], &bary[5] ) )
        {
            ++num_intersections;
        }
        
        if ( num_intersections == 0 || num_intersections == 2 )
        {
            return false;
        }
        
        return true;
        
    }
    
    
    // --------------------------------------------------------------------------------------------------------------
    
    bool tunicate_point_triangle_collision(const Vec3d& x0, const Vec3d& xnew0, size_t,
                                           const Vec3d& x1, const Vec3d& xnew1, size_t index1,
                                           const Vec3d& x2, const Vec3d& xnew2, size_t index2,
                                           const Vec3d& x3, const Vec3d& xnew3, size_t index3,
                                           double& bary1, double& bary2, double& bary3,
                                           Vec3d& normal,
                                           double& t,
                                           double& relative_normal_displacement,
                                           bool /*verbose*/ )
    {
        
        assert( index1 < index2 && index2 < index3 );
        
        const int segment_tetrahedron_test = 2;
        
        double p0[4] = { x0[0], x0[1], x0[2], 0.0 };
        double p1[4] = { x1[0], x1[1], x1[2], 0.0 };
        double p2[4] = { x2[0], x2[1], x2[2], 0.0 };
        double p3[4] = { x3[0], x3[1], x3[2], 0.0 };
        
        double pnew0[4] = { xnew0[0], xnew0[1], xnew0[2], 1.0 };
        double pnew1[4] = { xnew1[0], xnew1[1], xnew1[2], 1.0 };  
        double pnew2[4] = { xnew2[0], xnew2[1], xnew2[2], 1.0 };
        double pnew3[4] = { xnew3[0], xnew3[1], xnew3[2], 1.0 };
        
        unsigned int num_intersections = 0;
        t = 2.0;
        double bary[6];
        bool any_degen = false;
        
        if ( simplex_intersection4d( segment_tetrahedron_test,
                                    p0, pnew0, p1, p2, p3, pnew3,
                                    &bary[0], &bary[1], &bary[2], &bary[3], &bary[4], &bary[5] ) )
        {
            ++num_intersections;
            
            bary1=0;
            bary2=0;
            bary3=1;      
            t=bary[1];
            normal=get_normal(x2-x1, x3-x1);
            relative_normal_displacement=dot(normal, (xnew0-x0)-(xnew3-x3));
            
            for ( unsigned int i = 0; i < 6; ++i ) { if ( bary[i] == 0.0 ) { any_degen = true; }  }
        }
        
        if ( simplex_intersection4d( segment_tetrahedron_test,
                                    p0, pnew0, p1, p2, pnew2, pnew3,
                                    &bary[0], &bary[1], &bary[2], &bary[3], &bary[4], &bary[5] ) )
        {
            ++num_intersections;
            
            if( bary[1]<t )
            {
                bary1=0;
                bary2=(bary[4]+1e-300)/(bary[4]+bary[5]+2e-300); // guard against zero/zero
                bary3=1-bary2;                  
                t=bary[1];
                normal=get_normal(x2-x1, xnew3-xnew1);
                relative_normal_displacement=dot(normal, (xnew0-x0)-(xnew2-x2));
            }
            
            for ( unsigned int i = 0; i < 6; ++i ) { if ( bary[i] == 0.0 ) { any_degen = true; }  }
        }
        
        if ( simplex_intersection4d( segment_tetrahedron_test,
                                    p0, pnew0, p1, pnew1, pnew2, pnew3,
                                    &bary[0], &bary[1], &bary[2], &bary[3], &bary[4], &bary[5] ) )
        {
            ++num_intersections;
            
            if( bary[1]<t)
            {
                bary1=(bary[3]+1e-300)/(bary[3]+bary[4]+bary[5]+3e-300); // guard against zero/zero
                bary2=(bary[4]+1e-300)/(bary[3]+bary[4]+bary[5]+3e-300); // guard against zero/zero
                bary3=1-bary1-bary2;         
                t=bary[1];
                normal=get_normal(xnew2-xnew1, xnew3-xnew1);
                relative_normal_displacement=dot(normal, (xnew0-x0)-(xnew1-x1));
            }
            
            for ( unsigned int i = 0; i < 6; ++i ) { if ( bary[i] == 0.0 ) { any_degen = true; }  }
            
        }
        
        if ( tunicate_verbose )
        {
            std::cout << "point-triangle, num_intersections: " << num_intersections << std::endl;
        }
        
        if ( num_intersections == 0 || num_intersections == 2 )
        {
            if ( any_degen ) 
            { 
                //g_stats.add_to_int( "tunicate_pt_degens", 1 );
                return true; 
            }
            
            return false;
        }
        
        return true;
        
    }
    
    // --------------------------------------------------------------------------------------------------------------
    
    bool tunicate_segment_segment_collision(const Vec3d& x0, const Vec3d& xnew0, size_t index0,
                                            const Vec3d& x1, const Vec3d& xnew1, size_t index1,
                                            const Vec3d& x2, const Vec3d& xnew2, size_t index2,
                                            const Vec3d& x3, const Vec3d& xnew3, size_t index3)
    {
        
        assert( index0 < index1 );
        assert( index2 < index3 );
        
        const int triangle_triangle_test = 3;
        
        double p0[4] = { x0[0], x0[1], x0[2], 0.0 };
        double pnew0[4] = { xnew0[0], xnew0[1], xnew0[2], 1.0 };
        double p1[4] = { x1[0], x1[1], x1[2], 0.0 };
        double pnew1[4] = { xnew1[0], xnew1[1], xnew1[2], 1.0 };
        double p2[4] = { x2[0], x2[1], x2[2], 0.0 };
        double pnew2[4] = { xnew2[0], xnew2[1], xnew2[2], 1.0 };
        double p3[4] = { x3[0], x3[1], x3[2], 0.0 };
        double pnew3[4] = { xnew3[0], xnew3[1], xnew3[2], 1.0 };
        
        unsigned int num_intersections = 0;
        double bary[6];
        
        if ( simplex_intersection4d( triangle_triangle_test,
                                    p0, p1, pnew1, p2, p3, pnew3,
                                    &bary[0], &bary[1], &bary[2], &bary[3], &bary[4], &bary[5] ) )
        {
            ++num_intersections;
        }
        
        if ( simplex_intersection4d( triangle_triangle_test,
                                    p0, pnew0, pnew1, p2, p3, pnew3,
                                    &bary[0], &bary[1], &bary[2], &bary[3], &bary[4], &bary[5] ) )
        {
            ++num_intersections;
        }
        
        if ( simplex_intersection4d( triangle_triangle_test,
                                    p0, p1, pnew1, p2, pnew2, pnew3,
                                    &bary[0], &bary[1], &bary[2], &bary[3], &bary[4], &bary[5] ) )
        {
            ++num_intersections;
        }
        
        if ( simplex_intersection4d( triangle_triangle_test,
                                    p0, pnew0, pnew1, p2, pnew2, pnew3,
                                    &bary[0], &bary[1], &bary[2], &bary[3], &bary[4], &bary[5] ) )
        {
            ++num_intersections;
        }
        
        if ( num_intersections % 2 == 0 )
        {
            return false;
        }
        
        return true;
        
    }
    
    
    // --------------------------------------------------------------------------------------------------------------
    
    bool tunicate_segment_segment_collision( const Vec3d& x0, const Vec3d& xnew0, size_t index0,
                                            const Vec3d& x1, const Vec3d& xnew1,  size_t index1,
                                            const Vec3d& x2, const Vec3d& xnew2,  size_t index2,
                                            const Vec3d& x3, const Vec3d& xnew3,  size_t index3,
                                            double& bary0, double& bary2,
                                            Vec3d& normal,
                                            double& t,
                                            double& relative_normal_displacement,
                                            bool /*verbose*/ )
    {
        
        assert( index0 < index1 );
        assert( index2 < index3 );
        
        const int triangle_triangle_test = 3;
        
        double p0[4] = { x0[0], x0[1], x0[2], 0.0 };
        double p1[4] = { x1[0], x1[1], x1[2], 0.0 };
        double p2[4] = { x2[0], x2[1], x2[2], 0.0 };
        double p3[4] = { x3[0], x3[1], x3[2], 0.0 };
        
        double pnew0[4] = { xnew0[0], xnew0[1], xnew0[2], 1.0 };
        double pnew1[4] = { xnew1[0], xnew1[1], xnew1[2], 1.0 };
        double pnew2[4] = { xnew2[0], xnew2[1], xnew2[2], 1.0 };
        double pnew3[4] = { xnew3[0], xnew3[1], xnew3[2], 1.0 };
        
        double bary[6];
        t = 2.0;
        
        unsigned int num_intersections = 0;
        
        std::vector<unsigned int> degen_counts(10, 0);
        
        if ( simplex_intersection4d( triangle_triangle_test,
                                    p0, p1, pnew1, p2, p3, pnew3,
                                    &bary[0], &bary[1], &bary[2], &bary[3], &bary[4], &bary[5] ) )
        {
            ++num_intersections;
            
            if ( bary[0] == 0 ) { degen_counts[2]++; degen_counts[4]++; }
            if ( bary[1] == 0 ) { degen_counts[1]++; }
            if ( bary[2] == 0 ) { degen_counts[0]++; degen_counts[3]; }
            if ( bary[3] == 0 ) { degen_counts[7]++; degen_counts[9]++; }
            if ( bary[4] == 0 ) { degen_counts[6]++; }
            if ( bary[5] == 0 ) { degen_counts[5]++; degen_counts[8]++; }      
            
            if ( tunicate_verbose )
            {
                std::cout << "intersection A, barys: ";
                std::cout << bary[0] << " " << bary[1] << " " << bary[2] << " " << bary[3] << " " << bary[4] << " " << bary[5] << std::endl;
            }
            
            bary0=0;
            bary2=0;
            t=bary[2];
            normal=get_normal(x1-x0, x3-x2);
            relative_normal_displacement=dot(normal, (xnew1-x1)-(xnew3-x3));            
        }
        
        if ( simplex_intersection4d( triangle_triangle_test,
                                    p0, pnew0, pnew1, p2, p3, pnew3,
                                    &bary[0], &bary[1], &bary[2], &bary[3], &bary[4], &bary[5] ) )
        {
            ++num_intersections;
            
            if ( bary[0] == 0 ) { degen_counts[2]++; degen_counts[4]++; }
            if ( bary[1] == 0 ) { degen_counts[1]++; }
            if ( bary[2] == 0 ) { degen_counts[0]++; degen_counts[3]; }
            if ( bary[3] == 0 ) { degen_counts[7]++; degen_counts[9]++; }
            if ( bary[4] == 0 ) { degen_counts[6]++; }
            if ( bary[5] == 0 ) { degen_counts[5]++; degen_counts[8]++; }
            
            if ( tunicate_verbose )
            {
                std::cout << "intersection B, barys: ";
                std::cout << bary[0] << " " << bary[1] << " " << bary[2] << " " << bary[3] << " " << bary[4] << " " << bary[5] << std::endl;
            }
            
            if( bary[5]<t )
            {
                bary0=(bary[1]+1e-300)/(bary[1]+bary[2]+2e-300); // guard against zero/zero
                bary2=0;
                t=bary[5];
                normal=get_normal(xnew1-xnew0, x3-x2);
                relative_normal_displacement=dot(normal, (xnew0-x0)-(xnew3-x3));
            }
        }
        
        
        if ( simplex_intersection4d( triangle_triangle_test,
                                    p0, p1, pnew1, p2, pnew2, pnew3,
                                    &bary[0], &bary[1], &bary[2], &bary[3], &bary[4], &bary[5] ) )
        {
            ++num_intersections;
            
            if ( bary[0] == 0 ) { degen_counts[2]++; degen_counts[4]++; }
            if ( bary[1] == 0 ) { degen_counts[1]++; }
            if ( bary[2] == 0 ) { degen_counts[0]++; degen_counts[3]; }
            if ( bary[3] == 0 ) { degen_counts[7]++; degen_counts[9]++; }
            if ( bary[4] == 0 ) { degen_counts[6]++; }
            if ( bary[5] == 0 ) { degen_counts[5]++; degen_counts[8]++; }
            
            if ( tunicate_verbose )
            {
                std::cout << "intersection C, barys: ";
                std::cout << bary[0] << " " << bary[1] << " " << bary[2] << " " << bary[3] << " " << bary[4] << " " << bary[5] << std::endl;
            }
            
            if( bary[2]<t )
            {
                bary0=0;
                bary2=(bary[4]+1e-300)/(bary[4]+bary[5]+2e-300); // guard against zero/zero
                t=bary[2];
                normal=get_normal(x1-x0, xnew3-xnew2);
                relative_normal_displacement=dot(normal, (xnew1-x1)-(xnew2-x2));
            }
        }
        
        if ( simplex_intersection4d( triangle_triangle_test,
                                    p0, pnew0, pnew1, p2, pnew2, pnew3,
                                    &bary[0], &bary[1], &bary[2], &bary[3], &bary[4], &bary[5] ) )
        {
            
            ++num_intersections;
            
            if ( bary[0] == 0 ) { degen_counts[2]++; degen_counts[4]++; }
            if ( bary[1] == 0 ) { degen_counts[1]++; }
            if ( bary[2] == 0 ) { degen_counts[0]++; degen_counts[3]; }
            if ( bary[3] == 0 ) { degen_counts[7]++; degen_counts[9]++; }
            if ( bary[4] == 0 ) { degen_counts[6]++; }
            if ( bary[5] == 0 ) { degen_counts[5]++; degen_counts[8]++; }
            
            if ( tunicate_verbose )
            {
                std::cout << "intersection D, barys: ";
                std::cout << bary[0] << " " << bary[1] << " " << bary[2] << " " << bary[3] << " " << bary[4] << " " << bary[5] << std::endl;
            }
            
            if( 1-bary[0]<t)
            {
                bary0=(bary[1]+1e-300)/(bary[1]+bary[2]+2e-300); // guard against zero/zero
                bary2=(bary[4]+1e-300)/(bary[4]+bary[5]+2e-300); // guard against zero/zero
                t=1-bary[0];
                normal=get_normal(xnew1-xnew0, xnew3-xnew2);
                relative_normal_displacement=dot(normal, (xnew0-x0)-(xnew2-x2));
            }
        }
        
        if ( tunicate_verbose )
        {
            std::cout << "edge-edge, num_intersections: " << num_intersections;
            std::cout << "degen_counts: ";
            for ( size_t i = 0; i < degen_counts.size(); ++i ) { std::cout << degen_counts[i] << " "; }
            std::cout << std::endl;
            
        }
        
        if ( num_intersections % 2 == 0 )
        {
            for ( size_t i = 0; i < degen_counts.size(); ++i ) 
            { 
                if( degen_counts[i] > 0 ) 
                { 
                    //g_stats.add_to_int( "tunicate_ee_degens", 1 );
                    return true; 
                } 
            }
            
            return false;
        }
        
        return true;
        
    }
    
}     // namespace


// --------------------------------------------------------------------------------------------------
// 2D Continuous collision detection
// --------------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------------------------

bool point_segment_collision( const Vec2d& x0, const Vec2d& xnew0, size_t index0,
                             const Vec2d& x1, const Vec2d& xnew1, size_t index1,
                             const Vec2d& x2, const Vec2d& xnew2, size_t index2 )
{
    return tunicate_point_segment_collision( x0, xnew0, index0, 
                                            x1, xnew1, index1, 
                                            x2, xnew2, index2 );
}

bool point_segment_collision(const Vec2d& x0, const Vec2d& xnew0, size_t index0,
                             const Vec2d& x1, const Vec2d& xnew1, size_t index1,
                             const Vec2d& x2, const Vec2d& xnew2, size_t index2,
                             double& edge_alpha, Vec2d& normal, double& time, double& relative_normal_displacement )
{
    bool tunicate_result = tunicate_point_segment_collision( x0, xnew0, index0, 
                                                            x1, xnew1, index1, 
                                                            x2, xnew2, index2,
                                                            edge_alpha, normal, time, relative_normal_displacement );
    
    return tunicate_result;
    
}


// --------------------------------------------------------------------------------------------------
// 2D Static intersection detection / distance queries
// --------------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------------------------

bool segment_segment_intersection(const Vec2d& x0, size_t /*index0*/, 
                                  const Vec2d& x1, size_t /*index1*/,
                                  const Vec2d& x2, size_t /*index2*/,
                                  const Vec2d& x3, size_t /*index3*/)
{
    double bary[4];   // not returned   
    return simplex_intersection2d( 2, x0.v, x1.v, x2.v, x3.v, &bary[0], &bary[1], &bary[2], &bary[3] );
}

// --------------------------------------------------------------------------------------------------------------

bool segment_segment_intersection(const Vec2d& x0, size_t /*index0*/, 
                                  const Vec2d& x1, size_t /*index1*/,
                                  const Vec2d& x2, size_t /*index2*/,
                                  const Vec2d& x3, size_t /*index3*/,
                                  double &s0, double& s2 )
{
    double s1, s3;    // not returned
    return simplex_intersection2d( 2, x0.v, x1.v, x2.v, x3.v, &s0, &s1, &s2, &s3 );
}


// --------------------------------------------------------------------------------------------------
// 3D Continuous collision detection
// --------------------------------------------------------------------------------------------------


bool point_triangle_collision(const Vec3d& x0, const Vec3d& xnew0, size_t index0,
                              const Vec3d& x1, const Vec3d& xnew1, size_t index1,
                              const Vec3d& x2, const Vec3d& xnew2, size_t index2,
                              const Vec3d& x3, const Vec3d& xnew3, size_t index3 )
{
    bool tunicate_result = tunicate_point_triangle_collision( x0, xnew0, index0,
                                                             x1, xnew1, index1,
                                                             x2, xnew2, index2,
                                                             x3, xnew3, index3 );                                            
    
    return tunicate_result;
    
} 

// --------------------------------------------------------------------------------------------------------------

bool point_triangle_collision( const Vec3d& x0, const Vec3d& xnew0, size_t index0,
                              const Vec3d& x1, const Vec3d& xnew1, size_t index1,
                              const Vec3d& x2, const Vec3d& xnew2, size_t index2,
                              const Vec3d& x3, const Vec3d& xnew3, size_t index3,
                              double& bary1, double& bary2, double& bary3,
                              Vec3d& normal,
                              double& relative_normal_displacement )
{
    
    double time;
    bool verbose = false;
    bool tunicate_result = tunicate_point_triangle_collision( x0, xnew0, index0,
                                                             x1, xnew1, index1,
                                                             x2, xnew2, index2,
                                                             x3, xnew3, index3,
                                                             bary1, bary2, bary3,
                                                             normal, time, relative_normal_displacement, verbose );
    
    return tunicate_result;
    
}


// --------------------------------------------------------------------------------------------------------------


bool segment_segment_collision(const Vec3d& x0, const Vec3d& xnew0, size_t index0,
                               const Vec3d& x1, const Vec3d& xnew1, size_t index1,
                               const Vec3d& x2, const Vec3d& xnew2, size_t index2,
                               const Vec3d& x3, const Vec3d& xnew3, size_t index3)
{
    bool tunicate_result = tunicate_segment_segment_collision( x0, xnew0, index0,
                                                              x1, xnew1, index1,
                                                              x2, xnew2, index2,
                                                              x3, xnew3, index3 );
    
    return tunicate_result;
}

// --------------------------------------------------------------------------------------------------------------


bool segment_segment_collision( const Vec3d& x0, const Vec3d& xnew0, size_t index0,
                               const Vec3d& x1, const Vec3d& xnew1, size_t index1,
                               const Vec3d& x2, const Vec3d& xnew2, size_t index2,
                               const Vec3d& x3, const Vec3d& xnew3, size_t index3,
                               double& bary0, double& bary2,
                               Vec3d& normal,
                               double& relative_normal_displacement )
{
    double time;
    bool verbose = false;
    bool tunicate_result = tunicate_segment_segment_collision( x0, xnew0, index0,
                                                              x1, xnew1, index1,
                                                              x2, xnew2, index2,
                                                              x3, xnew3, index3,
                                                              bary0, bary2, normal, time, relative_normal_displacement, verbose );
    
    return tunicate_result;
    
}


// --------------------------------------------------------------------------------------------------
// 3D Static intersection detection
// --------------------------------------------------------------------------------------------------

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
    return simplex_intersection3d( 2, x0.v, x1.v, x2.v, x3.v, x4.v, &bary[0], &bary[1], &bary[2], &bary[3], &bary[4] );
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
    return simplex_intersection3d( 2, x0.v, x1.v, x2.v, x3.v, x4.v, &bary0, &bary1, &bary2, &bary3, &bary4 );
}


// --------------------------------------------------------------------------------------------------


// x0 is the point and x1-x2-x3-x4 is the tetrahedron. Order is irrelevant.
bool point_tetrahedron_intersection(const Vec3d& x0, size_t /*index0*/,
                                    const Vec3d& x1, size_t /*index1*/,
                                    const Vec3d& x2, size_t /*index2*/,
                                    const Vec3d& x3, size_t /*index3*/,
                                    const Vec3d& x4, size_t /*index4*/)
{
    double bary[5];
    return simplex_intersection3d( 1, x0.v, x1.v, x2.v, x3.v, x4.v, &bary[0], &bary[1], &bary[2], &bary[3], &bary[4] );   
}

#endif

}


