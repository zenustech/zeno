#include <cfloat>
#include <predicates.h>
#include <iostream>

namespace LosTopos {

static void print_hex( double d )
{
    std::ios_base::fmtflags originalFlags = std::cout.flags();
    
    assert( sizeof(double) == 2 * sizeof(long int) );
    
    union double_ints
    {
        double d;
        long int c[2];
    };
    
    double_ints dc;
    dc.d = d;
    
    std::cout.setf(std::ios::hex, std::ios::basefield );
    
    std::cout << dc.c[0] << dc.c[1];
    
    std::cout.flags(originalFlags);
}

static void print_hex( const Vec3d& v )
{
    print_hex( v[0] );
    std::cout << " ";
    print_hex( v[1] );
    std::cout << " ";
    print_hex( v[2] );
    std::cout << std::endl;   
}



//=======================================================================================================
//=======================================================================================================
// DOUBLE PRECISION VERSIONS
//=======================================================================================================
//=======================================================================================================

// Determine the orientation of 4 points in 2d space and 1d time
// with end0 indicating if point 0 is at t=1 instead of t=0, etc.
// Returns zero if the determinant is too small to reliably know the sign.
static double orient(const Vec2d& x0, bool end0,
                     const Vec2d& x1, bool end1,
                     const Vec2d& x2, bool end2,
                     const Vec2d& x3, bool end3)
{
    assert(end0 || end1 || end2 || end3);
    assert(!end0 || !end1 || !end2 || !end3);
    // first do all six 2x2 determinants and permanents (error ratio slightly more than 2*DBL_EPSILON)
    double d01=x0[0]*x1[1]-x0[1]*x1[0], p01=std::fabs(x0[0]*x1[1])+std::fabs(x0[1]*x1[0]); 
    double d02=x0[0]*x2[1]-x0[1]*x2[0], p02=std::fabs(x0[0]*x2[1])+std::fabs(x0[1]*x2[0]); 
    double d03=x0[0]*x3[1]-x0[1]*x3[0], p03=std::fabs(x0[0]*x3[1])+std::fabs(x0[1]*x3[0]); 
    double d12=x1[0]*x2[1]-x1[1]*x2[0], p12=std::fabs(x1[0]*x2[1])+std::fabs(x1[1]*x2[0]); 
    double d13=x1[0]*x3[1]-x1[1]*x3[0], p13=std::fabs(x1[0]*x3[1])+std::fabs(x1[1]*x3[0]); 
    double d23=x2[0]*x3[1]-x2[1]*x3[0], p23=std::fabs(x2[0]*x3[1])+std::fabs(x2[1]*x3[0]); 
    // then add up as needed
    double det=0, perm=0;
    if(end0){
        det=d12+d23-d13;
        perm=p12+p23-p13;
    }
    if(end1){
        det-=d02+d23-d03;
        perm+=p02+p23+p03;
    }
    if(end2){
        det+=d01+d13-d03;
        perm+=p01+p13+p03;
    }
    if(end3){
        det-=d01+d12-d02;
        perm+=p01+p12+p02;
    }
    double err=perm*4.1*DBL_EPSILON; // should be safe, I think
    if(det>err || det<-err) return det;
    else return 0;
}

static bool point_point_collision(const Vec2d& x0, const Vec2d& xnew0,
                                  const Vec2d& x1, const Vec2d& xnew1)
{
    if(orient(x0, false, xnew0, true, x1, false, xnew1, true)) return false; // lines are skew
    // check if points retain ordering along any axis
    if((x0[0]<x1[0] && xnew0[0]<xnew1[0]) || (x0[0]>x1[0] && xnew0[0]>xnew1[0])) return false;
    if((x0[1]<x1[1] && xnew0[1]<xnew1[1]) || (x0[1]>x1[1] && xnew0[1]>xnew1[1])) return false;
    // otherwise, must have intersection or they're exactly the same point
    return true;
}

static bool segment_triangle_intersection(const Vec2d& x0, const Vec2d& xnew0,
                                          const Vec2d& x1, bool end1,
                                          const Vec2d& x2, bool end2,
                                          const Vec2d& x3, bool end3,
                                          bool verbose = false )
{
    // first check if the segment crosses the plane of the triangle
    double s= orient(x0, false, x1, end1, x2, end2, x3, end3),
    t=-orient(xnew0, true, x1, end1, x2, end2, x3, end3);
    if((s<0 && t>0) || (s>0 && t<0)) return false;
    // check if the line of the segment pierces the triangle
    double a= orient(x0, false, xnew0, true, x2, end2, x3, end3),
    b=-orient(x0, false, xnew0, true, x1, end1, x3, end3),
    c= orient(x0, false, xnew0, true, x1, end1, x2, end2);
    
    if ( verbose ) 
    { 
        std::cout << "a: " << a << "\nb: " << b << "\nc: " << c << std::endl; 
        std::cout << "ends: " << end1 << ", " << end2 << ", " << end3 << std::endl;
    }
    
    if((s==0 || t==0) || (a==0 || b==0 || c==0)){ // degenerate cases
        if(end1){
            if(end2) return point_point_collision(x0, xnew0, x3, x1) || point_point_collision(x0, xnew0, x3, x2);
            else     return point_point_collision(x0, xnew0, x3, x1) || point_point_collision(x0, xnew0, x2, x1);
        }else if(end2){
            if(end3) return point_point_collision(x0, xnew0, x1, x2) || point_point_collision(x0, xnew0, x1, x3);
            else     return point_point_collision(x0, xnew0, x1, x2) || point_point_collision(x0, xnew0, x3, x2);
        }else
        {
            if ( verbose ) 
            {
                std::cout << "point-point: " << x0 << ", " << xnew0 << ", " << x1 << ", " << x2 << std::endl;
                std::cout << "point-point: " << x0 << ", " << xnew0 << ", " << x1 << ", " << x3 << std::endl;
            }
            return point_point_collision(x0, xnew0, x1, x2) || point_point_collision(x0, xnew0, x1, x3);
        }
    }
    // non-degenerate case
    return ((a<=0 && b<=0 && c<=0) || (a>=0 && b>=0 && c>=0));
}

static bool segment_triangle_intersection(const Vec2d& x0, const Vec2d& xnew0,
                                          const Vec2d& x1, bool end1,
                                          const Vec2d& x2, bool end2,
                                          const Vec2d& x3, bool end3,
                                          double& bary )
{
    // first check if the segment crosses the plane of the triangle
    double s = orient(x0, false, x1, end1, x2, end2, x3, end3);
    double t = -orient(xnew0, true, x1, end1, x2, end2, x3, end3);
    
    //std::cout << "segment_triangle_intersection -- s: " << s << ", t: " << t << std::endl;
    
    if((s<0 && t>0) || (s>0 && t<0)) return false;
    
    // check if the line of the segment pierces the triangle
    double a= orient(x0, false, xnew0, true, x2, end2, x3, end3),
    b=-orient(x0, false, xnew0, true, x1, end1, x3, end3),
    c= orient(x0, false, xnew0, true, x1, end1, x2, end2);
    
    //std::cout << "segment_triangle_intersection -- a: " << a << ", b: " << b << ", c: " << c << std::endl;
    
    if((s==0 || t==0) || (a==0 || b==0 || c==0)){ // degenerate cases
        bary = 0.0;    // in degenerate case
        if(end1){
            if(end2) return point_point_collision(x0, xnew0, x3, x1) || point_point_collision(x0, xnew0, x3, x2);
            else     return point_point_collision(x0, xnew0, x3, x1) || point_point_collision(x0, xnew0, x2, x1);
        }else if(end2){
            if(end3) return point_point_collision(x0, xnew0, x1, x2) || point_point_collision(x0, xnew0, x1, x3);
            else     return point_point_collision(x0, xnew0, x1, x2) || point_point_collision(x0, xnew0, x3, x2);
        }else
            return point_point_collision(x0, xnew0, x1, x2) || point_point_collision(x0, xnew0, x1, x3);
    }
    
    // non-degenerate case
    bary = s / (s+t);
    return ((a<=0 && b<=0 && c<=0) || (a>=0 && b>=0 && c>=0));
}


bool fe_point_segment_collision(const Vec2d& x0, const Vec2d& xnew0,
                                const Vec2d& x1, const Vec2d& xnew1,
                                const Vec2d& x2, const Vec2d& xnew2)
{
    bool intersections[2] = { false, false };
    if(segment_triangle_intersection(x0, xnew0, x1, false, x2, false, xnew1, true))
    {
        intersections[0] = true;
    }
    
    if(segment_triangle_intersection(x0, xnew0, x2, false, xnew2, true, xnew1, true))
    {
        intersections[1] = true;
    }
    
    return (intersections[0] ^ intersections[1]);
}


bool fe_point_segment_collision(const Vec2d& x0, const Vec2d& xnew0,
                                const Vec2d& x1, const Vec2d& xnew1,
                                const Vec2d& x2, const Vec2d& xnew2,
                                double& edge_bary, Vec2d& normal, double& t,
                                double& relative_normal_displacement )
{
    bool intersections[2] = { false, false };
    double bary = 0.0;
    
    if( segment_triangle_intersection(x0, xnew0, x1, false, x2, false, xnew1, true, bary) )
    {
        //std::cout << "collision (a) t: " << bary << std::endl;
        
        // collision happens in triangle with one "new" point
        intersections[0] = true;
        edge_bary = 1.0;
        t = bary;
        normal = perp( x2 - x1 );
        normal /= mag(normal);
        relative_normal_displacement=dot( normal, (xnew0-x0) - (xnew1-x1) );
    }
    
    if( segment_triangle_intersection(x0, xnew0, x2, false, xnew2, true, xnew1, true, bary) )
    {
        //std::cout << "collision (b) t: " << bary << std::endl;
        // collision happens in triangle with two "new" points
        intersections[1] = true;
        edge_bary = 0.5;
        t = bary;
        normal = perp( xnew2 - xnew1 );
        normal /= mag(normal);
        relative_normal_displacement=dot( normal, (xnew0-x0) - (xnew2-x2) );         
    }
    
    return (intersections[0] ^ intersections[1]);
    
}


//=======================================================================================================

// determine the orientation of 3 points in 2d space
static double orient(const Vec2d& x0,
                     const Vec2d& x1,
                     const Vec2d& x2)
{
    double det=x0[0]*x1[1] + x1[0]*x2[1] + x2[0]*x0[1]
    - x0[1]*x1[0] - x1[1]*x2[0] - x2[1]*x0[0];
    double perm=std::fabs(x0[0]*x1[1]) + std::fabs(x1[0]*x2[1]) + std::fabs(x2[0]*x0[1])
    + std::fabs(x0[1]*x1[0]) + std::fabs(x1[1]*x2[0]) + std::fabs(x2[1]*x0[0]);
    double err=perm*6.1*DBL_EPSILON; // should be safe, I think
    if(det>err || det<-err) return det;
    else return 0;
}

static bool point_segment_intersection(const Vec2d& x0,
                                       const Vec2d& x1, const Vec2d& x2)
{
    if(orient(x0,x1,x2)) return false; // point is not on the line of the segment
    if(x1[0]==x2[0]){
        if(x1[1]==x2[1]) return x0[0]==x1[0] && x0[1]==x1[1];
        else return (x0[1]>=x1[1] && x0[1]<=x2[1]) || (x0[1]>=x2[1] && x0[1]<=x1[1]);
    }else
        return (x0[0]>=x1[0] && x0[0]<=x2[0]) || (x0[0]>=x2[0] && x0[0]<=x1[0]);
}

bool fe_segment_segment_intersection(const Vec2d& x0, const Vec2d& x1,
                                     const Vec2d& x2, const Vec2d& x3)
{
    double d0=orient(x0,x2,x3), d1=orient(x1,x2,x3);
    if((d0<0 && d1<0) || (d0>0 && d1>0)) return false; // 0-1 lies entirely on one side of 2-3
    double d2=orient(x2,x0,x1), d3=orient(x3,x0,x1);
    if((d2<0 && d3<0) || (d2>0 && d3>0)) return false; // 2-3 lies entirely on one side of 1-2
    if((d0==0 || d1==0) || (d2==0 || d3==0)){ // check for degeneracy
        return point_segment_intersection(x0, x2, x3)
        || point_segment_intersection(x1, x2, x3)
        || point_segment_intersection(x2, x0, x1)
        || point_segment_intersection(x3, x0, x1);
    }
    return true;
}

bool fe_segment_segment_intersection(const Vec2d& x0, const Vec2d& x1,
                                     const Vec2d& x2, const Vec2d& x3,
                                     double& alpha, double& beta )
{
    double d0=orient(x0,x2,x3), d1=orient(x1,x2,x3);
    if((d0<0 && d1<0) || (d0>0 && d1>0)) return false; // 0-1 lies entirely on one side of 2-3
    double d2=orient(x2,x0,x1), d3=orient(x3,x0,x1);
    if((d2<0 && d3<0) || (d2>0 && d3>0)) return false; // 2-3 lies entirely on one side of 1-2
    if((d0==0 || d1==0) || (d2==0 || d3==0)) { // check for degeneracy
        return point_segment_intersection(x0, x2, x3)
        || point_segment_intersection(x1, x2, x3)
        || point_segment_intersection(x2, x0, x1)
        || point_segment_intersection(x3, x0, x1);
    }
    alpha = 0.0;
    beta = 1.0;
    return true;
}

//=======================================================================================================

static double determinant(const Vec3d& x0,
                          const Vec3d& x1,
                          const Vec3d& x2)
{
    return x0[0]*(x1[1]*x2[2]-x2[1]*x1[2])
    -x1[0]*(x0[1]*x2[2]-x2[1]*x0[2])
    +x2[0]*(x0[1]*x1[2]-x1[1]*x0[2]);
}

// make sure inputs are all non-negative!
static double permanent(const Vec3d& a0,
                        const Vec3d& a1,
                        const Vec3d& a2)
{
    return a0[0]*(a1[1]*a2[2]+a2[1]*a1[2])
    +a1[0]*(a0[1]*a2[2]+a2[1]*a0[2])
    +a2[0]*(a0[1]*a1[2]+a1[1]*a0[2]);
}

static double check_error(double value, double bound, double multiplier)
{
    double err=bound*multiplier*DBL_EPSILON;
    //std::cout<<" ######### checking value "<<value<<" against error bound "<<err<<std::endl;
    if(value>err || value<-err) return value;
    return 0;
}

// determine the orientations of 6 points in 3d space and 1d time
// with end0 indicating if point 0 is at t=1 instead of t=0, etc.
static void orient(const Vec3d& x0, bool end0,
                   const Vec3d& x1, bool end1,
                   const Vec3d& x2, bool end2,
                   const Vec3d& x3, bool end3,
                   const Vec3d& x4, bool end4,
                   const Vec3d& x5, bool end5,
                   double orientations[6])
{
    assert(end0 || end1 || end2 || end3 || end4 || end5);
    assert(!end0 || !end1 || !end2 || !end3 || !end4 || !end5);
    // @@@
    // Determinants may be a bad way to do this (since there is so much arithmetic)!
    // Actually solving the linear system with QR might make more sense, though it would involve
    // nastier floating-point error analysis.
    // @@@
    // first do the 3x3's
    double d012, d013, d014, d015, d023, d024, d025, d034, d035, d045,
    d123, d124, d125, d134, d135, d145, d234, d235, d245, d345;
    d012=determinant(x0, x1, x2); d013=determinant(x0, x1, x3); d014=determinant(x0, x1, x4);
    d015=determinant(x0, x1, x5); d023=determinant(x0, x2, x3); d024=determinant(x0, x2, x4);
    d025=determinant(x0, x2, x5); d034=determinant(x0, x3, x4); d035=determinant(x0, x3, x5);
    d045=determinant(x0, x4, x5); d123=determinant(x1, x2, x3); d124=determinant(x1, x2, x4);
    d125=determinant(x1, x2, x5); d134=determinant(x1, x3, x4); d135=determinant(x1, x3, x5);
    d145=determinant(x1, x4, x5); d234=determinant(x2, x3, x4); d235=determinant(x2, x3, x5);
    d245=determinant(x2, x4, x5); d345=determinant(x3, x4, x5);
    double p012, p013, p014, p015, p023, p024, p025, p034, p035, p045,
    p123, p124, p125, p134, p135, p145, p234, p235, p245, p345;
    Vec3d a0(std::fabs(x0[0]), std::fabs(x0[1]), std::fabs(x0[2])),
    a1(std::fabs(x1[0]), std::fabs(x1[1]), std::fabs(x1[2])),
    a2(std::fabs(x2[0]), std::fabs(x2[1]), std::fabs(x2[2])),
    a3(std::fabs(x3[0]), std::fabs(x3[1]), std::fabs(x3[2])),
    a4(std::fabs(x4[0]), std::fabs(x4[1]), std::fabs(x4[2])),
    a5(std::fabs(x5[0]), std::fabs(x5[1]), std::fabs(x5[2]));
    p012=permanent(a0, a1, a2); p013=permanent(a0, a1, a3); p014=permanent(a0, a1, a4);
    p015=permanent(a0, a1, a5); p023=permanent(a0, a2, a3); p024=permanent(a0, a2, a4);
    p025=permanent(a0, a2, a5); p034=permanent(a0, a3, a4); p035=permanent(a0, a3, a5);
    p045=permanent(a0, a4, a5); p123=permanent(a1, a2, a3); p124=permanent(a1, a2, a4);
    p125=permanent(a1, a2, a5); p134=permanent(a1, a3, a4); p135=permanent(a1, a3, a5);
    p145=permanent(a1, a4, a5); p234=permanent(a2, a3, a4); p235=permanent(a2, a3, a5);
    p245=permanent(a2, a4, a5); p345=permanent(a3, a4, a5);
    // error in dABC is bounded by pABC*5.1*DBL_EPSILON I believe
    
    // now do the 4x4's
    double d0123=(d123-d023)+(d013-d012), p0123=(p123+p023)+(p013+p012),
    d0124=(d124-d024)+(d014-d012), p0124=(p124+p024)+(p014+p012),
    d0125=(d125-d025)+(d015-d012), p0125=(p125+p025)+(p015+p012),
    d0134=(d134-d034)+(d014-d013), p0134=(p134+p034)+(p014+p013),
    d0135=(d135-d035)+(d015-d013), p0135=(p135+p035)+(p015+p013),
    d0145=(d145-d045)+(d015-d014), p0145=(p145+p045)+(p015+p014),
    d0234=(d234-d034)+(d024-d023), p0234=(p234+p034)+(p024+p023),
    d0235=(d235-d035)+(d025-d023), p0235=(p235+p035)+(p025+p023),
    d0245=(d245-d045)+(d025-d024), p0245=(p245+p045)+(p025+p024),
    d0345=(d345-d045)+(d035-d034), p0345=(p345+p045)+(p035+p034),
    d1234=(d234-d134)+(d124-d123), p1234=(p234+p134)+(p124+p123),
    d1235=(d235-d135)+(d125-d123), p1235=(p235+p135)+(p125+p123),
    d1245=(d245-d145)+(d125-d124), p1245=(p245+p145)+(p125+p124),
    d1345=(d345-d145)+(d135-d134), p1345=(p345+p145)+(p135+p134),
    d2345=(d345-d245)+(d235-d234), p2345=(p345+p245)+(p235+p234);
    // error in dABCD is bounded by pABCD*8.1*DBL_EPSILON I believe
    
    // and the answers are the 5x5's, with an error less than permanent*11.1*DBL_EPSILON I believe
    orientations[0]= check_error((end1?d2345:0)-(end2?d1345:0)+(end3?d1245:0)-(end4?d1235:0)+(end5?d1234:0),
                                 (end1?p2345:0)+(end2?p1345:0)+(end3?p1245:0)+(end4?p1235:0)+(end5?p1234:0), 12);
    orientations[1]=-check_error((end0?d2345:0)-(end2?d0345:0)+(end3?d0245:0)-(end4?d0235:0)+(end5?d0234:0),
                                 (end0?p2345:0)+(end2?p0345:0)+(end3?p0245:0)+(end4?p0235:0)+(end5?p0234:0), 12);
    orientations[2]= check_error((end0?d1345:0)-(end1?d0345:0)+(end3?d0145:0)-(end4?d0135:0)+(end5?d0134:0),
                                 (end0?p1345:0)+(end1?p0345:0)+(end3?p0145:0)+(end4?p0135:0)+(end5?p0134:0), 12);
    orientations[3]=-check_error((end0?d1245:0)-(end1?d0245:0)+(end2?d0145:0)-(end4?d0125:0)+(end5?d0124:0),
                                 (end0?p1245:0)+(end1?p0245:0)+(end2?p0145:0)+(end4?p0125:0)+(end5?p0124:0), 12);
    orientations[4]= check_error((end0?d1235:0)-(end1?d0235:0)+(end2?d0135:0)-(end3?d0125:0)+(end5?d0123:0),
                                 (end0?p1235:0)+(end1?p0235:0)+(end2?p0135:0)+(end3?p0125:0)+(end5?p0123:0), 12);
    orientations[5]=-check_error((end0?d1234:0)-(end1?d0234:0)+(end2?d0134:0)-(end3?d0124:0)+(end4?d0123:0),
                                 (end0?p1234:0)+(end1?p0234:0)+(end2?p0134:0)+(end3?p0124:0)+(end4?p0123:0), 12);
    
    // self check
    double sumo=orientations[0]+orientations[1]+orientations[2]
    +orientations[3]+orientations[4]+orientations[5];
    double sumt=orientations[0]*end0+orientations[1]*end1+orientations[2]*end2
    +orientations[3]*end3+orientations[4]*end4+orientations[5]*end5;
    Vec3d sumx=orientations[0]*x0+ orientations[1]*x1+orientations[2]*x2+
    orientations[3]*x3+orientations[4]*x4+orientations[5]*x5;
    double rough_size=cube(mag(a0)+mag(a1)+mag(a2)+mag(a3));
    double rough_x=sqr(sqr(mag(a0)+mag(a1)+mag(a2)+mag(a3)));
    assert(std::fabs(sumo)<=1000*DBL_EPSILON*rough_size);
    assert(std::fabs(sumt)<=1000*DBL_EPSILON*rough_size);
    assert(mag(sumx)<=1000*DBL_EPSILON*rough_x);
}

//=======================================================================================================

// return a unit-length vector orthogonal to u and v
static Vec3d get_normal(const Vec3d& u, const Vec3d& v)
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
    c[0]=(random()%2 ? -0.577350269189626 : 0.577350269189626);
    c[1]=(random()%2 ? -0.577350269189626 : 0.577350269189626);
    c[2]=(random()%2 ? -0.577350269189626 : 0.577350269189626);
    return c;
}

//=======================================================================================================

// check 0 against 1-2-3
static bool segment_triangle_intersection(const Vec3d& x0, const Vec3d& xnew0,
                                          const Vec3d& x1, bool end1,
                                          const Vec3d& x2, bool end2,
                                          const Vec3d& x3, bool end3,
                                          double orientation, bool verbose = false )
{
    assert(end1 || end2 || end3);
    assert(!end1 || !end2 || !end3);
    // First want to know if we're in a skew situation - no possibility of intersection.
    if(orientation) return false;
    
    if ( verbose ) { std::cout << "checking projections" << std::endl; }
    
    return segment_triangle_intersection(Vec2d(x0[0],x0[1]),
                                         Vec2d(xnew0[0],xnew0[1]),
                                         Vec2d(x1[0],x1[1]), end1,
                                         Vec2d(x2[0],x2[1]), end2,
                                         Vec2d(x3[0],x3[1]), end3, verbose)
    && segment_triangle_intersection(Vec2d(x0[0],x0[2]),
                                     Vec2d(xnew0[0],xnew0[2]),
                                     Vec2d(x1[0],x1[2]), end1,
                                     Vec2d(x2[0],x2[2]), end2,
                                     Vec2d(x3[0],x3[2]), end3, verbose)
    && segment_triangle_intersection(Vec2d(x0[1],x0[2]),
                                     Vec2d(xnew0[1],xnew0[2]),
                                     Vec2d(x1[1],x1[2]), end1,
                                     Vec2d(x2[1],x2[2]), end2,
                                     Vec2d(x3[1],x3[2]), end3, verbose);
}

//=======================================================================================================

static bool segment_tetrahedron_intersection(const Vec3d& x0, const Vec3d& xnew0,
                                             const Vec3d& x1, bool end1,
                                             const Vec3d& x2, bool end2,
                                             const Vec3d& x3, bool end3,
                                             const Vec3d& x4, bool end4, bool )
{
    double o[6];
    orient(x0, false, xnew0, true, x1, end1, x2, end2, x3, end3, x4, end4, o);
    if((o[0]==0 || o[1]==0) || (o[2]==0 || o[3]==0 || o[4]==0 || o[5]==0)){ // check for degeneracy
        if(!(end1==end2 && end1==end3)
           && segment_triangle_intersection(x0, xnew0, x1, end1, x2, end2, x3, end3, o[5])) return true;
        if(!(end1==end2 && end1==end4)
           && segment_triangle_intersection(x0, xnew0, x1, end1, x2, end2, x4, end4, o[4])) return true;
        if(!(end1==end3 && end1==end4)
           && segment_triangle_intersection(x0, xnew0, x1, end1, x3, end3, x4, end4, o[3])) return true;
        if(!(end2==end3 && end2==end4)
           && segment_triangle_intersection(x0, xnew0, x2, end2, x3, end3, x4, end4, o[2])) return true;
        return false;
    }
    // otherwise, just some sign checks
    if((o[0]<0 && o[1]>0) || (o[0]>0 && o[1]<0)) return false;
    return (o[2]<=0 && o[3]<=0 && o[4]<=0 && o[5]<=0)
    || (o[2]>=0 && o[3]>=0 && o[4]>=0 && o[5]>=0);
}

static bool segment_tetrahedron_intersection(const Vec3d& x0, const Vec3d& xnew0,
                                             const Vec3d& x1, bool end1,
                                             const Vec3d& x2, bool end2,
                                             const Vec3d& x3, bool end3,
                                             const Vec3d& x4, bool end4,
                                             double bary[6], bool verbose )
{
    double o[6];
    orient(x0, false, xnew0, true, x1, end1, x2, end2, x3, end3, x4, end4, o);
    
    if ( verbose )
    {
        std::cout << "-------" << std::endl;
        std::cout << "orient: " << std::endl;
        std::cout << o[0] << std::endl;
        std::cout << o[1] << std::endl;
        std::cout << o[2] << std::endl;
        std::cout << o[3] << std::endl;
        std::cout << o[4] << std::endl;
        std::cout << o[5] << std::endl;
        std::cout << "-------" << std::endl;
    }
    
    if((o[0]==0 || o[1]==0) || (o[2]==0 || o[3]==0 || o[4]==0 || o[5]==0)){ // check for degeneracy
        // what do we do for bary in this case?
        // probably best to do something smart, but since degeneracies should be rare and not too
        // important to handle exactly, just assume equal barycentric coordinates at end of time step
        
        if ( verbose ) 
        { 
            std::cout << "degen \n ends:" << std::endl; 
            std::cout << end1 << std::endl;
            std::cout << end2 << std::endl;
            std::cout << end3 << std::endl;
            std::cout << end4 << std::endl;         
        }
        
        bary[0]=0;
        bary[1]=1;
        bary[2]=(double)end1/(double)(end1+end2+end3+end4);
        bary[3]=(double)end2/(double)(end1+end2+end3+end4);
        bary[4]=(double)end3/(double)(end1+end2+end3+end4);
        bary[5]=(double)end4/(double)(end1+end2+end3+end4);
        if(!(end1==end2 && end1==end3)
           && segment_triangle_intersection(x0, xnew0, x1, end1, x2, end2, x3, end3, o[5], verbose )) { if ( verbose ) { std::cout << "1" << std::endl; } return true; }
        if(!(end1==end2 && end1==end4)
           && segment_triangle_intersection(x0, xnew0, x1, end1, x2, end2, x4, end4, o[4], verbose )) { if ( verbose ) { std::cout << "2" << std::endl; } return true; }
        if(!(end1==end3 && end1==end4)
           && segment_triangle_intersection(x0, xnew0, x1, end1, x3, end3, x4, end4, o[3], verbose )) { if ( verbose ) { std::cout << "3" << std::endl; } return true; }
        if(!(end2==end3 && end2==end4)
           && segment_triangle_intersection(x0, xnew0, x2, end2, x3, end3, x4, end4, o[2], verbose )) { if ( verbose ) { std::cout << "4" << std::endl; } return true; }
        
        return false;
    }
    // otherwise, just some sign checks
    if((o[0]<0 && o[1]>0) || (o[0]>0 && o[1]<0)) return false;
    if((o[2]<=0 && o[3]<=0 && o[4]<=0 && o[5]<=0) || (o[2]>=0 && o[3]>=0 && o[4]>=0 && o[5]>=0)){
        bary[0]=o[0]/(o[0]+o[1]);
        bary[1]=1-bary[0];
        double sum=o[2]+o[3]+o[4]+o[5]; // actually should be identical to -(o[0]+o[1]) up to rounding error
        bary[2]=o[2]/sum;
        bary[3]=o[3]/sum;
        bary[4]=o[4]/sum;
        bary[5]=o[5]/sum;
        return true;
    }else
        return false;
}

// =================================================================================================================================

bool fe_point_triangle_collision(const Vec3d& x0, const Vec3d& xnew0,
                                 const Vec3d& x1, const Vec3d& xnew1,
                                 const Vec3d& x2, const Vec3d& xnew2,
                                 const Vec3d& x3, const Vec3d& xnew3)
{
    if(segment_tetrahedron_intersection(x0, xnew0, x1, false, x2, false, x3, false, xnew3, true, false )) return true;
    if(segment_tetrahedron_intersection(x0, xnew0, x1, false, x2, false, xnew2, true, xnew3, true, false)) return true;
    if(segment_tetrahedron_intersection(x0, xnew0, x1, false, xnew1, true, xnew2, true, xnew3, true, false)) return true;
    return false;
}

bool fe_point_triangle_collision(const Vec3d& x0, const Vec3d& xnew0,
                                 const Vec3d& x1, const Vec3d& xnew1,
                                 const Vec3d& x2, const Vec3d& xnew2,
                                 const Vec3d& x3, const Vec3d& xnew3,
                                 double& bary1, double& bary2, double& bary3,
                                 Vec3d& normal,
                                 double& relative_normal_displacement,
                                 bool verbose )
{
    
    double t;
    
    if ( verbose )
    {
        std::cout.precision(20);
        std::cout << x0 << std::endl;
        std::cout << xnew0 << std::endl;
        std::cout << x1 << std::endl;
        std::cout << xnew1 << std::endl;
        std::cout << x2 << std::endl;
        std::cout << xnew2 << std::endl;
        std::cout << x3 << std::endl;
        std::cout << xnew3 << std::endl;
        std::cout << std::endl;
        
        print_hex( x0 );
        print_hex( xnew0 );      
        print_hex( x1 );
        print_hex( xnew1 );      
        print_hex( x2 );
        print_hex( xnew2 );      
        print_hex( x3 );
        print_hex( xnew3 );      
        std::cout << std::endl;
    }
    
    
    bool collision=false;
    double bary[6];
    if(segment_tetrahedron_intersection(x0, xnew0, x1, false, x2, false, x3, false, xnew3, true, bary, verbose)){
        if ( verbose ) { std::cout << "seg-tet 1" << std::endl; }
        collision=true;
        bary1=0;
        bary2=0;
        bary3=1;
        t=bary[1];
        normal=get_normal(x2-x1, x3-x1);
        relative_normal_displacement=dot(normal, (xnew0-x0)-(xnew3-x3));
    }
    if(segment_tetrahedron_intersection(x0, xnew0, x1, false, x2, false, xnew2, true, xnew3, true, bary, verbose)){
        if(!collision || bary[1]<t){
            if ( verbose ) { std::cout << "seg-tet 2" << std::endl; }
            collision=true;
            bary1=0;
            bary2=0.5;//(bary[4]+1e-300)/(bary[4]+bary[5]+2e-300); // guard against zero/zero
            bary3=0.5;//1-bary2;         
            t=bary[1];
            normal=get_normal(x2-x1, xnew3-xnew2);
            relative_normal_displacement=dot(normal, (xnew0-x0)-(xnew2-x2));
        }
    }
    if(segment_tetrahedron_intersection(x0, xnew0, x1, false, xnew1, true, xnew2, true, xnew3, true, bary, verbose)){
        if(!collision || bary[1]<t){
            if ( verbose ) { std::cout << "seg-tet 3" << std::endl; }
            collision=true;
            bary1=0.3333333333333333333333333;//(bary[3]+1e-300)/(bary[3]+bary[4]+bary[5]+3e-300); // guard against zero/zero
            bary2=0.3333333333333333333333333;//(bary[4]+1e-300)/(bary[3]+bary[4]+bary[5]+3e-300); // guard against zero/zero
            bary3=0.3333333333333333333333333;//1-bary1-bary2;
            t=bary[1];
            normal=get_normal(xnew2-xnew1, xnew3-xnew1);
            relative_normal_displacement=dot(normal, (xnew0-x0)-(xnew1-x1));
        }
    }
    return collision;
}


//=======================================================================================================

static bool triangle_triangle_intersection(const Vec3d& x0, bool end0,
                                           const Vec3d& x1, bool end1,
                                           const Vec3d& x2, bool end2,
                                           const Vec3d& x3, bool end3,
                                           const Vec3d& x4, bool end4,
                                           const Vec3d& x5, bool end5)
{
    double o[6];
    orient(x0, end0, x1, end1, x2, end2, x3, end3, x4, end4, x5, end5, o);
    if((o[0]==0 || o[1]==0 || o[2]==0) || (o[3]==0 || o[4]==0 || o[5]==0)){ // check for degeneracy
        if(!end0 && end1 && segment_triangle_intersection(x0, x1, x3, end3, x4, end4, x5, end5, o[2])) return true;
        if(end0 && !end1 && segment_triangle_intersection(x1, x0, x3, end3, x4, end4, x5, end5, o[2])) return true;
        if(!end0 && end2 && segment_triangle_intersection(x0, x2, x3, end3, x4, end4, x5, end5, o[1])) return true;
        if(end0 && !end2 && segment_triangle_intersection(x2, x0, x3, end3, x4, end4, x5, end5, o[1])) return true;
        if(!end1 && end2 && segment_triangle_intersection(x1, x2, x3, end3, x4, end4, x5, end5, o[0])) return true;
        if(end1 && !end2 && segment_triangle_intersection(x2, x1, x3, end3, x4, end4, x5, end5, o[0])) return true;
        if(!end3 && end4 && segment_triangle_intersection(x3, x4, x0, end0, x1, end1, x2, end2, o[5])) return true;
        if(end3 && !end4 && segment_triangle_intersection(x4, x3, x0, end0, x1, end1, x2, end2, o[5])) return true;
        if(!end3 && end5 && segment_triangle_intersection(x3, x5, x0, end0, x1, end1, x2, end2, o[4])) return true;
        if(end3 && !end5 && segment_triangle_intersection(x5, x3, x0, end0, x1, end1, x2, end2, o[4])) return true;
        if(!end4 && end5 && segment_triangle_intersection(x4, x5, x0, end0, x1, end1, x2, end2, o[3])) return true;
        if(end4 && !end5 && segment_triangle_intersection(x5, x4, x0, end0, x1, end1, x2, end2, o[3])) return true;
        return false;
    }
    // otherwise, just some simple sign checks
    if((o[0]<=0 && o[1]<=0 && o[2]<=0) || (o[0]>=0 && o[1]>=0 && o[2]>=0))
        return (o[3]<=0 && o[4]<=0 && o[5]<=0) || (o[3]>=0 && o[4]>=0 && o[5]>=0);
    else
        return false;
}

static bool triangle_triangle_intersection(const Vec3d& x0, bool end0,
                                           const Vec3d& x1, bool end1,
                                           const Vec3d& x2, bool end2,
                                           const Vec3d& x3, bool end3,
                                           const Vec3d& x4, bool end4,
                                           const Vec3d& x5, bool end5,
                                           double bary[6])
{
    double o[6];
    orient(x0, end0, x1, end1, x2, end2, x3, end3, x4, end4, x5, end5, o);
    if((o[0]==0 || o[1]==0 || o[2]==0) || (o[3]==0 || o[4]==0 || o[5]==0)){ // check for degeneracy
        // what do we do for bary here?
        // probably best to do something smart, but since degeneracies should be rare and not too
        // important to handle exactly, just assume equal barycentric coordinates at the end of the time step
        bary[0]=(double)end0/(double)(end0+end1+end2); // I added the double cast on top for sanity 
        bary[1]=(double)end1/(double)(end0+end1+end2); // (and to silence warnings). Hopefully it doesn't break things.
        bary[2]=(double)end2/(double)(end0+end1+end2); // - Christopher
        bary[3]=(double)end3/(double)(end3+end4+end5);
        bary[4]=(double)end4/(double)(end3+end4+end5);
        bary[5]=(double)end5/(double)(end3+end4+end5);
        if(!end0 && end1 && segment_triangle_intersection(x0, x1, x3, end3, x4, end4, x5, end5, o[2])) return true;
        if(end0 && !end1 && segment_triangle_intersection(x1, x0, x3, end3, x4, end4, x5, end5, o[2])) return true;
        if(!end0 && end2 && segment_triangle_intersection(x0, x2, x3, end3, x4, end4, x5, end5, o[1])) return true;
        if(end0 && !end2 && segment_triangle_intersection(x2, x0, x3, end3, x4, end4, x5, end5, o[1])) return true;
        if(!end1 && end2 && segment_triangle_intersection(x1, x2, x3, end3, x4, end4, x5, end5, o[0])) return true;
        if(end1 && !end2 && segment_triangle_intersection(x2, x1, x3, end3, x4, end4, x5, end5, o[0])) return true;
        if(!end3 && end4 && segment_triangle_intersection(x3, x4, x0, end0, x1, end1, x2, end2, o[5])) return true;
        if(end3 && !end4 && segment_triangle_intersection(x4, x3, x0, end0, x1, end1, x2, end2, o[5])) return true;
        if(!end3 && end5 && segment_triangle_intersection(x3, x5, x0, end0, x1, end1, x2, end2, o[4])) return true;
        if(end3 && !end5 && segment_triangle_intersection(x5, x3, x0, end0, x1, end1, x2, end2, o[4])) return true;
        if(!end4 && end5 && segment_triangle_intersection(x4, x5, x0, end0, x1, end1, x2, end2, o[3])) return true;
        if(end4 && !end5 && segment_triangle_intersection(x5, x4, x0, end0, x1, end1, x2, end2, o[3])) return true;
        return false;
    }
    // otherwise, just some simple sign checks
    if((o[0]<=0 && o[1]<=0 && o[2]<=0) || (o[0]>=0 && o[1]>=0 && o[2]>=0)){
        if((o[3]<=0 && o[4]<=0 && o[5]<=0) || (o[3]>=0 && o[4]>=0 && o[5]>=0)){
            double sum=o[0]+o[1]+o[2];
            bary[0]=o[0]/sum;
            bary[1]=o[1]/sum;
            bary[2]=1-bary[0]-bary[1];
            sum=o[3]+o[4]+o[5]; // recompute for safety, but it should be the negative of the first sum up to rounding
            bary[3]=o[3]/sum;
            bary[4]=o[4]/sum;
            bary[5]=1-bary[3]-bary[4];
            return true;
        }else
            return false;
    }else
        return false;
}

bool fe_segment_segment_collision(const Vec3d& x0, const Vec3d& xnew0,
                                  const Vec3d& x1, const Vec3d& xnew1,
                                  const Vec3d& x2, const Vec3d& xnew2,
                                  const Vec3d& x3, const Vec3d& xnew3)
{
    if(triangle_triangle_intersection(x0, false, x1, false, xnew1, true, x2, false, x3, false, xnew3, true))
        return true;
    if(triangle_triangle_intersection(x0, false, xnew0, true, xnew1, true, x2, false, x3, false, xnew3, true))
        return true;
    if(triangle_triangle_intersection(x0, false, x1, false, xnew1, true, x2, false, xnew2, true, xnew3, true))
        return true;
    if(triangle_triangle_intersection(x0, false, xnew0, true, xnew1, true, x2, false, xnew2, true, xnew3, true))
        return true;
    return false;
}

bool fe_segment_segment_collision(const Vec3d& x0, const Vec3d& xnew0,
                                  const Vec3d& x1, const Vec3d& xnew1,
                                  const Vec3d& x2, const Vec3d& xnew2,
                                  const Vec3d& x3, const Vec3d& xnew3,
                                  double& bary0, double& bary2,
                                  Vec3d& normal,
                                  double& relative_normal_displacement,
                                  bool verbose )
{
    double t;
    
    if ( verbose )
    {
        std::cout << x0 << std::endl;
        std::cout << xnew0 << std::endl;
        std::cout << x1 << std::endl;
        std::cout << xnew1 << std::endl;
        std::cout << x2 << std::endl;
        std::cout << xnew2 << std::endl;
        std::cout << x3 << std::endl;
        std::cout << xnew3 << std::endl;
        std::cout << std::endl;
        
        print_hex( x0 );
        print_hex( xnew0 );      
        print_hex( x1 );
        print_hex( xnew1 );      
        print_hex( x2 );
        print_hex( xnew2 );      
        print_hex( x3 );
        print_hex( xnew3 );      
        std::cout << std::endl;
    }
    
    bool collision=false;
    double bary[6];
    if(triangle_triangle_intersection(x0, false, x1, false, xnew1, true, x2, false, x3, false, xnew3, true, bary)){
        collision=true;
        bary0=0;
        bary2=0;
        t=bary[2];
        normal=get_normal(x1-x0, x3-x2);
        relative_normal_displacement=dot(normal, (xnew1-x1)-(xnew3-x3));
    }
    if(triangle_triangle_intersection(x0, false, xnew0, true, xnew1, true, x2, false, x3, false, xnew3, true, bary)){
        if(!collision || bary[5]<t){
            collision=true;
            bary0=0.5;//(bary[1]+1e-300)/(bary[1]+bary[2]+2e-300); // guard against zero/zero
            bary2=0;
            t=bary[5];
            normal=get_normal(xnew1-xnew0, x3-x2);
            relative_normal_displacement=dot(normal, (xnew0-x0)-(xnew3-x3));
        }
    }
    if(triangle_triangle_intersection(x0, false, x1, false, xnew1, true, x2, false, xnew2, true, xnew3, true, bary)){
        if(!collision || bary[2]<t){
            collision=true;
            bary0=0;
            bary2=0.5;//(bary[4]+1e-300)/(bary[4]+bary[5]+2e-300); // guard against zero/zero
            t=bary[2];
            normal=get_normal(x1-x0, xnew3-xnew2);
            relative_normal_displacement=dot(normal, (xnew1-x1)-(xnew2-x2));
        }
    }
    if(triangle_triangle_intersection(x0, false, xnew0, true, xnew1, true, x2, false, xnew2, true, xnew3, true, bary)){
        if(!collision || 1-bary[0]<t){
            collision=true;
            bary0=0.5;//(bary[1]+1e-300)/(bary[1]+bary[2]+2e-300); // guard against zero/zero
            bary2=0.5;//(bary[4]+1e-300)/(bary[4]+bary[5]+2e-300); // guard against zero/zero
            t=1-bary[0];
            normal=get_normal(xnew1-xnew0, xnew3-xnew2);
            relative_normal_displacement=dot(normal, (xnew0-x0)-(xnew2-x2));
        }
    }
    return collision;
}


//=======================================================================================================

static bool segment_segment_intersection(const Vec3d& x0, const Vec3d& x1,
                                         const Vec3d& x2, const Vec3d& x3,
                                         double coplanarity)
{
    if(coplanarity) return false; // points are not coplanar, so segments are skew, therefore no intersection
    // now we know they're in a single plane in 3d; try all projections
    return fe_segment_segment_intersection(Vec2d(x0[0],x0[1]),
                                           Vec2d(x1[0],x1[1]),
                                           Vec2d(x2[0],x2[1]),
                                           Vec2d(x3[0],x3[1]))
    && fe_segment_segment_intersection(Vec2d(x0[0],x0[2]),
                                       Vec2d(x1[0],x1[2]),
                                       Vec2d(x2[0],x2[2]),
                                       Vec2d(x3[0],x3[2]))
    && fe_segment_segment_intersection(Vec2d(x0[1],x0[2]),
                                       Vec2d(x1[1],x1[2]),
                                       Vec2d(x2[1],x2[2]),
                                       Vec2d(x3[1],x3[2]));
}


bool fe_segment_triangle_intersection(const Vec3d& x0, const Vec3d& x1,
                                      const Vec3d& x2, const Vec3d& x3, const Vec3d& x4, 
                                      double& a, double& b, double& c, double& s, double& t, 
                                      bool /*degenerate_counts_as_intersection*/,
                                      bool /*verbose*/ )
{
    double d012, d013, d014, d023, d024, d034, d123, d124, d134, d234;
    d012=determinant(x0, x1, x2); d013=determinant(x0, x1, x3); d014=determinant(x0, x1, x4);
    d023=determinant(x0, x2, x3); d024=determinant(x0, x2, x4); d034=determinant(x0, x3, x4);
    d123=determinant(x1, x2, x3); d124=determinant(x1, x2, x4); d134=determinant(x1, x3, x4);
    d234=determinant(x2, x3, x4);
    double p012, p013, p014, p023, p024, p034, p123, p124, p134, p234;
    Vec3d a0(std::fabs(x0[0]), std::fabs(x0[1]), std::fabs(x0[2])),
    a1(std::fabs(x1[0]), std::fabs(x1[1]), std::fabs(x1[2])),
    a2(std::fabs(x2[0]), std::fabs(x2[1]), std::fabs(x2[2])),
    a3(std::fabs(x3[0]), std::fabs(x3[1]), std::fabs(x3[2])),
    a4(std::fabs(x4[0]), std::fabs(x4[1]), std::fabs(x4[2]));
    p012=permanent(a0, a1, a2); p013=permanent(a0, a1, a3); p014=permanent(a0, a1, a4);
    p023=permanent(a0, a2, a3); p024=permanent(a0, a2, a4); p034=permanent(a0, a3, a4);
    p123=permanent(a1, a2, a3); p124=permanent(a1, a2, a4); p134=permanent(a1, a3, a4);
    p234=permanent(a2, a3, a4);
    // error in dABC is certainly bounded by pABC*5.1*DBL_EPSILON
    
    // now do the 4x4's
    double d0123=(d123-d023)+(d013-d012), p0123=(p123+p023)+(p013+p012),
    d0124=(d124-d024)+(d014-d012), p0124=(p124+p024)+(p014+p012),
    d0134=(d134-d034)+(d014-d013), p0134=(p134+p034)+(p014+p013),
    d0234=(d234-d034)+(d024-d023), p0234=(p234+p034)+(p024+p023),
    d1234=(d234-d134)+(d124-d123), p1234=(p234+p134)+(p124+p123);
    // error in dABCD is certainly bounded by pABCD*8.1*DBL_EPSILON
    
    // and the actual values
    s=check_error(d1234, p1234, 9), t=-check_error(d0234, p0234, 9);
    if((s<0 && t>0) || (s>0 && t<0)) return false;
    a=check_error(d0134, p0134, 9),
    b=-check_error(d0124, p0124, 9),
    c=check_error(d0123, p0123, 9);
    if((s==0 || t==0) || (a==0 || b==0 || c==0)){ // check for degeneracy
        return segment_segment_intersection(x0, x1, x2, x3, c)
        || segment_segment_intersection(x0, x1, x2, x4, b)
        || segment_segment_intersection(x0, x1, x3, x4, a);
    }
    return (a<=0 && b<=0 && c<=0) || (a>=0 && b>=0 && c>=0);
}



bool fe_segment_triangle_intersection(const Vec3d& x0, const Vec3d& x1,
                                      const Vec3d& x2, const Vec3d& x3, const Vec3d& x4, 
                                      bool degenerate_counts_as_intersection,
                                      bool verbose )
{
    double a, b, c, s, t;
    return fe_segment_triangle_intersection( x0, x1, x2, x3, x4, a, b, c, s, t, degenerate_counts_as_intersection, verbose );
}




//=======================================================================================================

static bool point_triangle_intersection(const Vec2d& x0, const Vec2d& x1,
                                        const Vec2d& x2, const Vec2d& x3)
{
    double a=orient(x0, x2, x3), b=orient(x0, x1, x3), c=orient(x0, x1, x2);
    if(a==0 || b==0 || c==0){ // check for degeneracy
        return point_segment_intersection(x0, x1, x2)
        || point_segment_intersection(x0, x1, x3)
        || point_segment_intersection(x0, x2, x3);
    }
    return (a<=0 && b<=0 && c<=0) || (a>=0 && b>=0 && c>=0);
}


static bool point_triangle_intersection(const Vec3d& x0, const Vec3d& x1,
                                        const Vec3d& x2, const Vec3d& x3,
                                        double coplanarity)
{
    if(coplanarity) return false; // points are not coplanar, so point is not in triangle
    // now we know they're in a single plane in 3d - check all projections
    return point_triangle_intersection(Vec2d(x0[0],x0[1]),
                                       Vec2d(x1[0],x1[1]),
                                       Vec2d(x2[0],x2[1]),
                                       Vec2d(x3[0],x3[1]))
    && point_triangle_intersection(Vec2d(x0[0],x0[2]),
                                   Vec2d(x1[0],x1[2]),
                                   Vec2d(x2[0],x2[2]),
                                   Vec2d(x3[0],x3[2]))
    && point_triangle_intersection(Vec2d(x0[1],x0[2]),
                                   Vec2d(x1[1],x1[2]),
                                   Vec2d(x2[1],x2[2]),
                                   Vec2d(x3[1],x3[2]));
}

bool fe_point_tetrahedron_intersection(const Vec3d& x0, const Vec3d& x1,
                                       const Vec3d& x2, const Vec3d& x3, const Vec3d& x4)
{
    double d012, d013, d014, d023, d024, d034, d123, d124, d134, d234;
    d012=determinant(x0, x1, x2); d013=determinant(x0, x1, x3); d014=determinant(x0, x1, x4);
    d023=determinant(x0, x2, x3); d024=determinant(x0, x2, x4); d034=determinant(x0, x3, x4);
    d123=determinant(x1, x2, x3); d124=determinant(x1, x2, x4); d134=determinant(x1, x3, x4);
    d234=determinant(x2, x3, x4);
    double p012, p013, p014, p023, p024, p034, p123, p124, p134, p234;
    Vec3d a0(std::fabs(x0[0]), std::fabs(x0[1]), std::fabs(x0[2])),
    a1(std::fabs(x1[0]), std::fabs(x1[1]), std::fabs(x1[2])),
    a2(std::fabs(x2[0]), std::fabs(x2[1]), std::fabs(x2[2])),
    a3(std::fabs(x3[0]), std::fabs(x3[1]), std::fabs(x3[2])),
    a4(std::fabs(x4[0]), std::fabs(x4[1]), std::fabs(x4[2]));
    p012=permanent(a0, a1, a2); p013=permanent(a0, a1, a3); p014=permanent(a0, a1, a4);
    p023=permanent(a0, a2, a3); p024=permanent(a0, a2, a4); p034=permanent(a0, a3, a4);
    p123=permanent(a1, a2, a3); p124=permanent(a1, a2, a4); p134=permanent(a1, a3, a4);
    p234=permanent(a2, a3, a4);
    // error in dABC is certainly bounded by pABC*5.1*DBL_EPSILON
    
    // now do the 4x4's
    double d0123=(d123-d023)+(d013-d012), p0123=(p123+p023)+(p013+p012),
    d0124=(d124-d024)+(d014-d012), p0124=(p124+p024)+(p014+p012),
    d0134=(d134-d034)+(d014-d013), p0134=(p134+p034)+(p014+p013),
    d0234=(d234-d034)+(d024-d023), p0234=(p234+p034)+(p024+p023);
    // error in dABCD is certainly bounded by pABCD*8.1*DBL_EPSILON
    
    // and the actual values
    double a=check_error(d0234, p0234, 9),
    b=-check_error(d0134, p0134, 9),
    c=check_error(d0124, p0124, 9),
    d=-check_error(d0123, p0123, 9);
    if(a==0 || b==0 || c==0 || d==0){ // check for degeneracy
        return point_triangle_intersection(x0, x2, x3, x4, a)
        || point_triangle_intersection(x0, x1, x3, x4, b)
        || point_triangle_intersection(x0, x1, x2, x4, c)
        || point_triangle_intersection(x0, x1, x2, x3, d);
    }
    return (a<=0 && b<=0 && c<=0 && d<=0) || (a>=0 && b>=0 && c>=0 && d>=0);
}

}