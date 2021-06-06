
// ---------------------------------------------------------
//
//  cubic_ccd_wrapper.cpp
//  Tyson Brochu 2009
//  Christopher Batty, Fang Da 2014
//
//  Cubic solver-based implementation of collision and intersection queries.
//
// ---------------------------------------------------------


#include <ccd_defs.h>
#include <ccd_wrapper.h>

bool simplex_verbose = false;

#ifdef USE_CUBIC_SOLVER_CCD

#include <collisionqueries.h>

namespace LosTopos {


namespace
{
    
    //
    // Local function declarations
    //
    
    /// Tolerance on the cubic solver for coplanarity time
    const double g_cubic_solver_tol = 1e-8;
    
    /// Tolerance for trusting computed collision normal
    const double g_degen_normal_epsilon = 1e-6;
    
    /// Tolerance for static distance query at coplanarity time to be considered a collision
    const double g_collision_epsilon = 1e-6;
    
    /// Check if segment x0-x1 and segment x2-x3 are intersecting and return barycentric coordinates of intersection if so
    ///
    bool check_edge_edge_intersection(const Vec2d &x0, 
                                      const Vec2d &x1, 
                                      const Vec2d &x2, 
                                      const Vec2d &x3, 
                                      double &s01, 
                                      double &s23, 
                                      double tolerance );
    
    /// Find the roots in [0,1] of the specified quadratic (append to possible_t).
    ///
    void find_possible_quadratic_roots_in_01( double A, double B, double C, std::vector<double> &possible_t, double tol );


    /// Check if point x0 collides with segment x1-x2 during the motion from old to new positions.
    ///
    bool check_point_edge_collision(const Vec2d &x0old, const Vec2d &x1old, const Vec2d &x2old,
                                    const Vec2d &x0new, const Vec2d &x1new, const Vec2d &x2new,
                                    double collision_epsilon );
    
    /// Check if point x0 collides with segment x1-x2 during the motion from old to new positions. Return the barycentric 
    /// coordinates, collision normal, and time if so.
    ///
    bool check_point_edge_collision(const Vec2d &x0old, const Vec2d &x1old, const Vec2d &x2old,
                                    const Vec2d &x0new, const Vec2d &x1new, const Vec2d &x2new,
                                    double &s12, Vec2d &normal, double &collision_time, double collision_epsilon );

    /// Find the possible coplanarity times of the four vertices with trajectories specified by x and xnew in the range [0,1].
    ///
    void find_coplanarity_times(const Vec3d &x0, const Vec3d &x1, const Vec3d &x2, const Vec3d &x3,
                                const Vec3d &xnew0, const Vec3d &xnew1, const Vec3d &xnew2, const Vec3d &xnew3,
                                std::vector<double> &possible_times);

    /// Check if segment x0-x1 collides with segment x2-x3 during the motion from old to new positions.
    ///
    bool check_edge_edge_collision(const Vec3d &x0, const Vec3d &x1, const Vec3d &x2, const Vec3d &x3,
                                   const Vec3d &xnew0, const Vec3d &xnew1, const Vec3d &xnew2, const Vec3d &xnew3,
                                   double collision_epsilon);

    /// If the collision normal found during CCD testing is degenerate, this function will attempt to pick a suitable normal.
    ///
    void degenerate_get_edge_edge_collision_normal(const Vec3d &x0, const Vec3d &x1, const Vec3d &x2, const Vec3d &x3,
                                                   double s0, double s2, Vec3d& normal );

    /// Check if segment x0-x1 collides with segment x2-x3 during the motion from old to new positions. Return the barycentric 
    /// coordinates, collision normal, and time if so.
    ///
    bool check_edge_edge_collision(const Vec3d &x0, const Vec3d &x1, const Vec3d &x2, const Vec3d &x3,
                                   const Vec3d &xnew0, const Vec3d &xnew1, const Vec3d &xnew2, const Vec3d &xnew3,
                                   double &s0, double &s2, Vec3d &normal, double &t, double collision_epsilon);
    
    /// Check if point x0 collides with triangle x1-x2-x3 during the motion from old to new positions.
    ///
    bool check_point_triangle_collision(const Vec3d &x0, const Vec3d &x1, const Vec3d &x2, const Vec3d &x3,
                                        const Vec3d &xnew0, const Vec3d &xnew1, const Vec3d &xnew2, const Vec3d &xnew3,
                                        double collision_epsilon);

    /// If the collision normal found during CCD testing is degenerate, this function will attempt to pick a suitable normal.
    ///
    void degenerate_get_point_triangle_collision_normal(const Vec3d &x0, const Vec3d &x1, const Vec3d &x2, const Vec3d &x3,                                               
                                                        double &s1, double &s2, double &s3,
                                                        Vec3d& normal );

    /// Check if point x0 collides with triangle x1-x2-x3 during the motion from old to new positions.  Return the barycentric 
    /// coordinates, collision normal, and time if so.
    ///
    bool check_point_triangle_collision(const Vec3d &x0, const Vec3d &x1, const Vec3d &x2, const Vec3d &x3,
                                        const Vec3d &xnew0, const Vec3d &xnew1, const Vec3d &xnew2, const Vec3d &xnew3,
                                        double &s1, double &s2, double &s3, Vec3d &normal, double &t,
                                        double collision_epsilon);
            
    
    //
    // Local function definitions
    //

    // --------------------------------------------------------
    ///
    /// Check if segment x0-x1 and segment x2-x3 are intersecting and return barycentric coordinates of intersection if so.
    ///
    // --------------------------------------------------------
    
    bool check_edge_edge_intersection( const Vec2d &x0, const Vec2d &x1, const Vec2d &x2, const Vec2d &x3, double &s01, double &s23, double tolerance )
    {
        double x10=x1[0]-x0[0], y10=x1[1]-x0[1];
        double x31=x3[0]-x1[0], y31=x3[1]-x1[1];
        double x32=x3[0]-x2[0], y32=x3[1]-x2[1];
        double det=x32*y10-x10*y32;
        s01=(x31*y32-x32*y31)/det;
        if(s01 < -tolerance || s01 > 1+tolerance) return false;
        s23=(x31*y10-x10*y31)/det;
        if(s23< -tolerance || s23 > 1+tolerance) return false;
        // clamp
        if(s01<0) s01=0; else if(s01>1) s01=1;
        if(s23<0) s23=0; else if(s23>1) s23=1;
        return true;
    }
    
    // --------------------------------------------------------
    ///
    /// Find the roots in [0,1] of the specified quadratic (append to possible_t).
    ///
    // --------------------------------------------------------
    
    void find_possible_quadratic_roots_in_01( double A, double B, double C, std::vector<double> &possible_t, double tol )
    {
        if(A!=0){
            double discriminant=B*B-4*A*C;
            if(discriminant>0){
                double numer;
                if(B>0) numer=0.5*(-B-sqrt(discriminant));
                else    numer=0.5*(-B+sqrt(discriminant));
                double t0=numer/A, t1=C/numer;
                if(t0<t1){
                    if(t0>=-tol && t0<1)
                        possible_t.push_back(max(0.,t0));
                    if(t1>=-tol && t1<1)
                        possible_t.push_back(max(0.,t1));
                }else{
                    if(t1>=-tol && t1<1)
                        possible_t.push_back(max(0.,t1));
                    if(t0>=-tol && t0<1)
                        possible_t.push_back(max(0.,t0));
                }
            }else{
                double t=-B/(2*A); // the extremum of the quadratic
                if(t>=-tol && t<1)
                    possible_t.push_back(max(0.,t));
            }
        }else if(B!=0){
            double t=-C/B;
            if(t>=-tol && t<1)
                possible_t.push_back(max(0.,t));
        }
    }
    
    // --------------------------------------------------------
    ///
    /// Check if point x0 collides with segment x1-x2 during the motion from old to new positions.
    ///
    // --------------------------------------------------------
    
    bool check_point_edge_collision(const Vec2d &x0old, const Vec2d &x1old, const Vec2d &x2old,
                                    const Vec2d &x0new, const Vec2d &x1new, const Vec2d &x2new, 
                                    double collision_epsilon )
    {
        Vec2d x10=x1old-x0old, x20=x2old-x0old;
        Vec2d d10=(x1new-x0new)-x10, d20=(x2new-x0new)-x20;
        // figure out possible collision times to check
        std::vector<double> possible_t;
        double A=cross(d10,d20), B=cross(d10,x20)+cross(x10,d20), C=cross(x10,x20);
        find_possible_quadratic_roots_in_01(A, B, C, possible_t, collision_epsilon );
        possible_t.push_back(1); // always check the end
        // check proximities at possible collision times
        double proximity_tol=collision_epsilon*std::sqrt(mag2(x0old)+mag2(x0new)+mag2(x1old)+mag2(x1new)+mag2(x2new)+mag2(x2old));
        for(size_t i=0; i<possible_t.size(); ++i)
        {
            double collision_time=possible_t[i];
            double u=1-collision_time;
            double distance;
            check_point_edge_proximity( false, u*x0old+collision_time*x0new, u*x1old+collision_time*x1new, u*x2old+collision_time*x2new, distance );
            if(distance<=proximity_tol) return true;
        }
        return false;
    }
    
    // --------------------------------------------------------
    ///
    /// Check if point x0 collides with segment x1-x2 during the motion from old to new positions. Return the barycentric 
    /// coordinates, collision normal, and time if so.
    ///
    // --------------------------------------------------------
    
    bool check_point_edge_collision( const Vec2d &x0old, const Vec2d &x1old, const Vec2d &x2old,
                                    const Vec2d &x0new, const Vec2d &x1new, const Vec2d &x2new,
                                    double &s12, Vec2d &normal, double &collision_time, double tol )
    {
        Vec2d x10=x1old-x0old, x20=x2old-x0old;
        Vec2d d10=(x1new-x0new)-x10, d20=(x2new-x0new)-x20;
        // figure out possible collision times to check
        std::vector<double> possible_t;
        double A=cross(d10,d20), B=cross(d10,x20)+cross(x10,d20), C=cross(x10,x20);
        find_possible_quadratic_roots_in_01(A, B, C, possible_t, tol );
        possible_t.push_back(1); // always check the end
        // check proximities at possible collision times
        double proximity_tol=tol*std::sqrt(mag2(x0old)+mag2(x0new)+mag2(x1old)+mag2(x1new)+mag2(x2new)+mag2(x2old));
        for(size_t i=0; i<possible_t.size(); ++i)
        {
            collision_time=possible_t[i];
            double u=1-collision_time;
            double distance;
            check_point_edge_proximity( false, 
                                       u*x0old+collision_time*x0new, 
                                       u*x1old+collision_time*x1new, 
                                       u*x2old+collision_time*x2new,
                                       distance, s12, normal, 1.0 );
            
            if(distance<=proximity_tol) 
            {
                return true;
            }
        }
        return false;
    }
        
    // --------------------------------------------------------
    ///
    /// Find the possible coplanarity times of the four vertices with trajectories specified by x and xnew in the range [0,1].
    ///
    // --------------------------------------------------------
    
    void find_coplanarity_times(const Vec3d &x0, const Vec3d &x1, const Vec3d &x2, const Vec3d &x3,
                                const Vec3d &xnew0, const Vec3d &xnew1, const Vec3d &xnew2, const Vec3d &xnew3,
                                std::vector<double> &possible_times)
    {
        
        if ( simplex_verbose )
        {
            std::cout << "finding coplanarity times... " << std::endl;
        }
        
        possible_times.clear();
        
        // cubic coefficients, A*t^3+B*t^2+C*t+D (for t in [0,1])
        Vec3d x03=x0-x3, x13=x1-x3, x23=x2-x3;
        Vec3d v03=(xnew0-xnew3)-x03, v13=(xnew1-xnew3)-x13, v23=(xnew2-xnew3)-x23;
        double A=triple(v03,v13,v23),
        B=triple(x03,v13,v23)+triple(v03,x13,v23)+triple(v03,v13,x23),
        C=triple(x03,x13,v23)+triple(x03,v13,x23)+triple(v03,x13,x23),
        D=triple(x03,x13,x23);
        const double convergence_tol=g_cubic_solver_tol*(std::fabs(A)+std::fabs(B)+std::fabs(C)+std::fabs(D));
        
        // find intervals to check, or just solve it if it reduces to a quadratic =============================
        std::vector<double> interval_times;
        double discriminant=B*B-3*A*C; // of derivative of cubic, 3*A*t^2+2*B*t+C, divided by 4 for convenience
        if(discriminant<=0){ // monotone cubic: only one root in [0,1] possible
            
            if ( simplex_verbose ) { std::cout << "monotone cubic" << std::endl; }
            
            // so we just 
            interval_times.push_back(0);
            interval_times.push_back(1);
        }else{ // positive discriminant, B!=0
            if(A==0){ // the cubic is just a quadratic, B*t^2+C*t+D ========================================
                discriminant=C*C-4*B*D; // of the quadratic
                if(discriminant<=0){
                    double t=-C/(2*B);
                    if(t>=-g_cubic_solver_tol && t<=1+g_cubic_solver_tol){
                        t=clamp(t, 0., 1.);
                        if(std::fabs(LosTopos::signed_volume((1-t)*x0+t*xnew0, (1-t)*x1+t*xnew1, (1-t)*x2+t*xnew2, (1-t)*x3+t*xnew3))<convergence_tol)
                            possible_times.push_back(t);
                    }
                }else{ // two separate real roots
                    double t0, t1;
                    if(C>0) t0=(-C-std::sqrt(discriminant))/(2*B);
                    else    t0=(-C+std::sqrt(discriminant))/(2*B);
                    t1=D/(B*t0);
                    if(t1<t0) swap(t0,t1);
                    if(t0>=-g_cubic_solver_tol && t0<=1+g_cubic_solver_tol) possible_times.push_back(clamp(t0, 0., 1.));
                    if(t1>=-g_cubic_solver_tol && t1<=1+g_cubic_solver_tol) add_unique(possible_times, clamp(t1, 0., 1.));
                }
                
                if ( simplex_verbose )
                {
                    std::cout << "A == 0" << std::endl;
                    for ( size_t i = 0; i < possible_times.size(); ++i )
                    {
                        std::cout << "possible_time: " << possible_times[i] << std::endl;
                    }
                    std::cout << std::endl;
                }
                
                return;
            }else{ // cubic is not monotone: divide up [0,1] accordingly =====================================
                double t0, t1;
                if(B>0) t0=(-B-std::sqrt(discriminant))/(3*A);
                else    t0=(-B+std::sqrt(discriminant))/(3*A);
                t1=C/(3*A*t0);
                if(t1<t0) swap(t0,t1);
                
                if ( simplex_verbose ) { std::cout << "interval times: " << t0 << ", " << t1 << std::endl; }
                
                interval_times.push_back(0);
                if(t0>0 && t0<1)
                    interval_times.push_back(t0);
                if(t1>0 && t1<1)
                    interval_times.push_back(t1);
                interval_times.push_back(1);
            }
        }
        
        if ( simplex_verbose )
        {
            unsigned int n_samples = 20;
            double dt = 1.0 / (double)n_samples;
            double min_val = 1e30;
            for ( unsigned int i = 0; i < n_samples; ++i )
            {
                double sample_t = dt * i;
                double sample_val = LosTopos::signed_volume((1-sample_t)*x0+sample_t*xnew0,
                                                  (1-sample_t)*x1+sample_t*xnew1, 
                                                  (1-sample_t)*x2+sample_t*xnew2, 
                                                  (1-sample_t)*x3+sample_t*xnew3);
                
                std::cout << "sample_val: " << sample_val << std::endl;
                
                min_val = std::min( min_val, std::fabs(sample_val) );
            }
            std::cout << "min_val: " << min_val << std::endl;
        }   
        
        
        // look for roots in indicated intervals ==============================================================
        // evaluate coplanarity more accurately at each endpoint of the intervals
        std::vector<double> interval_values(interval_times.size());
        for(size_t i=0; i<interval_times.size(); ++i){
            double t=interval_times[i];
            interval_values[i]=LosTopos::signed_volume((1-t)*x0+t*xnew0, (1-t)*x1+t*xnew1, (1-t)*x2+t*xnew2, (1-t)*x3+t*xnew3);
            if ( simplex_verbose ) 
            {  
                std::cout << "interval time: " << t << ", value: " << interval_values[i] << std::endl; 
            }
        }
        
        if ( simplex_verbose ) 
        {  
            std::cout << "convergence_tol: " << convergence_tol << std::endl;
        }
        
        // first look for interval endpoints that are close enough to zero, without a sign change
        for(size_t i=0; i<interval_times.size(); ++i){
            if(interval_values[i]==0){
                possible_times.push_back(interval_times[i]);
            }else if(std::fabs(interval_values[i])<convergence_tol){
                if((i==0 || (interval_values[i-1]>=0 && interval_values[i]>=0) || (interval_values[i-1]<=0 && interval_values[i]<=0))    
                   &&(i==interval_times.size()-1 || (interval_values[i+1]>=0 && interval_values[i]>=0) || (interval_values[i+1]<=0 && interval_values[i]<=0))){
                    possible_times.push_back(interval_times[i]);
                }
            }
        }
        // and then search in intervals with a sign change
        for(size_t i=1; i<interval_times.size(); ++i){
            double tlo=interval_times[i-1], thi=interval_times[i], tmid;
            double vlo=interval_values[i-1], vhi=interval_values[i], vmid;
            if((vlo<0 && vhi>0) || (vlo>0 && vhi<0)){
                // start off with secant approximation (in case the cubic is actually linear)
                double alpha=vhi/(vhi-vlo);
                tmid=alpha*tlo+(1-alpha)*thi;
                int iteration=0;
                
                if ( simplex_verbose ) { std::cout << "cubic solver tol: " << 1e-2*convergence_tol << std::endl; }
                
                for(; iteration<50; ++iteration){
                    vmid=LosTopos::signed_volume((1-tmid)*x0+tmid*xnew0, (1-tmid)*x1+tmid*xnew1,
                                       (1-tmid)*x2+tmid*xnew2, (1-tmid)*x3+tmid*xnew3);
                    if(std::fabs(vmid)<1e-2*convergence_tol) break;
                    if((vlo<0 && vmid>0) || (vlo>0 && vmid<0)){ // if sign change between lo and mid
                        thi=tmid;
                        vhi=vmid;
                    }else{ // otherwise sign change between hi and mid
                        tlo=tmid;
                        vlo=vmid;
                    }
                    if(iteration%2) alpha=0.5; // sometimes go with bisection to guarantee we make progress
                    else alpha=vhi/(vhi-vlo); // other times go with secant to hopefully get there fast
                    tmid=alpha*tlo+(1-alpha)*thi;
                }
                if ( iteration >= 50 && simplex_verbose )
                {
                    std::cout << "cubic solve failed" << std::endl;
                }
                possible_times.push_back(tmid);
            }
        }
        sort(possible_times.begin(), possible_times.end());
        
        if ( simplex_verbose )
        {
            std::cout << "=================" << std::endl;
            
            for ( size_t i = 0; i < possible_times.size(); ++i )
            {
                std::cout << "possible_time: " << possible_times[i] << std::endl;
            }
            std::cout << std::endl;
        }
    }
    

    // --------------------------------------------------------
    ///
    /// Check if segment x0-x1 collides with segment x2-x3 during the motion from old to new positions.
    ///
    // --------------------------------------------------------
    
    bool check_edge_edge_collision(const Vec3d &x0, const Vec3d &x1, const Vec3d &x2, const Vec3d &x3,
                                   const Vec3d &xnew0, const Vec3d &xnew1, const Vec3d &xnew2, const Vec3d &xnew3,
                                   double collision_epsilon)
    {
        std::vector<double> possible_times;
        find_coplanarity_times(x0, x1, x2, x3, xnew0, xnew1, xnew2, xnew3, possible_times);
        for(size_t a=0; a<possible_times.size(); ++a){
            double t=possible_times[a];
            Vec3d xt0=(1-t)*x0+t*xnew0, xt1=(1-t)*x1+t*xnew1, xt2=(1-t)*x2+t*xnew2, xt3=(1-t)*x3+t*xnew3;
            double distance;
            LosTopos::check_edge_edge_proximity(xt0, xt1, xt2, xt3, distance);
            if(distance<collision_epsilon)
                return true;
        }
        return false;
    }
    
    
    // -------------------------------------------------------
    ///
    /// If the collision normal found during CCD testing is degenerate, this function will attempt to pick a suitable normal.
    ///
    // --------------------------------------------------------
    
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
        
    
    // ---------------------------------------------------------
    ///
    /// Check if segment x0-x1 collides with segment x2-x3 during the motion from old to new positions. Return the barycentric 
    /// coordinates, collision normal, and time if so.
    ///
    // --------------------------------------------------------
    
    bool check_edge_edge_collision(const Vec3d &x0, const Vec3d &x1, const Vec3d &x2, const Vec3d &x3,
                                   const Vec3d &xnew0, const Vec3d &xnew1, const Vec3d &xnew2, const Vec3d &xnew3,
                                   double &s0, double &s2, Vec3d &normal, double &t, double collision_epsilon)
    {
        std::vector<double> possible_times;
        find_coplanarity_times(x0, x1, x2, x3, xnew0, xnew1, xnew2, xnew3, possible_times);
        for(size_t a=0; a<possible_times.size(); ++a){
            t=possible_times[a];
            Vec3d xt0=(1-t)*x0+t*xnew0, xt1=(1-t)*x1+t*xnew1, xt2=(1-t)*x2+t*xnew2, xt3=(1-t)*x3+t*xnew3;
            double distance;
            LosTopos::check_edge_edge_proximity(xt0, xt1, xt2, xt3, distance, s0, s2, normal);
            if(distance<collision_epsilon){
                // now figure out a decent normal
                if(distance<1e-2*g_degen_normal_epsilon){ // if we don't trust the normal...
                    // first try the cross-product of edges at collision time
                    normal=cross(xt1-xt0, xt3-xt2);
                    double m=mag(normal);
                    if(m>sqr(g_degen_normal_epsilon)){
                        normal/=m;
                    }else
                    {
                        degenerate_get_edge_edge_collision_normal( x0, x1, x2, x3, s0, s2, normal );
                    }
                }
                return true;
            }
        }
        return false;
    }
    
    // ---------------------------------------------------------
    ///
    /// Check if point x0 collides with triangle x1-x2-x3 during the motion from old to new positions.
    ///
    // --------------------------------------------------------
    
    bool check_point_triangle_collision(const Vec3d &x0, const Vec3d &x1, const Vec3d &x2, const Vec3d &x3,
                                        const Vec3d &xnew0, const Vec3d &xnew1, const Vec3d &xnew2, const Vec3d &xnew3,
                                        double collision_epsilon)
    {
        std::vector<double> possible_times;
        find_coplanarity_times(x0, x1, x2, x3, xnew0, xnew1, xnew2, xnew3, possible_times);
        for(size_t a=0; a<possible_times.size(); ++a){
            double t=possible_times[a];
            Vec3d xt0=(1-t)*x0+t*xnew0, xt1=(1-t)*x1+t*xnew1, xt2=(1-t)*x2+t*xnew2, xt3=(1-t)*x3+t*xnew3;
            double distance;
            LosTopos::check_point_triangle_proximity(xt0, xt1, xt2, xt3, distance);
            if(distance<collision_epsilon)
                return true;
        }
        return false;
    }
    
    // --------------------------------------------------------
    ///
    /// If the collision normal found during CCD testing is degenerate, this function will attempt to pick a suitable normal.
    ///
    // ---------------------------------------------------------
    
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
    
    
    // ---------------------------------------------------------
    ///
    /// Check if point x0 collides with triangle x1-x2-x3 during the motion from old to new positions.  Return the barycentric 
    /// coordinates, collision normal, and time if so.
    ///
    // --------------------------------------------------------
    
    bool check_point_triangle_collision(const Vec3d &x0, const Vec3d &x1, const Vec3d &x2, const Vec3d &x3,
                                        const Vec3d &xnew0, const Vec3d &xnew1, const Vec3d &xnew2, const Vec3d &xnew3,
                                        double &s1, double &s2, double &s3, Vec3d &normal, double &t, double collision_epsilon)
    {
        std::vector<double> possible_times;
        find_coplanarity_times(x0, x1, x2, x3, xnew0, xnew1, xnew2, xnew3, possible_times);
        
        for(size_t a=0; a<possible_times.size(); ++a){
            t=possible_times[a];
            Vec3d xt0=(1-t)*x0+t*xnew0, xt1=(1-t)*x1+t*xnew1, xt2=(1-t)*x2+t*xnew2, xt3=(1-t)*x3+t*xnew3;
            double distance;
            LosTopos::check_point_triangle_proximity(xt0, xt1, xt2, xt3, distance, s1, s2, s3, normal);
            if(distance<collision_epsilon){
                // now figure out a decent normal
                if(distance<1e-2*g_degen_normal_epsilon)
                { // if we don't trust the normal...
                    // first try the triangle normal at collision time
                    normal=cross(xt2-xt1, xt3-xt1);
                    double m=mag(normal);
                    if(m>sqr(g_degen_normal_epsilon)){
                        normal/=m;
                    }
                    else
                    {
                        degenerate_get_point_triangle_collision_normal( x0, x1, x2, x3, s1, s2, s3, normal );               
                    }
                }
                return true;
            }
        }
        return false;
    }
     
    
    // ---------------------------------------------------------
    ///
    /// Return true if triangle (x1,x2,x3) intersects segment (x4,x5)
    ///
    // ---------------------------------------------------------
    
    // TODO: Swap segment-triangle order
    
    bool triangle_intersects_segment(const Vec3d &x1, const Vec3d &x2, const Vec3d &x3, 
                                     const Vec3d &x4, const Vec3d &x5,
                                     double a, double b, double c, double d, double e,
                                     double /*tolerance*/, bool verbose, bool& degenerate )
    {
        static const double machine_epsilon = 1e-7;
        
        degenerate = false;
        
        d=LosTopos::signed_volume(x1, x2, x3, x5);
        e=-LosTopos::signed_volume(x1, x2, x3, x4);
        
        if ( verbose )
        {
            std::cout << "d: " << d << std::endl;
            std::cout << "e: " << e << std::endl;
        }
        
        if ( ( std::fabs(d) < machine_epsilon ) || ( std::fabs(e) < machine_epsilon ) )
        {
            if ( verbose )
            {
                std::cout << "degenerate: d = " << d << ", e = " << e << std::endl;
            }
            degenerate = true;
        }
        
        if((d>0) ^ (e>0))
        {
            return false;
        }
        
        // note: using the triangle edges in the first two spots guarantees the same floating point result (up to sign)
        // if we transpose the triangle vertices -- e.g. testing an adjacent triangle -- so this triangle-line test is
        // watertight.
        a=LosTopos::signed_volume(x2, x3, x4, x5);
        b=LosTopos::signed_volume(x3, x1, x4, x5);
        c=LosTopos::signed_volume(x1, x2, x4, x5);
        
        if ( verbose )
        {
            std::cout << "a: " << a << std::endl;
            std::cout << "b: " << b << std::endl;
            std::cout << "c: " << c << std::endl;
        }
        
        double sum_abc=a+b+c;
        
        if ( verbose ) std::cout << "sum_abc: " << sum_abc << std::endl;
        
        if( std::fabs(sum_abc) < machine_epsilon )
        {
            if ( verbose ) { std::cout << "sum_abc degenerate" << std::endl;  }      
            degenerate = true;
            return false;            // degenerate situation
        }
        
        double sum_de=d+e;
        
        if ( verbose ) std::cout << "sum_de: " << sum_de << std::endl;
        
        if( std::fabs(sum_de) < machine_epsilon )
        {
            if ( verbose ) { std::cout << "sum_de degenerate" << std::endl;  }            
            degenerate = true;
            return false; // degenerate situation
        }
        
        
        if ( ( std::fabs(a) < machine_epsilon ) || ( std::fabs(b) < machine_epsilon ) || (std::fabs(c) < machine_epsilon) )
        {
            if ( verbose ) { std::cout << "degenerate: a = " << a << ", b = " << b << ", c = " << c << std::endl;  }            
            degenerate = true;
        }
        
        
        if((a>0) ^ (b>0))
        {
            return false;
        }
        
        if((a>0) ^ (c>0))
        {
            return false;
        }
        
        double over_abc=1/sum_abc;
        a*=over_abc;
        b*=over_abc;
        c*=over_abc;
        
        double over_de=1/sum_de;
        d*=over_de;
        e*=over_de;
        
        if ( verbose ) 
        {
            std::cout << "normalized coords: " << a << " " << b << " " << c << " " << d << " " << e << std::endl;
        }
        
        return true;
    }
    
    // ---------------------------------------------------------
    ///
    /// Return true if triangle (xtri0,xtri1,xtri2) intersects segment (xedge0, xedge1), within the specified tolerance.
    /// If degenerate_counts_as_intersection is true, this function will return true in a degenerate situation.
    ///
    // ---------------------------------------------------------
    
    bool check_edge_triangle_intersection(const Vec3d &xedge0, const Vec3d &xedge1,
                                          const Vec3d &xtri0, const Vec3d &xtri1, const Vec3d &xtri2,
                                          double bary_e0, double bary_e1, double bary_t0, double bary_t1, double bary_t2,                                          
                                          double tolerance, bool degenerate_counts_as_intersection, bool verbose )
    {
        bool is_degenerate;
        if ( triangle_intersects_segment( xtri0, xtri1, xtri2, xedge0, xedge1, 
                                          bary_t0, bary_t1, bary_t2, bary_e0, bary_e1, 
                                          tolerance, verbose, is_degenerate ) )
        {
            if ( is_degenerate )
            {
                // we think we have an intersection, but it's a degenerate case
                
                if ( degenerate_counts_as_intersection )
                {
                    return true;
                }
                else
                {
                    return false;
                }
                
            }
            else
            {
                return true;
            }
        }
        
        return false;
    }
    
    
    // ---------------------------------------------------------
    ///
    /// Detect if point p lies within the tetrahedron defined by x1 x2 x3 x4.
    /// Assumes tet is given with x123 forming an oriented triangle.
    /// Returns true if vertex proximity to any of the tet's faces is less than epsilon.
    ///
    // ---------------------------------------------------------
    
    bool vertex_is_in_tetrahedron(const Vec3d &p, 
                                  const Vec3d &x1, const Vec3d &x2, const Vec3d &x3, const Vec3d &x4, double epsilon )
    {
        double distance;  
        
        // triangle 1 - x1 x2 x3
        double a = LosTopos::signed_volume(p, x1, x2, x3);
        
        if (std::fabs(a) < epsilon)     // degenerate
        {        
            LosTopos::check_point_triangle_proximity(p, x1, x2, x3, distance);
            if ( distance < epsilon )
            {
                return true;
            }
        }
        
        // triangle 2 - x2 x4 x3
        double b = LosTopos::signed_volume(p, x2, x4, x3);
        
        if (std::fabs(b) < epsilon)         // degenerate
        {
            LosTopos::check_point_triangle_proximity(p, x2, x4, x3, distance);
            if ( distance < epsilon )
            {
                return true;
            }
        }
        
        if ((a > epsilon) ^ (b > epsilon))
        {
            return false;
        }
        
        // triangle 3 - x1 x4 x2
        double c = LosTopos::signed_volume(p, x1, x4, x2);
        if (std::fabs(c) < epsilon) 
        {
            LosTopos::check_point_triangle_proximity(p, x1, x4, x2, distance);
            if ( distance < epsilon )
            {
                return true;
            }
        }
        
        if ((a > epsilon) ^ (c > epsilon))
        {
            return false;
        }
        
        // triangle 4 - x1 x3 x4
        double d = LosTopos::signed_volume(p, x1, x3, x4);
        if (std::fabs(d) < epsilon) 
        { 
            LosTopos::check_point_triangle_proximity(p, x1, x3, x4, distance);
            if ( distance < epsilon )
            {
                return true;
            }
        }
        
        if ((a > epsilon) ^ (d > epsilon))
        {
            return false;
        }
        
        // if there was a degenerate case, but the point was not in any triangle, the point must be outside the tet
        if ( (std::fabs(a) < epsilon) || (std::fabs(b) < epsilon) || (std::fabs(c) < epsilon) || (std::fabs(d) < epsilon) ) 
        {
            return false;
        }
        
        return true;    // point is on the same side of all triangles
    }
    

    
} // namespace


// --------------------------------------------------------------------------------------------------
// 2D Continuous collision detection
// --------------------------------------------------------------------------------------------------

bool point_segment_collision(const Vec2d& x0, const Vec2d& xnew0, size_t ,
                             const Vec2d& x1, const Vec2d& xnew1, size_t ,
                             const Vec2d& x2, const Vec2d& xnew2, size_t ,
                             double& edge_alpha, Vec2d& normal, double& rel_disp)
{
    double t;
    bool result = check_point_edge_collision( x0, x1, x2, xnew0, xnew1, xnew2, edge_alpha, normal, t, g_collision_epsilon );
    
    if ( result )
    {
        Vec2d dx0 = xnew0 - x0;
        Vec2d dx1 = xnew1 - x1;
        Vec2d dx2 = xnew2 - x2;
        rel_disp = dot( normal, dx0 - (edge_alpha)*dx1 - (1.0-edge_alpha)*dx2 );
    }
    
    return result;
}

bool point_segment_collision(const Vec2d& x0, const Vec2d& xnew0, size_t ,
                             const Vec2d& x1, const Vec2d& xnew1, size_t ,
                             const Vec2d& x2, const Vec2d& xnew2, size_t  )
{
    bool result = check_point_edge_collision( x0, x1, x2, xnew0, xnew1, xnew2, g_collision_epsilon );
    return result;
}

// --------------------------------------------------------------------------------------------------
// 2D Static intersection detection
// --------------------------------------------------------------------------------------------------


bool segment_segment_intersection(const Vec2d& x0, size_t , 
                                  const Vec2d& x1, size_t ,
                                  const Vec2d& x2, size_t ,
                                  const Vec2d& x3, size_t )
{
    double s0, s2;
    return check_edge_edge_intersection( x0, x1, x2, x3, s0, s2, g_collision_epsilon );
}


bool segment_segment_intersection(const Vec2d& x0, size_t , 
                                  const Vec2d& x1, size_t ,
                                  const Vec2d& x2, size_t ,
                                  const Vec2d& x3, size_t ,
                                  double &s0, double& s2 )
{
    return check_edge_edge_intersection( x0, x1, x2, x3, s0, s2, g_collision_epsilon );
}


// --------------------------------------------------------------------------------------------------
// 3D Continuous collision detection
// --------------------------------------------------------------------------------------------------


bool point_triangle_collision(const Vec3d& x0, const Vec3d& xnew0, size_t ,
                              const Vec3d& x1, const Vec3d& xnew1, size_t ,
                              const Vec3d& x2, const Vec3d& xnew2, size_t ,
                              const Vec3d& x3, const Vec3d& xnew3, size_t  )
{   
    bool cubic_result = check_point_triangle_collision( x0, x1, x2, x3, xnew0, xnew1, xnew2, xnew3, g_collision_epsilon );
    return cubic_result;
}


bool point_triangle_collision(const Vec3d& x0, const Vec3d& xnew0, size_t ,
                              const Vec3d& x1, const Vec3d& xnew1, size_t ,
                              const Vec3d& x2, const Vec3d& xnew2, size_t ,
                              const Vec3d& x3, const Vec3d& xnew3, size_t ,
                              double& bary1, double& bary2, double& bary3,
                              Vec3d& normal,
                              double& relative_normal_displacement )
{
    
    double t;
    bool cubic_result = check_point_triangle_collision( x0, x1, x2, x3, 
                                                       xnew0, xnew1, xnew2, xnew3,
                                                       bary1, bary2, bary3,
                                                       normal, t, g_collision_epsilon );
    
    Vec3d dx0 = xnew0 - x0;
    Vec3d dx1 = xnew1 - x1;
    Vec3d dx2 = xnew2 - x2;
    Vec3d dx3 = xnew3 - x3;   
    relative_normal_displacement = dot( normal, dx0 - bary1*dx1 - bary2*dx2 - bary3*dx3 );
    
    return cubic_result;
}


bool segment_segment_collision(const Vec3d& x0, const Vec3d& xnew0, size_t ,
                               const Vec3d& x1, const Vec3d& xnew1, size_t ,
                               const Vec3d& x2, const Vec3d& xnew2, size_t ,
                               const Vec3d& x3, const Vec3d& xnew3, size_t )
{
    bool cubic_result = check_edge_edge_collision( x0, x1, x2, x3,
                                                  xnew0, xnew1, xnew2, xnew3,
                                                  g_collision_epsilon );
    
    return cubic_result;
}


bool segment_segment_collision(const Vec3d& x0, const Vec3d& xnew0, size_t ,
                               const Vec3d& x1, const Vec3d& xnew1, size_t ,
                               const Vec3d& x2, const Vec3d& xnew2, size_t ,
                               const Vec3d& x3, const Vec3d& xnew3, size_t ,
                               double& bary0, double& bary2,
                               Vec3d& normal,
                               double& relative_normal_displacement )
{
    double t;
    bool cubic_result = check_edge_edge_collision( x0, x1, x2, x3,
                                                  xnew0, xnew1, xnew2, xnew3,
                                                  bary0, bary2,
                                                  normal, t, 
                                                  g_collision_epsilon );
    
    Vec3d dx0 = xnew0 - x0;
    Vec3d dx1 = xnew1 - x1;
    Vec3d dx2 = xnew2 - x2;
    Vec3d dx3 = xnew3 - x3;   
    
    relative_normal_displacement = dot( normal, bary0*dx0 + (1.0-bary0)*dx1 - bary2*dx2 - (1.0-bary2)*dx3 );
    
    return cubic_result;
}


// --------------------------------------------------------------------------------------------------
// 3D Static intersection detection
// --------------------------------------------------------------------------------------------------

/// x0-x1 is the segment and and x2-x3-x4 is the triangle.
bool segment_triangle_intersection(const Vec3d& x0, size_t ,
                                   const Vec3d& x1, size_t ,
                                   const Vec3d& x2, size_t ,
                                   const Vec3d& x3, size_t ,
                                   const Vec3d& x4, size_t ,
                                   bool degenerate_counts_as_intersection,
                                   bool verbose )
{
    double bary[5] = {0,0,0,0,0};
    return check_edge_triangle_intersection( x0, x1, x2, x3, x4, 
                                             bary[0], bary[1], bary[2], bary[3], bary[4],
                                             g_collision_epsilon, degenerate_counts_as_intersection, verbose );
    
}

/// x0-x1 is the segment and and x2-x3-x4 is the triangle.
bool segment_triangle_intersection(const Vec3d& x0, size_t ,
                                   const Vec3d& x1, size_t ,
                                   const Vec3d& x2, size_t ,
                                   const Vec3d& x3, size_t ,
                                   const Vec3d& x4, size_t ,
                                   double& bary0, double& bary1, double& bary2, double& bary3, double& bary4,
                                   bool degenerate_counts_as_intersection,
                                   bool verbose )
{
    return check_edge_triangle_intersection( x0, x1, x2, x3, x4, 
                                            bary0, bary1, bary2, bary3, bary4,
                                            g_collision_epsilon, degenerate_counts_as_intersection, verbose );    
}


/// x0 is the point and x1-x2-x3-x4 is the tetrahedron. Order is irrelevant.
bool point_tetrahedron_intersection(const Vec3d& x0, size_t ,
                                    const Vec3d& x1, size_t ,
                                    const Vec3d& x2, size_t ,
                                    const Vec3d& x3, size_t ,
                                    const Vec3d& x4, size_t )
{
    return vertex_is_in_tetrahedron(x0,x1,x2,x3,x4,g_collision_epsilon);
}

}

#endif



