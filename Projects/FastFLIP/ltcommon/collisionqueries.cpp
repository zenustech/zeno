#include <collisionqueries.h>
#include <commonoptions.h>
#include "ccd_wrapper.h"

namespace LosTopos {

void check_point_edge_proximity(bool update, const Vec3d &x0, const Vec3d &x1, const Vec3d &x2,
                                double &distance)
{
    Vec3d dx(x2-x1);
    double m2=mag2(dx);
    // find parameter value of closest point on segment
    double s=clamp(dot(x2-x0, dx)/m2, 0., 1.);
    // and find the distance
    if(update){
        distance=min(distance, dist(x0,s*x1+(1-s)*x2));
    }else{
        distance=dist(x0,s*x1+(1-s)*x2);
    }
}

// normal is from 1-2 towards 0, unless normal_multiplier<0
void check_point_edge_proximity(bool update, const Vec3d &x0, const Vec3d &x1, const Vec3d &x2,
                                double &distance, double &s, Vec3d &normal, double normal_multiplier)
{
    Vec3d dx(x2-x1);
    double m2=mag2(dx);
    if(update){
        // find parameter value of closest point on segment
        double this_s=clamp(dot(x2-x0, dx)/m2, 0., 1.);
        // and find the distance
        Vec3d this_normal=x0-(this_s*x1+(1-this_s)*x2);
        double this_distance=mag(this_normal);
        if(this_distance<distance){
            s=this_s;
            distance=this_distance;
            normal=(normal_multiplier/(this_distance+1e-30))*this_normal;
        }
    }else{
        // find parameter value of closest point on segment
        s=clamp(dot(x2-x0, dx)/m2, 0., 1.);
        // and find the distance
        normal=x0-(s*x1+(1-s)*x2);
        distance=mag(normal);
        normal*=normal_multiplier/(distance+1e-30);
    }
}

void check_point_edge_proximity( bool update, const Vec2d &x0, const Vec2d &x1, const Vec2d &x2,
                                double &distance)
{
    Vec2d dx(x2-x1);
    double m2=mag2(dx);
    // find parameter value of closest point on segment
    double s=clamp(dot(x2-x0, dx)/m2, 0., 1.);
    // and find the distance
    if(update){
        distance=min(distance, dist(x0,s*x1+(1-s)*x2));
    }else{
        distance=dist(x0, s*x1+(1-s)*x2);
    }
}

// normal is from 1-2 towards 0, unless normal_multiplier<0
void check_point_edge_proximity(bool update, const Vec2d &x0, const Vec2d &x1, const Vec2d &x2,
                                double &distance, double &s, Vec2d &normal, double normal_multiplier)
{
    Vec2d dx(x2-x1);
    double m2=mag2(dx);
    if(update){
        // find parameter value of closest point on segment
        double this_s=clamp(dot(x2-x0, dx)/m2, 0., 1.);
        // and find the distance
        Vec2d this_normal=x0-(this_s*x1+(1-this_s)*x2);
        double this_distance=mag(this_normal);
        if(this_distance<distance){
            s=this_s;
            distance=this_distance;
            normal=(normal_multiplier/(this_distance+1e-30))*this_normal;
        }
    }else{
        // find parameter value of closest point on segment
        s=clamp(dot(x2-x0, dx)/m2, 0., 1.);
        // and find the distance
        normal=x0-(s*x1+(1-s)*x2);
        distance=mag(normal);
        if ( distance < 1e-10 )
        {
            normal = normalized(perp(x2 - x1));
            return;
        }
        
        normal*=normal_multiplier/(distance+1e-30);
    }
}

void check_edge_edge_proximity(const Vec3d &x0, const Vec3d &x1, const Vec3d &x2, const Vec3d &x3, double &distance)
{
    // let's do it the QR way for added robustness
    Vec3d x01=x0-x1;
    double r00=mag(x01)+1e-30;
    x01/=r00;
    Vec3d x32=x3-x2;
    double r01=dot(x32,x01);
    x32-=r01*x01;
    double r11=mag(x32)+1e-30;
    x32/=r11;
    Vec3d x31=x3-x1;
    double s2=dot(x32,x31)/r11;
    double s0=(dot(x01,x31)-r01*s2)/r00;
    // check if we're in range
    if(s0<0){
        if(s2<0){
            // check both x1 against 2-3 and 3 against 0-1
            check_point_edge_proximity(false, x1, x2, x3, distance);
            check_point_edge_proximity(true, x3, x0, x1, distance);
        }else if(s2>1){
            // check both x1 against 2-3 and 2 against 0-1
            check_point_edge_proximity(false, x1, x2, x3, distance);
            check_point_edge_proximity(true, x2, x0, x1, distance);
        }else{
            s0=0;
            // check x1 against 2-3
            check_point_edge_proximity(false, x1, x2, x3, distance);
        }
    }else if(s0>1){
        if(s2<0){
            // check both x0 against 2-3 and 3 against 0-1
            check_point_edge_proximity(false, x0, x2, x3, distance);
            check_point_edge_proximity(true, x3, x0, x1, distance);
        }else if(s2>1){
            // check both x0 against 2-3 and 2 against 0-1
            check_point_edge_proximity(false, x0, x2, x3, distance);
            check_point_edge_proximity(true, x2, x0, x1, distance);
        }else{
            s0=1;
            // check x0 against 2-3
            check_point_edge_proximity(false, x0, x2, x3, distance);
        }
    }else{
        if(s2<0){
            s2=0;
            // check x3 against 0-1
            check_point_edge_proximity(false, x3, x0, x1, distance);
        }else if(s2>1){
            s2=1;
            // check x2 against 0-1
            check_point_edge_proximity(false, x2, x0, x1, distance);
        }else{ // we already got the closest points!
            distance=dist(s2*x2+(1-s2)*x3, s0*x0+(1-s0)*x1);
        }
    }
}

// find distance between 0-1 and 2-3, with barycentric coordinates for closest points, and
// a normal that points from 0-1 towards 2-3 (unreliable if distance==0 or very small)
void check_edge_edge_proximity(const Vec3d &x0, const Vec3d &x1, const Vec3d &x2, const Vec3d &x3,
                               double &distance, double &s0, double &s2, Vec3d &normal)
{
    // let's do it the QR way for added robustness
    Vec3d x01=x0-x1;
    double r00=mag(x01)+1e-30;
    x01/=r00;
    Vec3d x32=x3-x2;
    double r01=dot(x32,x01);
    x32-=r01*x01;
    double r11=mag(x32)+1e-30;
    x32/=r11;
    Vec3d x31=x3-x1;
    s2=dot(x32,x31)/r11;
    s0=(dot(x01,x31)-r01*s2)/r00;
    // check if we're in range
    if(s0<0){
        if(s2<0){
            // check both x1 against 2-3 and 3 against 0-1
            check_point_edge_proximity(false, x1, x2, x3, distance, s2, normal, 1.);
            check_point_edge_proximity(true, x3, x0, x1, distance, s0, normal, -1.);
        }else if(s2>1){
            // check both x1 against 2-3 and 2 against 0-1
            check_point_edge_proximity(false, x1, x2, x3, distance, s2, normal, 1.);
            check_point_edge_proximity(true, x2, x0, x1, distance, s0, normal, -1.);
        }else{
            s0=0;
            // check x1 against 2-3
            check_point_edge_proximity(false, x1, x2, x3, distance, s2, normal, 1.);
        }
    }else if(s0>1){
        if(s2<0){
            // check both x0 against 2-3 and 3 against 0-1
            check_point_edge_proximity(false, x0, x2, x3, distance, s2, normal, 1.);
            check_point_edge_proximity(true, x3, x0, x1, distance, s0, normal, -1.);
        }else if(s2>1){
            // check both x0 against 2-3 and 2 against 0-1
            check_point_edge_proximity(false, x0, x2, x3, distance, s2, normal, 1.);
            check_point_edge_proximity(true, x2, x0, x1, distance, s0, normal, -1.);
        }else{
            s0=1;
            // check x0 against 2-3
            check_point_edge_proximity(false, x0, x2, x3, distance, s2, normal, 1.);
        }
    }else{
        if(s2<0){
            s2=0;
            // check x3 against 0-1
            check_point_edge_proximity(false, x3, x0, x1, distance, s0, normal, -1.);
        }else if(s2>1){
            s2=1;
            // check x2 against 0-1
            check_point_edge_proximity(false, x2, x0, x1, distance, s0, normal, -1.);
        }else{ // we already got the closest points!
            normal=(s0*x0+(1-s0)*x1)-(s2*x2+(1-s2)*x3);
            distance=mag(normal);
            if(distance>0) normal/=distance;
            else{
                normal=cross(x1-x0, x3-x2);
                normal/=mag(normal)+1e-300;
            }
        }
    }
}

void check_point_triangle_proximity(const Vec3d &x0, const Vec3d &x1, const Vec3d &x2, const Vec3d &x3,
                                    double &distance)
{
    // do it the QR way for added robustness
    Vec3d x13=x1-x3;
    double r00=mag(x13)+1e-30;
    x13/=r00;
    Vec3d x23=x2-x3;
    double r01=dot(x23,x13);
    x23-=r01*x13;
    double r11=mag(x23)+1e-30;
    x23/=r11;
    Vec3d x03=x0-x3;
    double s2=dot(x23,x03)/r11;
    double s1=(dot(x13,x03)-r01*s2)/r00;
    double s3=1-s1-s2;
    // check if we are in range
    if(s1>=0 && s2>=0 && s3>=0){
        distance=dist(x0, s1*x1+s2*x2+s3*x3);
    }else{
        if(s1>0){ // rules out edge 2-3
            check_point_edge_proximity(false, x0, x1, x2, distance);
            check_point_edge_proximity(true, x0, x1, x3, distance);
        }else if(s2>0){ // rules out edge 1-3
            check_point_edge_proximity(false, x0, x1, x2, distance);
            check_point_edge_proximity(true, x0, x2, x3, distance);
        }else{ // s3>0: rules out edge 1-2
            check_point_edge_proximity(false, x0, x2, x3, distance);
            check_point_edge_proximity(true, x0, x1, x3, distance);
        }
    }
}

// find distance between 0 and 1-2-3, with barycentric coordinates for closest point, and
// a normal that points from 1-2-3 towards 0 (unreliable if distance==0 or very small)
void check_point_triangle_proximity(const Vec3d &x0, const Vec3d &x1, const Vec3d &x2, const Vec3d &x3,
                                    double &distance, double &s1, double &s2, double &s3, Vec3d &normal)
{
    // do it the QR way for added robustness
    Vec3d x13=x1-x3;
    double r00=mag(x13)+1e-30;
    x13/=r00;
    Vec3d x23=x2-x3;
    double r01=dot(x23,x13);
    x23-=r01*x13;
    double r11=mag(x23)+1e-30;
    x23/=r11;
    Vec3d x03=x0-x3;
    s2=dot(x23,x03)/r11;
    s1=(dot(x13,x03)-r01*s2)/r00;
    s3=1-s1-s2;
    // check if we are in range
    if(s1>=0 && s2>=0 && s3>=0){
        normal=x0-(s1*x1+s2*x2+s3*x3);
        distance=mag(normal);
        if(distance>0) normal/=distance;
        else{
            normal=cross(x2-x1, x3-x1);
            normal/=mag(normal)+1e-300;
        }
    }else{
        double s, d;
        if(s1>0){ // rules out edge 2-3
            check_point_edge_proximity(false, x0, x1, x2, distance, s, normal, 1.);
            s1=s; s2=1-s; s3=0; d=distance;
            check_point_edge_proximity(true, x0, x1, x3, distance, s, normal, 1.);
            if(distance<d){
                s1=s; s2=0; s3=1-s;
            }
        }else if(s2>0){ // rules out edge 1-3
            check_point_edge_proximity(false, x0, x1, x2, distance, s, normal, 1.);
            s1=s; s2=1-s; s3=0; d=distance;
            check_point_edge_proximity(true, x0, x2, x3, distance, s, normal, 1.);
            if(distance<d){
                s1=0; s2=s; s3=1-s; d=distance;
            }
        }else{ // s3>0: rules out edge 1-2
            check_point_edge_proximity(false, x0, x2, x3, distance, s, normal, 1.);
            s1=0; s2=s; s3=1-s; d=distance;
            check_point_edge_proximity(true, x0, x1, x3, distance, s, normal, 1.);
            if(distance<d){
                s1=s; s2=0; s3=1-s;
            }
        }
    }    
    if(distance == 0)
    {
        //std::cout << "x0 = " << x0 << " x1 = " << x1 << " x2 = " << x2 << " x3 = " << x3 << " s = " << s1 << " " << s2 << " " << s3 << std::endl;
    }
}

double signed_volume(const Vec3d &x0, const Vec3d &x1, const Vec3d &x2, const Vec3d &x3)
{
    // Equivalent to triple(x1-x0, x2-x0, x3-x0), six times the signed volume of the tetrahedron.
    // But, for robustness, we want the result (up to sign) to be independent of the ordering.
    // And want it as accurate as possible...
    // But all that stuff is hard, so let's just use the common assumption that all coordinates are >0,
    // and do something reasonably accurate in fp.
    
    // This formula does almost four times too much multiplication, but if the coordinates are non-negative
    // it suffers in a minimal way from cancellation error.
    return ( x0[0]*(x1[1]*x3[2]+x3[1]*x2[2]+x2[1]*x1[2])
            +x1[0]*(x2[1]*x3[2]+x3[1]*x0[2]+x0[1]*x2[2])
            +x2[0]*(x3[1]*x1[2]+x1[1]*x0[2]+x0[1]*x3[2])
            +x3[0]*(x1[1]*x2[2]+x2[1]*x0[2]+x0[1]*x1[2]) )
    
    - ( x0[0]*(x2[1]*x3[2]+x3[1]*x1[2]+x1[1]*x2[2])
       +x1[0]*(x3[1]*x2[2]+x2[1]*x0[2]+x0[1]*x3[2])
       +x2[0]*(x1[1]*x3[2]+x3[1]*x0[2]+x0[1]*x1[2])
       +x3[0]*(x2[1]*x1[2]+x1[1]*x0[2]+x0[1]*x2[2]) );
}

}

