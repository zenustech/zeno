// Released into the public domain by Robert Bridson, 2009.

#include <cassert>
#include <cmath>
#include <iostream>
#include <tunicate.h>

//==============================================================================
static bool
same_sign(double a, double b)
{
    return (a<=0 && b<=0) || (a>=0 && b>=0);
}

//==============================================================================
int
simplex_intersection1d(int k,
                       const double* x0,
                       const double* x1,
                       const double* x2,
                       double* alpha0, 
                       double* alpha1, 
                       double* alpha2)
{
    assert(1<=k && k<=2);
    assert(alpha0 != NULL && alpha1 != NULL && alpha2!= NULL );
    
    if(alpha0 == NULL || alpha1 == NULL || alpha2 == NULL) //prevent null pointer warning
       return -1;
    
    if(k==1){
        if(x1[0]<x2[0]){
            if(x0[0]<x1[0]) return 0;
            else if(x0[0]>x2[0]) return 0;
            *alpha0=1;
            *alpha1=(x2[0]-x0[0])/(x2[0]-x1[0]);
            *alpha2=(x0[0]-x1[0])/(x2[0]-x1[0]);
            return 1;
        }else if(x1[0]>x2[0]){
            if(x0[0]<x2[0]) return 0;
            else if(x0[0]>x1[0]) return 0;
            *alpha0=1;
            *alpha1=(x2[0]-x0[0])/(x2[0]-x1[0]);
            *alpha2=(x0[0]-x1[0])/(x2[0]-x1[0]);
            return 1;
        }else{ // x1[0]==x2[0]
            if(x0[0]!=x1[0]) return 0;
            *alpha0=1;
            *alpha1=0.5;
            *alpha2=0.5;
            return 1;
        }
    }else
        return simplex_intersection1d(1, x2, x1, 0, alpha2, alpha1, alpha0);
}

//==============================================================================
// degenerate test in 2d - assumes three points lie on the same line
static int
simplex_intersection2d(int k,
                       const double* x0,
                       const double* x1,
                       const double* x2,
                       double* alpha0, 
                       double* alpha1, 
                       double* alpha2)
{
    assert(k==1);
    // try projecting each coordinate out in turn
    double ax0, ax1, ax2;
    if(!simplex_intersection1d(1, x0+1, x1+1, x2+1, &ax0, &ax1, &ax2)) return 0;
    double ay0, ay1, ay2;
    if(!simplex_intersection1d(1, x0, x1, x2, &ay0, &ay1, &ay2)) return 0;
    // decide which solution is more accurate for barycentric coordinates
    double checkx=std::fabs(-ax0*x0[0]+ax1*x1[0]+ax2*x2[0])
    +std::fabs(-ax0*x0[1]+ax1*x1[1]+ax2*x2[1]);
    double checky=std::fabs(-ay0*x0[0]+ay1*x1[0]+ay2*x2[0])
    +std::fabs(-ay0*x0[1]+ay1*x1[1]+ay2*x2[1]);
    if(checkx<=checky){
        *alpha0=ax0;
        *alpha1=ax1;
        *alpha2=ax2;
    }else{
        *alpha0=ay0;
        *alpha1=ay1;
        *alpha2=ay2;
    }
    return 1;
}

//==============================================================================
int
simplex_intersection2d(int k,
                       const double* x0,
                       const double* x1,
                       const double* x2,
                       const double* x3,
                       double* alpha0, 
                       double* alpha1, 
                       double* alpha2,
                       double* alpha3)
{
    assert(1<=k && k<=3);
    double sum1, sum2;
    switch(k){
        case 1: // point vs. triangle
            *alpha1=-orientation2d(x0, x2, x3);
            *alpha2= orientation2d(x0, x1, x3);
            if(!same_sign(*alpha1, *alpha2)) return 0;
            *alpha3=-orientation2d(x0, x1, x2);
            if(!same_sign(*alpha1, *alpha3)) return 0;
            if(!same_sign(*alpha2, *alpha3)) return 0;
            sum2=*alpha1+*alpha2+*alpha3;
            if(sum2){ // triangle not degenerate?
                *alpha0=1;
                *alpha1/=sum2;
                *alpha2/=sum2;
                *alpha3/=sum2;
                return 1;
            }else{ // triangle is degenerate and point lies on same line
                if(simplex_intersection2d(1, x0, x1, x2, alpha0, alpha1, alpha2)){
                    *alpha3=0;
                    return 1;
                }
                if(simplex_intersection2d(1, x0, x1, x3, alpha0, alpha1, alpha3)){
                    *alpha2=0;
                    return 1;
                }
                if(simplex_intersection2d(1, x0, x2, x3, alpha0, alpha2, alpha3)){
                    *alpha1=0;
                    return 1;
                }
                return 0;
            }
            
        case 2: // segment vs. segment
            *alpha0= orientation2d(x1, x2, x3);
            *alpha1=-orientation2d(x0, x2, x3);
            if(!same_sign(*alpha0, *alpha1)) return 0;
            *alpha2= orientation2d(x0, x1, x3);
            *alpha3=-orientation2d(x0, x1, x2);
            if(!same_sign(*alpha2, *alpha3)) return 0;
            sum1=*alpha0+*alpha1;
            sum2=*alpha2+*alpha3;
            if(sum1 && sum2){
                *alpha0/=sum1;
                *alpha1/=sum1;
                *alpha2/=sum2;
                *alpha3/=sum2;
                return 1;
            }else{ // degenerate: segments lie on the same line
                if(simplex_intersection2d(1, x0, x2, x3, alpha0, alpha2, alpha3)){
                    *alpha1=0;
                    return 1;
                }
                if(simplex_intersection2d(1, x1, x2, x3, alpha1, alpha2, alpha3)){
                    *alpha0=0;
                    return 1;
                }
                if(simplex_intersection2d(1, x2, x0, x1, alpha2, alpha0, alpha1)){
                    *alpha3=0;
                    return 1;
                }
                if(simplex_intersection2d(1, x3, x0, x1, alpha3, alpha0, alpha1)){
                    *alpha2=0;
                    return 1;
                }
                return 0;
            }
        case 3: // triangle vs. point
            return simplex_intersection2d(1, x3, x2, x1, x0,
                                          alpha3, alpha2, alpha1, alpha0);
        default:
            return -1; // should never get here
    }
}

//==============================================================================
// degenerate test in 3d - assumes four points lie on the same plane
static int
simplex_intersection3d(int k,
                       const double* x0,
                       const double* x1,
                       const double* x2,
                       const double* x3,
                       double* alpha0, 
                       double* alpha1, 
                       double* alpha2,
                       double* )
{
    assert(k<=2);
    // try projecting each coordinate out in turn
    double ax0, ax1, ax2, ax3;
    if(!simplex_intersection2d(k, x0+1, x1+1, x2+1, x3+1, &ax0, &ax1, &ax2,&ax3))
        return 0;
    double ay0, ay1, ay2, ay3;
    double p0[2]={x0[0], x0[2]}, p1[2]={x1[0], x1[2]},
    p2[2]={x2[0], x2[2]}, p3[2]={x3[0], x3[2]};
    if(!simplex_intersection2d(k, p0, p1, p2, p3, &ay0, &ay1, &ay2, &ay3))
        return 0;
    double az0, az1, az2, az3;
    if(!simplex_intersection2d(k, x0, x1, x2, x3, &az0, &az1, &az2, &az3))
        return 0;
    // decide which solution is more accurate for barycentric coordinates
    double checkx, checky, checkz;
    if(k==1){
        checkx=std::fabs(-ax0*x0[0]+ax1*x1[0]+ax2*x2[0]+ax3*x3[0])
        +std::fabs(-ax0*x0[1]+ax1*x1[1]+ax2*x2[1]+ax3*x3[1])
        +std::fabs(-ax0*x0[2]+ax1*x1[2]+ax2*x2[2]+ax3*x3[2]);
        checky=std::fabs(-ay0*x0[0]+ay1*x1[0]+ay2*x2[0]+ay3*x3[0])
        +std::fabs(-ay0*x0[1]+ay1*x1[1]+ay2*x2[1]+ay3*x3[1])
        +std::fabs(-ay0*x0[2]+ay1*x1[2]+ay2*x2[2]+ay3*x3[2]);
        checkz=std::fabs(-az0*x0[0]+az1*x1[0]+az2*x2[0]+az3*x3[0])
        +std::fabs(-az0*x0[1]+az1*x1[1]+az2*x2[1]+az3*x3[1])
        +std::fabs(-az0*x0[2]+az1*x1[2]+az2*x2[2]+az3*x3[2]);
    }else{
        checkx=std::fabs(-ax0*x0[0]-ax1*x1[0]+ax2*x2[0]+ax3*x3[0])
        +std::fabs(-ax0*x0[1]-ax1*x1[1]+ax2*x2[1]+ax3*x3[1])
        +std::fabs(-ax0*x0[2]-ax1*x1[2]+ax2*x2[2]+ax3*x3[2]);
        checky=std::fabs(-ay0*x0[0]-ay1*x1[0]+ay2*x2[0]+ay3*x3[0])
        +std::fabs(-ay0*x0[1]-ay1*x1[1]+ay2*x2[1]+ay3*x3[1])
        +std::fabs(-ay0*x0[2]-ay1*x1[2]+ay2*x2[2]+ay3*x3[2]);
        checkz=std::fabs(-az0*x0[0]-az1*x1[0]+az2*x2[0]+az3*x3[0])
        +std::fabs(-az0*x0[1]-az1*x1[1]+az2*x2[1]+az3*x3[1])
        +std::fabs(-az0*x0[2]-az1*x1[2]+az2*x2[2]+az3*x3[2]);
    }
    if(checkx<=checky && checkx<=checkz){
        *alpha0=ax0;
        *alpha1=ax1;
        *alpha2=ax2;
        *alpha2=ax3;
    }else if(checky<=checkz){
        *alpha0=ay0;
        *alpha1=ay1;
        *alpha2=ay2;
        *alpha2=ay3;
    }else{
        *alpha0=az0;
        *alpha1=az1;
        *alpha2=az2;
        *alpha2=az3;
    }
    return 1;
}

//==============================================================================
int
simplex_intersection3d(int k,
                       const double* x0,
                       const double* x1,
                       const double* x2,
                       const double* x3,
                       const double* x4,
                       double* alpha0, 
                       double* alpha1, 
                       double* alpha2,
                       double* alpha3,
                       double* alpha4)
{
    assert(1<=k && k<=4);
    double sum1, sum2;
    switch(k){
        case 1: // point vs. tetrahedron
            *alpha1=-orientation3d(x0, x2, x3, x4);
            *alpha2= orientation3d(x0, x1, x3, x4);
            if(!same_sign(*alpha1, *alpha2)) return 0;
            *alpha3=-orientation3d(x0, x1, x2, x4);
            if(!same_sign(*alpha1, *alpha3)) return 0;
            if(!same_sign(*alpha2, *alpha3)) return 0;
            *alpha4= orientation3d(x0, x1, x2, x3);
            if(!same_sign(*alpha1, *alpha4)) return 0;
            if(!same_sign(*alpha2, *alpha4)) return 0;         
            if(!same_sign(*alpha3, *alpha4)) return 0;                  
            *alpha0=1;
            sum2=*alpha1+*alpha2+*alpha3+*alpha4;
            if(sum2){
                *alpha1/=sum2;
                *alpha2/=sum2;
                *alpha3/=sum2;
                *alpha4/=sum2;
                return 1;
            }else{ // degenerate: point and tetrahedron in same plane
                if(simplex_intersection3d(1, x0, x2, x3, x4,
                                          alpha0, alpha2, alpha3, alpha4)){
                    *alpha1=0;
                    return 1;
                }
                if(simplex_intersection3d(1, x0, x1, x3, x4,
                                          alpha0, alpha1, alpha3, alpha4)){
                    *alpha2=0;
                    return 1;
                }
                if(simplex_intersection3d(1, x0, x1, x2, x4,
                                          alpha0, alpha1, alpha2, alpha4)){
                    *alpha3=0;
                    return 1;
                }
                if(simplex_intersection3d(1, x0, x1, x2, x3,
                                          alpha0, alpha1, alpha2, alpha3)){
                    *alpha4=0;
                    return 1;
                }
                return 0;
            }
            
        case 2: // segment vs. triangle
            *alpha0= orientation3d(x1, x2, x3, x4);
            *alpha1=-orientation3d(x0, x2, x3, x4);
            if(!same_sign(*alpha0, *alpha1)) return 0;
            *alpha2= orientation3d(x0, x1, x3, x4);
            *alpha3=-orientation3d(x0, x1, x2, x4);
            if(!same_sign(*alpha2, *alpha3)) return 0;
            *alpha4= orientation3d(x0, x1, x2, x3);
            if(!same_sign(*alpha2, *alpha4)) return 0;
            if(!same_sign(*alpha3, *alpha4)) return 0;
            sum1=*alpha0+*alpha1;
            sum2=*alpha2+*alpha3+*alpha4;
            
            if(sum1 && sum2){
                *alpha0/=sum1;
                *alpha1/=sum1;
                *alpha2/=sum2;
                *alpha3/=sum2;
                *alpha4/=sum2;
                return 1;
            }else{ // degenerate: segment and triangle in same plane
                if(simplex_intersection3d(1, x1, x2, x3, x4,
                                          alpha1, alpha2, alpha3, alpha4)){
                    *alpha0=0;
                    return 1;
                }
                if(simplex_intersection3d(1, x0, x2, x3, x4,
                                          alpha0, alpha2, alpha3, alpha4)){
                    *alpha1=0;
                    return 1;
                }
                if(simplex_intersection3d(2, x0, x1, x3, x4,
                                          alpha0, alpha1, alpha3, alpha4)){
                    *alpha2=0;
                    return 1;
                }
                if(simplex_intersection3d(2, x0, x1, x2, x4,
                                          alpha0, alpha1, alpha2, alpha4)){
                    *alpha3=0;
                    return 1;
                }
                if(simplex_intersection3d(2, x0, x1, x2, x3,
                                          alpha0, alpha1, alpha2, alpha3)){
                    *alpha4=0;
                    return 1;
                }
                return 0;
            }
            
        case 3: // triangle vs. segment
        case 4: // tetrahedron vs. point
            return simplex_intersection3d(5-k, x4, x3, x2, x1, x0,
                                          alpha4, alpha3, alpha2, alpha1, alpha0);
        default:
            return -1; // should never get here
    }
}

//==============================================================================
// degenerate test in 3d+time - assumes five points lie on the same hyper-plane
static int
simplex_intersection_time3d(int k,
                            const double* x0, int time0,
                            const double* x1, int time1,
                            const double* x2, int time2,
                            const double* x3, int time3,
                            const double* x4, int time4,
                            double* alpha0, 
                            double* alpha1, 
                            double* alpha2,
                            double* alpha3,
                            double* )
{
    assert(k<=2);
    assert(time0==0 || time0==1);
    assert(time1==0 || time1==1);
    assert(time2==0 || time2==1);
    assert(time3==0 || time3==1);
    assert(time4==0 || time4==1);
    // try projecting each coordinate out in turn
    double ax0, ax1, ax2, ax3, ax4;
    double r0[3]={x0[0], x0[2], (double)time0}, r1[3]={x1[0], x1[2], (double)time1},
    r2[3]={x2[0], x2[2], (double)time2}, r3[3]={x3[0], x3[2], (double)time3},
    r4[3]={x4[0], x4[2], (double)time4};
    if(!simplex_intersection3d(k, r0, r1, r2, r3, r4,
                               &ax0, &ax1, &ax2, &ax3, &ax4)) return 0;
    double ay0, ay1, ay2, ay3, ay4;
    double p0[3]={x0[0], x0[2], (double)time0}, p1[3]={x1[0], x1[2], (double)time1},
    p2[3]={x2[0], x2[2], (double)time2}, p3[3]={x3[0], x3[2], (double)time3},
    p4[3]={x4[0], x4[2], (double)time4};
    if(!simplex_intersection3d(k, p0, p1, p2, p3, p4,
                               &ay0, &ay1, &ay2, &ay3, &ay4)) return 0;
    double az0, az1, az2, az3, az4;
    double q0[3]={x0[0], x0[1], (double)time0}, q1[3]={x1[0], x1[1], (double)time1},
    q2[3]={x2[0], x2[1], (double)time2}, q3[3]={x3[0], x3[1], (double)time3},
    q4[3]={x4[0], x4[1], (double)time4};
    if(!simplex_intersection3d(k, q0, q1, q2, q3, q4,
                               &az0, &az1, &az2, &az3, &az4)) return 0;
    double at0, at1, at2, at3, at4;
    if(!simplex_intersection3d(k, x0, x1, x2, x3, x4,
                               &at0, &at1, &at2, &at3, &at4)) return 0;
    // decide which solution is more accurate for barycentric coordinates
    double checkx, checky, checkz, checkt;
    if(k==1){
        checkx=std::fabs(-ax0*x0[0]+ax1*x1[0]+ax2*x2[0]+ax3*x3[0]+ax4*x4[0])
        +std::fabs(-ax0*x0[1]+ax1*x1[1]+ax2*x2[1]+ax3*x3[1]+ax4*x4[1])
        +std::fabs(-ax0*x0[2]+ax1*x1[2]+ax2*x2[2]+ax3*x3[2]+ax4*x4[2])
        +std::fabs(-ax0*time0+ax1*time1+ax2*time2+ax3*time3+ax4*time4);
        checky=std::fabs(-ay0*x0[0]+ay1*x1[0]+ay2*x2[0]+ay3*x3[0]+ay4*x4[0])
        +std::fabs(-ay0*x0[1]+ay1*x1[1]+ay2*x2[1]+ay3*x3[1]+ay4*x4[1])
        +std::fabs(-ay0*x0[2]+ay1*x1[2]+ay2*x2[2]+ay3*x3[2]+ay4*x4[2])
        +std::fabs(-ay0*time0+ay1*time1+ay2*time2+ay3*time3+ay4*time4);
        checkz=std::fabs(-az0*x0[0]+az1*x1[0]+az2*x2[0]+az3*x3[0]+az4*x4[0])
        +std::fabs(-az0*x0[1]+az1*x1[1]+az2*x2[1]+az3*x3[1]+az4*x4[1])
        +std::fabs(-az0*x0[2]+az1*x1[2]+az2*x2[2]+az3*x3[2]+az4*x4[2])
        +std::fabs(-az0*time0+az1*time1+az2*time2+az3*time3+az4*time4);
        checkt=std::fabs(-at0*x0[0]+at1*x1[0]+at2*x2[0]+at3*x3[0]+at4*x4[0])
        +std::fabs(-at0*x0[1]+at1*x1[1]+at2*x2[1]+at3*x3[1]+at4*x4[1])
        +std::fabs(-at0*x0[2]+at1*x1[2]+at2*x2[2]+at3*x3[2]+at4*x4[2])
        +std::fabs(-at0*time0+at1*time1+at2*time2+at3*time3+at4*time4);
    }else{
        checkx=std::fabs(-ax0*x0[0]-ax1*x1[0]+ax2*x2[0]+ax3*x3[0]+ax4*x4[0])
        +std::fabs(-ax0*x0[1]-ax1*x1[1]+ax2*x2[1]+ax3*x3[1]+ax4*x4[1])
        +std::fabs(-ax0*x0[2]-ax1*x1[2]+ax2*x2[2]+ax3*x3[2]+ax4*x4[2])
        +std::fabs(-ax0*time0-ax1*time1+ax2*time2+ax3*time3+ax4*time4);
        checky=std::fabs(-ay0*x0[0]-ay1*x1[0]+ay2*x2[0]+ay3*x3[0]+ay4*x4[0])
        +std::fabs(-ay0*x0[1]-ay1*x1[1]+ay2*x2[1]+ay3*x3[1]+ay4*x4[1])
        +std::fabs(-ay0*x0[2]-ay1*x1[2]+ay2*x2[2]+ay3*x3[2]+ay4*x4[2])
        +std::fabs(-ay0*time0-ay1*time1+ay2*time2+ay3*time3+ay4*time4);
        checkz=std::fabs(-az0*x0[0]-az1*x1[0]+az2*x2[0]+az3*x3[0]+az4*x4[0])
        +std::fabs(-az0*x0[1]-az1*x1[1]+az2*x2[1]+az3*x3[1]+az4*x4[1])
        +std::fabs(-az0*x0[2]-az1*x1[2]+az2*x2[2]+az3*x3[2]+az4*x4[2])
        +std::fabs(-az0*time0-az1*time1+az2*time2+az3*time3+az4*time4);
        checkt=std::fabs(-at0*x0[0]-at1*x1[0]+at2*x2[0]+at3*x3[0]+at4*x4[0])
        +std::fabs(-at0*x0[1]-at1*x1[1]+at2*x2[1]+at3*x3[1]+at4*x4[1])
        +std::fabs(-at0*x0[2]-at1*x1[2]+at2*x2[2]+at3*x3[2]+at4*x4[2])
        +std::fabs(-at0*time0-at1*time1+at2*time2+at3*time3+at4*time4);
    }
    if(checkx<=checky && checkx<=checkz && checkx<=checkt){
        *alpha0=ax0;
        *alpha1=ax1;
        *alpha2=ax2;
        *alpha3=ax3;
    }else if(checky<=checkz && checky<=checkt){
        *alpha0=ay0;
        *alpha1=ay1;
        *alpha2=ay2;
        *alpha3=ay3;
    }else if(checkz<=checkt){
        *alpha0=az0;
        *alpha1=az1;
        *alpha2=az2;
        *alpha3=az3;
    }else{
        *alpha0=at0;
        *alpha1=at1;
        *alpha2=at2;
        *alpha3=at3;
    }
    return 1;
}

//==============================================================================
int
simplex_intersection_time3d(int k,
                            const double* x0, int t0,
                            const double* x1, int t1,
                            const double* x2, int t2,
                            const double* x3, int t3,
                            const double* x4, int t4,
                            const double* x5, int t5,
                            double* alpha0, 
                            double* alpha1, 
                            double* alpha2,
                            double* alpha3,
                            double* alpha4,
                            double* alpha5)
{
    assert(1<=k && k<=5);
    assert(t0==0 || t0==1);
    assert(t1==0 || t1==1);
    assert(t2==0 || t2==1);
    assert(t3==0 || t3==1);
    assert(t4==0 || t4==1);
    assert(t5==0 || t5==1);
    double sum1, sum2;
    switch(k){
        case 1: // point vs. pentachoron
            *alpha1=-orientation_time3d(x0, t0, x2, t2, x3, t3, x4, t4, x5, t5);
            *alpha2= orientation_time3d(x0, t0, x1, t1, x3, t3, x4, t4, x5, t5);
            if(!same_sign(*alpha1, *alpha2)) return 0;
            *alpha3=-orientation_time3d(x0, t0, x1, t1, x2, t3, x4, t4, x5, t5);
            if(!same_sign(*alpha1, *alpha3)) return 0;
            *alpha4= orientation_time3d(x0, t0, x1, t1, x2, t2, x3, t3, x5, t5);
            if(!same_sign(*alpha1, *alpha4)) return 0;
            *alpha5=-orientation_time3d(x0, t0, x1, t1, x2, t2, x3, t3, x4, t4);
            if(!same_sign(*alpha1, *alpha5)) return 0;
            sum2=*alpha1+*alpha2+*alpha3+*alpha4+*alpha5;
            if(sum2){
                *alpha0=1;
                *alpha1/=sum2;
                *alpha2/=sum2;
                *alpha3/=sum2;
                *alpha4/=sum2;
                *alpha5/=sum2;
                return 1;
            }else{
                if(simplex_intersection_time3d(1, x0, t0, x2, t2, x3, t3, x4, t4,
                                               x5, t5, alpha0, alpha2, alpha3, alpha4, alpha5)){
                    *alpha1=0;
                    return 1;
                }
                if(simplex_intersection_time3d(1, x0, t0, x1, t1, x3, t3, x4, t4,
                                               x5, t5, alpha0, alpha1, alpha3, alpha4, alpha5)){
                    *alpha2=0;
                    return 1;
                }
                if(simplex_intersection_time3d(1, x0, t0, x1, t1, x2, t2, x4, t4,
                                               x5, t5, alpha0, alpha1, alpha2, alpha4, alpha5)){
                    *alpha3=0;
                    return 1;
                }
                if(simplex_intersection_time3d(1, x0, t0, x1, t1, x2, t2, x3, t3,
                                               x5, t5, alpha0, alpha1, alpha2, alpha3, alpha5)){
                    *alpha4=0;
                    return 1;
                }
                if(simplex_intersection_time3d(1, x0, t0, x1, t1, x2, t2, x3, t3,
                                               x4, t4, alpha0, alpha1, alpha2, alpha3, alpha4)){
                    *alpha5=0;
                    return 1;
                }
                return 0;
            }
            
        case 2: // segment vs. tetrahedron
            *alpha0= orientation_time3d(x1, t1, x2, t2, x3, t3, x4, t4, x5, t5);
            *alpha1=-orientation_time3d(x0, t0, x2, t2, x3, t3, x4, t4, x5, t5);
            if(!same_sign(*alpha0, *alpha1)) return 0;
            *alpha2= orientation_time3d(x0, t0, x1, t1, x3, t3, x4, t4, x5, t5);
            *alpha3=-orientation_time3d(x0, t0, x1, t1, x2, t2, x4, t4, x5, t5);
            if(!same_sign(*alpha2, *alpha3)) return 0;
            *alpha4= orientation_time3d(x0, t0, x1, t1, x2, t2, x3, t3, x5, t5);
            if(!same_sign(*alpha2, *alpha4)) return 0;
            *alpha5=-orientation_time3d(x0, t0, x1, t1, x2, t2, x3, t3, x4, t4);
            if(!same_sign(*alpha2, *alpha5)) return 0;
            sum1=*alpha0+*alpha1;
            sum2=*alpha2+*alpha3+*alpha4+*alpha5;
            if(sum1 && sum2){
                *alpha0/=sum1;
                *alpha1/=sum1;
                *alpha2/=sum2;
                *alpha3/=sum2;
                *alpha4/=sum2;
                *alpha5/=sum2;
                return 1;
            }else{
                if(simplex_intersection_time3d(1, x1, t1, x2, t2, x3, t3, x4, t4,
                                               x5, t5, alpha1, alpha2, alpha3, alpha4, alpha5)){
                    *alpha0=0;
                    return 1;
                }
                if(simplex_intersection_time3d(1, x0, t0, x2, t2, x3, t3, x4, t4,
                                               x5, t5, alpha0, alpha2, alpha3, alpha4, alpha5)){
                    *alpha1=0;
                    return 1;
                }
                if(simplex_intersection_time3d(2, x0, t0, x1, t1, x3, t3, x4, t4,
                                               x5, t5, alpha0, alpha1, alpha3, alpha4, alpha5)){
                    *alpha2=0;
                    return 1;
                }
                if(simplex_intersection_time3d(2, x0, t0, x1, t1, x2, t2, x4, t4,
                                               x5, t5, alpha0, alpha1, alpha2, alpha4, alpha5)){
                    *alpha3=0;
                    return 1;
                }
                if(simplex_intersection_time3d(2, x0, t0, x1, t1, x2, t2, x3, t3,
                                               x5, t5, alpha0, alpha1, alpha2, alpha3, alpha5)){
                    *alpha4=0;
                    return 1;
                }
                if(simplex_intersection_time3d(2, x0, t0, x1, t1, x2, t2, x3, t3,
                                               x4, t4, alpha0, alpha1, alpha2, alpha3, alpha4)){
                    *alpha5=0;
                    return 1;
                }
                return 0;
            }
            
        case 3: // triangle vs. triangle
            *alpha0= orientation_time3d(x1, t1, x2, t2, x3, t3, x4, t4, x5, t5);
            *alpha1=-orientation_time3d(x0, t0, x2, t2, x3, t3, x4, t4, x5, t5);
            if(!same_sign(*alpha0, *alpha1)) return 0;
            *alpha2= orientation_time3d(x0, t0, x1, t1, x3, t3, x4, t4, x5, t5);
            if(!same_sign(*alpha0, *alpha2)) return 0;
            *alpha3=-orientation_time3d(x0, t0, x1, t1, x2, t2, x4, t4, x5, t5);
            *alpha4= orientation_time3d(x0, t0, x1, t1, x2, t2, x3, t3, x5, t5);
            if(!same_sign(*alpha3, *alpha4)) return 0;
            *alpha5=-orientation_time3d(x0, t0, x1, t1, x2, t2, x3, t3, x4, t4);
            if(!same_sign(*alpha3, *alpha5)) return 0;
            sum1=*alpha0+*alpha1+*alpha2;
            sum2=*alpha3+*alpha4+*alpha5;
            if(sum1 && sum2){
                *alpha0/=sum1;
                *alpha1/=sum1;
                *alpha2/=sum1;
                *alpha3/=sum2;
                *alpha4/=sum2;
                *alpha5/=sum2;
                return 1;
            }else{
                if(simplex_intersection_time3d(2, x1, t1, x2, t2, x3, t3, x4, t4,
                                               x5, t5, alpha1, alpha2, alpha3, alpha4, alpha5)){
                    *alpha0=0;
                    return 1;
                }
                if(simplex_intersection_time3d(2, x0, t0, x2, t2, x3, t3, x4, t4,
                                               x5, t5, alpha0, alpha2, alpha3, alpha4, alpha5)){
                    *alpha1=0;
                    return 1;
                }
                if(simplex_intersection_time3d(2, x0, t0, x1, t1, x3, t3, x4, t4,
                                               x5, t5, alpha0, alpha1, alpha3, alpha4, alpha5)){
                    *alpha2=0;
                    return 1;
                }
                if(simplex_intersection_time3d(2, x4, t4, x5, t5, x0, t0, x1, t1,
                                               x2, t2, alpha4, alpha5, alpha0, alpha1, alpha2)){
                    *alpha3=0;
                    return 1;
                }
                if(simplex_intersection_time3d(2, x3, t3, x5, t5, x0, t0, x1, t1,
                                               x2, t2, alpha3, alpha5, alpha0, alpha1, alpha2)){
                    *alpha4=0;
                    return 1;
                }
                if(simplex_intersection_time3d(2, x3, t3, x4, t4, x0, t0, x1, t1,
                                               x2, t2, alpha3, alpha4, alpha0, alpha1, alpha2)){
                    *alpha5=0;
                    return 1;
                }
                return 0;
            }
            
        case 4: // tetrahedron vs. segment
        case 5: // pentachoron vs. point
            return simplex_intersection_time3d(6-k, x5, t5, x4, t4, x3, t3, x2, t2,
                                               x1, t1, x0, t0, alpha5, alpha4, alpha3, alpha2, alpha1, alpha0);
        default:
            return -1; // should never get here
    }
}

//==============================================================================
// degenerate test in 4d - assumes five points lie on the same hyper-plane
static int
simplex_intersection4d(int k,
                       const double* x0,
                       const double* x1,
                       const double* x2,
                       const double* x3,
                       const double* x4,
                       double* alpha0, 
                       double* alpha1, 
                       double* alpha2,
                       double* alpha3,
                       double* )
{
    assert(k<=2);
    // try projecting each coordinate out in turn
    double ax0, ax1, ax2, ax3, ax4;
    if(!simplex_intersection3d(k, x0+1, x1+1, x2+1, x3+1, x4+1,
                               &ax0, &ax1, &ax2, &ax3, &ax4)) return 0;
    double ay0, ay1, ay2, ay3, ay4;
    double p0[3]={x0[0], x0[2], x0[3]}, p1[3]={x1[0], x1[2], x1[3]},
    p2[3]={x2[0], x2[2], x2[3]}, p3[3]={x3[0], x3[2], x3[3]},
    p4[3]={x4[0], x4[2], x4[3]};
    if(!simplex_intersection3d(k, p0, p1, p2, p3, p4,
                               &ay0, &ay1, &ay2, &ay3, &ay4)) return 0;
    double az0, az1, az2, az3, az4;
    double q0[3]={x0[0], x0[1], x0[3]}, q1[3]={x1[0], x1[1], x1[3]},
    q2[3]={x2[0], x2[1], x2[3]}, q3[3]={x3[0], x3[1], x3[3]},
    q4[3]={x4[0], x4[1], x4[3]};
    if(!simplex_intersection3d(k, q0, q1, q2, q3, q4,
                               &az0, &az1, &az2, &az3, &az4)) return 0;
    double at0, at1, at2, at3, at4;
    if(!simplex_intersection3d(k, x0, x1, x2, x3, x4,
                               &at0, &at1, &at2, &at3, &at4)) return 0;
    // decide which solution is more accurate for barycentric coordinates
    double checkx, checky, checkz, checkt;
    if(k==1){
        checkx=std::fabs(-ax0*x0[0]+ax1*x1[0]+ax2*x2[0]+ax3*x3[0]+ax4*x4[0])
        +std::fabs(-ax0*x0[1]+ax1*x1[1]+ax2*x2[1]+ax3*x3[1]+ax4*x4[1])
        +std::fabs(-ax0*x0[2]+ax1*x1[2]+ax2*x2[2]+ax3*x3[2]+ax4*x4[2])
        +std::fabs(-ax0*x0[3]+ax1*x1[3]+ax2*x2[3]+ax3*x3[3]+ax4*x4[3]);
        checky=std::fabs(-ay0*x0[0]+ay1*x1[0]+ay2*x2[0]+ay3*x3[0]+ay4*x4[0])
        +std::fabs(-ay0*x0[1]+ay1*x1[1]+ay2*x2[1]+ay3*x3[1]+ay4*x4[1])
        +std::fabs(-ay0*x0[2]+ay1*x1[2]+ay2*x2[2]+ay3*x3[2]+ay4*x4[2])
        +std::fabs(-ay0*x0[3]+ay1*x1[3]+ay2*x2[3]+ay3*x3[3]+ay4*x4[3]);
        checkz=std::fabs(-az0*x0[0]+az1*x1[0]+az2*x2[0]+az3*x3[0]+az4*x4[0])
        +std::fabs(-az0*x0[1]+az1*x1[1]+az2*x2[1]+az3*x3[1]+az4*x4[1])
        +std::fabs(-az0*x0[2]+az1*x1[2]+az2*x2[2]+az3*x3[2]+az4*x4[2])
        +std::fabs(-az0*x0[3]+az1*x1[3]+az2*x2[3]+az3*x3[3]+az4*x4[3]);
        checkt=std::fabs(-at0*x0[0]+at1*x1[0]+at2*x2[0]+at3*x3[0]+at4*x4[0])
        +std::fabs(-at0*x0[1]+at1*x1[1]+at2*x2[1]+at3*x3[1]+at4*x4[1])
        +std::fabs(-at0*x0[2]+at1*x1[2]+at2*x2[2]+at3*x3[2]+at4*x4[2])
        +std::fabs(-at0*x0[3]+at1*x1[3]+at2*x2[3]+at3*x3[3]+at4*x4[3]);
    }else{
        checkx=std::fabs(-ax0*x0[0]-ax1*x1[0]+ax2*x2[0]+ax3*x3[0]+ax4*x4[0])
        +std::fabs(-ax0*x0[1]-ax1*x1[1]+ax2*x2[1]+ax3*x3[1]+ax4*x4[1])
        +std::fabs(-ax0*x0[2]-ax1*x1[2]+ax2*x2[2]+ax3*x3[2]+ax4*x4[2])
        +std::fabs(-ax0*x0[3]-ax1*x1[3]+ax2*x2[3]+ax3*x3[3]+ax4*x4[3]);
        checky=std::fabs(-ay0*x0[0]-ay1*x1[0]+ay2*x2[0]+ay3*x3[0]+ay4*x4[0])
        +std::fabs(-ay0*x0[1]-ay1*x1[1]+ay2*x2[1]+ay3*x3[1]+ay4*x4[1])
        +std::fabs(-ay0*x0[2]-ay1*x1[2]+ay2*x2[2]+ay3*x3[2]+ay4*x4[2])
        +std::fabs(-ay0*x0[3]-ay1*x1[3]+ay2*x2[3]+ay3*x3[3]+ay4*x4[3]);
        checkz=std::fabs(-az0*x0[0]-az1*x1[0]+az2*x2[0]+az3*x3[0]+az4*x4[0])
        +std::fabs(-az0*x0[1]-az1*x1[1]+az2*x2[1]+az3*x3[1]+az4*x4[1])
        +std::fabs(-az0*x0[2]-az1*x1[2]+az2*x2[2]+az3*x3[2]+az4*x4[2])
        +std::fabs(-az0*x0[3]-az1*x1[3]+az2*x2[3]+az3*x3[3]+az4*x4[3]);
        checkt=std::fabs(-at0*x0[0]-at1*x1[0]+at2*x2[0]+at3*x3[0]+at4*x4[0])
        +std::fabs(-at0*x0[1]-at1*x1[1]+at2*x2[1]+at3*x3[1]+at4*x4[1])
        +std::fabs(-at0*x0[2]-at1*x1[2]+at2*x2[2]+at3*x3[2]+at4*x4[2])
        +std::fabs(-at0*x0[3]-at1*x1[3]+at2*x2[3]+at3*x3[3]+at4*x4[3]);
    }
    if(checkx<=checky && checkx<=checkz && checkx<=checkt){
        *alpha0=ax0;
        *alpha1=ax1;
        *alpha2=ax2;
        *alpha3=ax3;
    }else if(checky<=checkz && checky<=checkt){
        *alpha0=ay0;
        *alpha1=ay1;
        *alpha2=ay2;
        *alpha3=ay3;
    }else if(checkz<=checkt){
        *alpha0=az0;
        *alpha1=az1;
        *alpha2=az2;
        *alpha3=az3;
    }else{
        *alpha0=at0;
        *alpha1=at1;
        *alpha2=at2;
        *alpha3=at3;
    }
    return 1;
}

//==============================================================================
int
simplex_intersection4d(int k,
                       const double* x0,
                       const double* x1,
                       const double* x2,
                       const double* x3,
                       const double* x4,
                       const double* x5,
                       double* alpha0, 
                       double* alpha1, 
                       double* alpha2,
                       double* alpha3,
                       double* alpha4,
                       double* alpha5)
{
    assert(1<=k && k<=5);
    double sum1, sum2;
    switch(k){
        case 1: // point vs. pentachoron
            *alpha1=-orientation4d(x0, x2, x3, x4, x5);
            *alpha2= orientation4d(x0, x1, x3, x4, x5);
            if(!same_sign(*alpha1, *alpha2)) return 0;
            *alpha3=-orientation4d(x0, x1, x2, x4, x5);
            if(!same_sign(*alpha1, *alpha3)) return 0;
            if(!same_sign(*alpha2, *alpha3)) return 0;
            *alpha4= orientation4d(x0, x1, x2, x3, x5);
            if(!same_sign(*alpha1, *alpha4)) return 0;
            if(!same_sign(*alpha2, *alpha4)) return 0;
            if(!same_sign(*alpha3, *alpha4)) return 0;
            *alpha5=-orientation4d(x0, x1, x2, x3, x4);
            if(!same_sign(*alpha1, *alpha5)) return 0;
            if(!same_sign(*alpha2, *alpha5)) return 0;
            if(!same_sign(*alpha3, *alpha5)) return 0;
            if(!same_sign(*alpha4, *alpha5)) return 0;
            
            
            sum2=*alpha1+*alpha2+*alpha3+*alpha4+*alpha5;
            if(sum2){
                *alpha0=1;
                *alpha1/=sum2;
                *alpha2/=sum2;
                *alpha3/=sum2;
                *alpha4/=sum2;
                *alpha5/=sum2;
                return 1;
            }else{
                if(simplex_intersection4d(1, x0, x2, x3, x4, x5,
                                          alpha0, alpha2, alpha3, alpha4, alpha5)){
                    *alpha1=0;
                    return 1;
                }
                if(simplex_intersection4d(1, x0, x1, x3, x4, x5,
                                          alpha0, alpha1, alpha3, alpha4, alpha5)){
                    *alpha2=0;
                    return 1;
                }
                if(simplex_intersection4d(1, x0, x1, x2, x4, x5,
                                          alpha0, alpha1, alpha2, alpha4, alpha5)){
                    *alpha3=0;
                    return 1;
                }
                if(simplex_intersection4d(1, x0, x1, x2, x3, x5,
                                          alpha0, alpha1, alpha2, alpha3, alpha5)){
                    *alpha4=0;
                    return 1;
                }
                if(simplex_intersection4d(1, x0, x1, x2, x3, x4,
                                          alpha0, alpha1, alpha2, alpha3, alpha4)){
                    *alpha5=0;
                    return 1;
                }
                return 0;
            }
            
        case 2: // segment vs. tetrahedron
            *alpha0= orientation4d(x1, x2, x3, x4, x5);
            *alpha1=-orientation4d(x0, x2, x3, x4, x5);
            if(!same_sign(*alpha0, *alpha1)) return 0;
            *alpha2= orientation4d(x0, x1, x3, x4, x5);
            *alpha3=-orientation4d(x0, x1, x2, x4, x5);
            if(!same_sign(*alpha2, *alpha3)) return 0;
            *alpha4= orientation4d(x0, x1, x2, x3, x5);
            if(!same_sign(*alpha2, *alpha4)) return 0;
            if(!same_sign(*alpha3, *alpha4)) return 0;         
            *alpha5=-orientation4d(x0, x1, x2, x3, x4);
            if(!same_sign(*alpha2, *alpha5)) return 0;
            if(!same_sign(*alpha3, *alpha5)) return 0;
            if(!same_sign(*alpha4, *alpha5)) return 0;         
            
            sum1=*alpha0+*alpha1;
            sum2=*alpha2+*alpha3+*alpha4+*alpha5;
            if(sum1 && sum2){
                *alpha0/=sum1;
                *alpha1/=sum1;
                *alpha2/=sum2;
                *alpha3/=sum2;
                *alpha4/=sum2;
                *alpha5/=sum2;
                return 1;
            }else{
                if(simplex_intersection4d(1, x1, x2, x3, x4, x5,
                                          alpha1, alpha2, alpha3, alpha4, alpha5)){
                    *alpha0=0;
                    return 1;
                }
                if(simplex_intersection4d(1, x0, x2, x3, x4, x5,
                                          alpha0, alpha2, alpha3, alpha4, alpha5)){
                    *alpha1=0;
                    return 1;
                }
                if(simplex_intersection4d(2, x0, x1, x3, x4, x5,
                                          alpha0, alpha1, alpha3, alpha4, alpha5)){
                    *alpha2=0;
                    return 1;
                }
                if(simplex_intersection4d(2, x0, x1, x2, x4, x5,
                                          alpha0, alpha1, alpha2, alpha4, alpha5)){
                    *alpha3=0;
                    return 1;
                }
                if(simplex_intersection4d(2, x0, x1, x2, x3, x5,
                                          alpha0, alpha1, alpha2, alpha3, alpha5)){
                    *alpha4=0;
                    return 1;
                }
                if(simplex_intersection4d(2, x0, x1, x2, x3, x4,
                                          alpha0, alpha1, alpha2, alpha3, alpha4)){
                    *alpha5=0;
                    return 1;
                }
                return 0;
            }
            
        case 3: // triangle vs. triangle
            *alpha0= orientation4d(x1, x2, x3, x4, x5);
            *alpha1=-orientation4d(x0, x2, x3, x4, x5);
            if(!same_sign(*alpha0, *alpha1)) return 0;
            *alpha2= orientation4d(x0, x1, x3, x4, x5);
            if(!same_sign(*alpha0, *alpha2)) return 0;
            if(!same_sign(*alpha1, *alpha2)) return 0;
            *alpha3=-orientation4d(x0, x1, x2, x4, x5);
            *alpha4= orientation4d(x0, x1, x2, x3, x5);
            if(!same_sign(*alpha3, *alpha4)) return 0;
            *alpha5=-orientation4d(x0, x1, x2, x3, x4);
            if(!same_sign(*alpha3, *alpha5)) return 0;
            if(!same_sign(*alpha4, *alpha5)) return 0;         
            
            sum1=*alpha0+*alpha1+*alpha2;
            sum2=*alpha3+*alpha4+*alpha5;
            if(sum1 && sum2){
                *alpha0/=sum1;
                *alpha1/=sum1;
                *alpha2/=sum1;
                *alpha3/=sum2;
                *alpha4/=sum2;
                *alpha5/=sum2;
                return 1;
            }else{
                if(simplex_intersection4d(2, x1, x2, x3, x4, x5,
                                          alpha1, alpha2, alpha3, alpha4, alpha5)){
                    *alpha0=0;
                    return 1;
                }
                if(simplex_intersection4d(2, x0, x2, x3, x4, x5,
                                          alpha0, alpha2, alpha3, alpha4, alpha5)){
                    *alpha1=0;
                    return 1;
                }
                if(simplex_intersection4d(2, x0, x1, x3, x4, x5,
                                          alpha0, alpha1, alpha3, alpha4, alpha5)){
                    *alpha2=0;
                    return 1;
                }
                if(simplex_intersection4d(2, x4, x5, x0, x1, x2,
                                          alpha4, alpha5, alpha0, alpha1, alpha2)){
                    *alpha3=0;
                    return 1;
                }
                if(simplex_intersection4d(2, x3, x5, x0, x1, x2,
                                          alpha3, alpha5, alpha0, alpha1, alpha2)){
                    *alpha4=0;
                    return 1;
                }
                if(simplex_intersection4d(2, x3, x4, x0, x1, x2,
                                          alpha3, alpha4, alpha0, alpha1, alpha2)){
                    *alpha5=0;
                    return 1;
                }
                return 0;
            }
            
        case 4: // tetrahedron vs. segment
        case 5: // pentachoron vs. point
            return simplex_intersection4d(6-k, x5, x4, x3, x2, x1, x0,
                                          alpha5, alpha4, alpha3, alpha2, alpha1, alpha0);
        default:
            return -1; // should never get here
    }
}
