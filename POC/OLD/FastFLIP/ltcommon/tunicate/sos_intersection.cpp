// Released into the public domain by Robert Bridson, 2009.

#include <cassert>
#include <tunicate.h>
#include <cstdlib>

//==============================================================================
// Note: it is assumed all arguments are nonzero (have a sign).

namespace {
    bool
    same_sign(double a, double b)
    {
        return (a<0 && b<0) || (a>0 && b>0);
    }
}

//==============================================================================
int
sos_simplex_intersection1d(int k,
                           int pri0, const double* x0,
                           int pri1, const double* x1,
                           int pri2, const double* x2,
                           double* alpha0, 
                           double* alpha1, 
                           double* alpha2)
{
    assert(1<=k && k<=2);
    assert(alpha0 && alpha1 && alpha2);
    
    if(alpha0 == NULL || alpha1 == NULL || alpha2 == NULL) //prevent null pointer warning
       return -1;

    double sum;
    switch(k){
        case 1: // point vs. segment
            *alpha1=-sos_orientation1d(pri0, x0, pri2, x2);
            *alpha2= sos_orientation1d(pri0, x0, pri1, x1);
            if(same_sign(*alpha1, *alpha2)){
                *alpha0=1;
                sum=*alpha1+*alpha2;
                *alpha1/=sum;
                *alpha2/=sum;
                return 1;
            }else
                return 0;
        case 2: // segment vs. point
            return sos_simplex_intersection1d(1, pri2, x2,
                                              pri1, x1,
                                              pri0, x0,
                                              alpha2, alpha1, alpha0);
        default:
            return -1; // should never get here
    }
}

//==============================================================================
int
sos_simplex_intersection2d(int k,
                           int pri0, const double* x0,
                           int pri1, const double* x1,
                           int pri2, const double* x2,
                           int pri3, const double* x3,
                           double* alpha0, 
                           double* alpha1, 
                           double* alpha2,
                           double* alpha3)
{
    assert(1<=k && k<=3);
    double sum1, sum2;
    switch(k){
        case 1: // point vs. triangle
            *alpha1=-sos_orientation2d(pri0, x0, pri2, x2, pri3, x3);
            *alpha2= sos_orientation2d(pri0, x0, pri1, x1, pri3, x3);
            if(!same_sign(*alpha1, *alpha2)) return 0;
            *alpha3=-sos_orientation2d(pri0, x0, pri1, x1, pri2, x2);
            if(!same_sign(*alpha1, *alpha3)) return 0;
            *alpha0=1;
            sum2=*alpha1+*alpha2+*alpha3;
            *alpha1/=sum2;
            *alpha2/=sum2;
            *alpha3/=sum2;
            return 1;
        case 2: // segment vs. segment
            *alpha0= sos_orientation2d(pri1, x1, pri2, x2, pri3, x3);
            *alpha1=-sos_orientation2d(pri0, x0, pri2, x2, pri3, x3);
            if(!same_sign(*alpha0, *alpha1)) return 0;
            *alpha2= sos_orientation2d(pri0, x0, pri1, x1, pri3, x3);
            *alpha3=-sos_orientation2d(pri0, x0, pri1, x1, pri2, x2);
            if(!same_sign(*alpha2, *alpha3)) return 0;
            sum1=*alpha0+*alpha1;
            *alpha0/=sum1;
            *alpha1/=sum1;
            sum2=*alpha2+*alpha3;
            *alpha2/=sum2;
            *alpha3/=sum2;
            return 1;
        case 3: // triangle vs. point
            return sos_simplex_intersection2d(1, pri3, x3,
                                              pri2, x2,
                                              pri1, x1,
                                              pri0, x0,
                                              alpha3, alpha2, alpha1, alpha0);
        default:
            return -1; // should never get here
    }
}

//==============================================================================
int
sos_simplex_intersection3d(int k,
                           int pri0, const double* x0,
                           int pri1, const double* x1,
                           int pri2, const double* x2,
                           int pri3, const double* x3,
                           int pri4, const double* x4,
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
            *alpha1=-sos_orientation3d(pri0, x0, pri2, x2, pri3, x3, pri4, x4);
            *alpha2= sos_orientation3d(pri0, x0, pri1, x1, pri3, x3, pri4, x4);
            if(!same_sign(*alpha1, *alpha2)) return 0;
            *alpha3=-sos_orientation3d(pri0, x0, pri1, x1, pri2, x2, pri4, x4);
            if(!same_sign(*alpha1, *alpha3)) return 0;
            *alpha4= sos_orientation3d(pri0, x0, pri1, x1, pri2, x2, pri3, x3);
            if(!same_sign(*alpha1, *alpha4)) return 0;
            *alpha0=1;
            sum2=*alpha1+*alpha2+*alpha3+*alpha4;
            *alpha1/=sum2;
            *alpha2/=sum2;
            *alpha3/=sum2;
            *alpha4/=sum2;
            return 1;
        case 2: // segment vs. triangle
            *alpha0= sos_orientation3d(pri1, x1, pri2, x2, pri3, x3, pri4, x4);
            *alpha1=-sos_orientation3d(pri0, x0, pri2, x2, pri3, x3, pri4, x4);
            if(!same_sign(*alpha0, *alpha1)) return 0;
            *alpha2= sos_orientation3d(pri0, x0, pri1, x1, pri3, x3, pri4, x4);
            *alpha3=-sos_orientation3d(pri0, x0, pri1, x1, pri2, x2, pri4, x4);
            if(!same_sign(*alpha2, *alpha3)) return 0;
            *alpha4= sos_orientation3d(pri0, x0, pri1, x1, pri2, x2, pri3, x3);
            if(!same_sign(*alpha2, *alpha4)) return 0;
            sum1=*alpha0+*alpha1;
            *alpha0/=sum1;
            *alpha1/=sum1;
            sum2=*alpha2+*alpha3+*alpha4;
            *alpha2/=sum2;
            *alpha3/=sum2;
            *alpha4/=sum2;
            return 1;
        case 3: // triangle vs. segment
        case 4: // tetrahedron vs. point
            return sos_simplex_intersection3d(5-k, pri4, x4,
                                              pri3, x3,
                                              pri2, x2,
                                              pri1, x1,
                                              pri0, x0,
                                              alpha4, alpha3, alpha2, alpha1, alpha0);
        default:
            return -1; // should never get here
    }
}

//==============================================================================
int
sos_simplex_intersection4d(int k,
                           int pri0, const double* x0,
                           int pri1, const double* x1,
                           int pri2, const double* x2,
                           int pri3, const double* x3,
                           int pri4, const double* x4,
                           int pri5, const double* x5,
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
            *alpha1=-sos_orientation4d(pri0,x0,pri2,x2,pri3,x3,pri4,x4,pri5,x5);
            *alpha2= sos_orientation4d(pri0,x0,pri1,x1,pri3,x3,pri4,x4,pri5,x5);
            if(!same_sign(*alpha1, *alpha2)) return 0;
            *alpha3=-sos_orientation4d(pri0,x0,pri1,x1,pri2,x2,pri4,x4,pri5,x5);
            if(!same_sign(*alpha1, *alpha3)) return 0;
            *alpha4= sos_orientation4d(pri0,x0,pri1,x1,pri2,x2,pri3,x3,pri5,x5);
            if(!same_sign(*alpha1, *alpha4)) return 0;
            *alpha5=-sos_orientation4d(pri0,x0,pri1,x1,pri2,x2,pri3,x3,pri4,x4);
            if(!same_sign(*alpha1, *alpha5)) return 0;
            *alpha0=1;
            sum2=*alpha1+*alpha2+*alpha3+*alpha4+*alpha5;
            *alpha1/=sum2;
            *alpha2/=sum2;
            *alpha3/=sum2;
            *alpha4/=sum2;
            *alpha5/=sum2;
            return 1;
        case 2: // segment vs. tetrahedron
            *alpha0= sos_orientation4d(pri1,x1,pri2,x2,pri3,x3,pri4,x4,pri5,x5);
            *alpha1=-sos_orientation4d(pri0,x0,pri2,x2,pri3,x3,pri4,x4,pri5,x5);
            if(!same_sign(*alpha0, *alpha1)) return 0;
            *alpha2= sos_orientation4d(pri0,x0,pri1,x1,pri3,x3,pri4,x4,pri5,x5);
            *alpha3=-sos_orientation4d(pri0,x0,pri1,x1,pri2,x2,pri4,x4,pri5,x5);
            if(!same_sign(*alpha2, *alpha3)) return 0;
            *alpha4= sos_orientation4d(pri0,x0,pri1,x1,pri2,x2,pri3,x3,pri5,x5);
            if(!same_sign(*alpha2, *alpha4)) return 0;
            *alpha5=-sos_orientation4d(pri0,x0,pri1,x1,pri2,x2,pri3,x3,pri4,x4);
            if(!same_sign(*alpha2, *alpha5)) return 0;
            sum1=*alpha0+*alpha1;
            *alpha0/=sum1;
            *alpha1/=sum1;
            sum2=*alpha2+*alpha3+*alpha4+*alpha5;
            *alpha2/=sum2;
            *alpha3/=sum2;
            *alpha4/=sum2;
            *alpha5/=sum2;
            return 1;
        case 3: // triangle vs. triangle
            *alpha0= sos_orientation4d(pri1,x1,pri2,x2,pri3,x3,pri4,x4,pri5,x5);
            *alpha1=-sos_orientation4d(pri0,x0,pri2,x2,pri3,x3,pri4,x4,pri5,x5);
            if(!same_sign(*alpha0, *alpha1)) return 0;
            *alpha2= sos_orientation4d(pri0,x0,pri1,x1,pri3,x3,pri4,x4,pri5,x5);
            if(!same_sign(*alpha0, *alpha2)) return 0;
            *alpha3=-sos_orientation4d(pri0,x0,pri1,x1,pri2,x2,pri4,x4,pri5,x5);
            *alpha4= sos_orientation4d(pri0,x0,pri1,x1,pri2,x2,pri3,x3,pri5,x5);
            if(!same_sign(*alpha3, *alpha4)) return 0;
            *alpha5=-sos_orientation4d(pri0,x0,pri1,x1,pri2,x2,pri3,x3,pri4,x4);
            if(!same_sign(*alpha3, *alpha5)) return 0;
            sum1=*alpha0+*alpha1+*alpha2;
            *alpha0/=sum1;
            *alpha1/=sum1;
            *alpha2/=sum1;
            sum2=*alpha3+*alpha4+*alpha5;
            *alpha3/=sum2;
            *alpha4/=sum2;
            *alpha5/=sum2;
            return 1;
        case 4: // tetrahedron vs. segment
        case 5: // pentachoron vs. point
            return sos_simplex_intersection4d(6-k, pri5, x5,
                                              pri4, x4,
                                              pri3, x3,
                                              pri2, x2,
                                              pri1, x1,
                                              pri0, x0,
                                              alpha5, alpha4, alpha3, alpha2, alpha1, alpha0);
        default:
            return -1; // should never get here
    }
}
