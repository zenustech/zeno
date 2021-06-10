// Released into the public domain by Robert Bridson, 2009.

#include <algorithm>
#include <cassert>
#include <cmath>
#include "fenv_include.h"
#include <limits>
#include <tunicate.h>

// short hand
static const double dbl_min=std::numeric_limits<double>::min();

//==============================================================================
double
sos_orientation1d(int priority0, const double* x0,
                  int priority1, const double* x1)
{
    assert(priority0!=priority1);
    double d=orientation1d(x0, x1);
    if(d) return d;
    // if we have an exact zero, use SoS to decide the sign 
    if(priority0>priority1) return dbl_min;
    else return -dbl_min;
}

//==============================================================================
namespace {
    
    void
    sort_points(int& priority0, const double*& x0,
                int& priority1, const double*& x1,
                double& sign)
    {
        if(priority0<priority1){
            std::swap(priority0, priority1);
            std::swap(x0, x1);
            sign=-sign;
        }
    }
    
} // namespace

//==============================================================================
double
sos_orientation2d(int priority0, const double* x0,
                  int priority1, const double* x1,
                  int priority2, const double* x2)
{
    assert(priority0!=priority1 && priority0!=priority2 && priority1!=priority2);
    double d=orientation2d(x0, x1, x2);
    if(d) return d;
    // If we have an exact zero, use SoS to decide the sign.
    // Sort by priority first, keeping track of sign of permutation.
    double sign=1;
    sort_points(priority0, x0, priority1, x1, sign);
    sort_points(priority0, x0, priority2, x2, sign);
    sort_points(priority1, x1, priority2, x2, sign);
    // Evaluate SoS terms one by one, looking for the first nonzero.
    
    // row 0
    d= orientation1d(x1+1, x2+1);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d=-orientation1d(x0+1, x2+1);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    
    // row 1
    d=-orientation1d(x1, x2);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    return -sign*dbl_min;
}

//==============================================================================
double
sos_orientation3d(int priority0, const double* x0,
                  int priority1, const double* x1,
                  int priority2, const double* x2,
                  int priority3, const double* x3)
{
    assert(priority0!=priority1 && priority0!=priority2 && priority0!=priority3);
    assert(priority1!=priority2 && priority1!=priority3);
    assert(priority2!=priority3);
    double d=orientation3d(x0, x1, x2, x3);
    if(d) return d;
    // If we have an exact zero, use SoS to decide the sign.
    // Sort by priority first, keeping track of sign of permutation.
    double sign=1;
    sort_points(priority0, x0, priority2, x2, sign);
    sort_points(priority1, x1, priority3, x3, sign);
    sort_points(priority0, x0, priority1, x1, sign);
    sort_points(priority2, x2, priority3, x3, sign);
    sort_points(priority1, x1, priority2, x2, sign);
    // Evaluate SoS terms one by one, looking for the first nonzero.
    // (We skip a few that must be zero if the preceding are zero, and stop
    // at the first term which must always be nonzero.)
    
    // row 0
    d= orientation2d(x1+1, x2+1, x3+1);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d=-orientation2d(x0+1, x2+1, x3+1);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d= orientation2d(x0+1, x1+1, x3+1);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d=-orientation2d(x0+1, x1+1, x2+1);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    
    // row 1
    double c0[2]={x0[0], x0[2]}, c1[2]={x1[0], x1[2]},
    c2[2]={x2[0], x2[2]}, c3[2]={x3[0], x3[2]};
    d=-orientation2d(c1, c2, c3);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d=-orientation1d(x2+2, x3+2);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d= orientation1d(x1+2, x3+2);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d= orientation2d(c0, c2, c3);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d= orientation1d(x0+2, x3+2);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    
    // row 2
    d= orientation2d(x1, x2, x3);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d= orientation1d(x2+1, x3+1);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d=-orientation1d(x1+1, x3+1);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d= orientation1d(x2, x3);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    return -sign*dbl_min;
}

//==============================================================================
double
sos_orientation4d(int priority0, const double* x0,
                  int priority1, const double* x1,
                  int priority2, const double* x2,
                  int priority3, const double* x3,
                  int priority4, const double* x4)
{
    assert(priority0!=priority1 && priority0!=priority2 && priority0!=priority3);
    assert(priority0!=priority4 && priority1!=priority2 && priority1!=priority3);
    assert(priority1!=priority4 && priority2!=priority3 && priority2!=priority4);
    assert(priority3!=priority4);
    double d=orientation4d(x0, x1, x2, x3, x4);
    if(d) return d;
    // If we have an exact zero, use SoS to decide the sign.
    // Sort by priority first, keeping track of sign of permutation.
    // (we could do a better job with this, but this is easy to code...)
    double sign=1;
    sort_points(priority0, x0, priority1, x1, sign);
    sort_points(priority2, x2, priority3, x3, sign);
    sort_points(priority0, x0, priority2, x2, sign);
    sort_points(priority1, x1, priority3, x3, sign);
    sort_points(priority1, x1, priority2, x2, sign);
    sort_points(priority0, x0, priority4, x4, sign);
    sort_points(priority1, x1, priority4, x4, sign);
    sort_points(priority2, x2, priority4, x4, sign);
    sort_points(priority3, x3, priority4, x4, sign);
    // Evaluate SoS terms one by one, looking for the first nonzero.
    // (We skip a few that must be zero if the preceding are zero, and stop
    // at the first term which must always be nonzero.)
    
    // row 0
    d= orientation3d(x1+1, x2+1, x3+1, x4+1);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d=-orientation3d(x0+1, x2+1, x3+1, x4+1);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d= orientation3d(x0+1, x1+1, x3+1, x4+1);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d=-orientation3d(x0+1, x1+1, x2+1, x4+1);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d= orientation3d(x0+1, x1+1, x2+1, x3+1);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    
    // row 1
    double c0[3]={x0[0], x0[2], x0[3]},
    c1[3]={x1[0], x1[2], x1[3]},
    c2[3]={x2[0], x2[2], x2[3]},
    c3[3]={x3[0], x3[2], x3[3]},
    c4[3]={x4[0], x4[2], x4[3]};
    d=-orientation3d(c1, c2, c3, c4);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d=-orientation2d(c2+1, c3+1, c4+1);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d= orientation2d(c1+1, c3+1, c4+1);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d=-orientation2d(c1+1, c2+1, c4+1);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d= orientation3d(c0, c2, c3, c4);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d=-orientation2d(c0+1, c3+1, c4+1);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d= orientation2d(c0+1, c2+1, c4+1);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d=-orientation3d(c0, c1, c3, c4);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d=-orientation2d(c0+1, c1+1, c4+1);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d= orientation3d(c0, c1, c2, c4);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d=-orientation3d(c0, c1, c2, c3);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    
    // row 2
    c0[1]=x0[1]; c1[1]=x1[1]; c2[1]=x2[1]; c3[1]=x3[1]; c4[1]=x4[1];
    d= orientation3d(c1, c2, c3, c4);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d= orientation2d(c2+1, c3+1, c4+1);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d=-orientation2d(c1+1, c3+1, c4+1);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d= orientation2d(c1+1, c2+1, c4+1);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    double b0[2]={x0[0], x0[3]},
    b1[2]={x1[0], x1[3]},
    b2[2]={x2[0], x2[3]},
    b3[2]={x3[0], x3[3]},
    b4[2]={x4[0], x4[3]};
    d=-orientation2d(b2, b3, b4);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d=-orientation1d(b3+1, b4+1);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d= orientation1d(b2+1, b4+1);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d= orientation2d(b1, b3, b4);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d=-orientation1d(b1+1, b4+1);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d=-orientation2d(b1, b2, b4);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d=-orientation3d(c0, c2, c3, c4);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d= orientation2d(c0+1, c3+1, c4+1);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d=-orientation2d(c0+1, c2+1, c4+1);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d=-orientation2d(b0, b3, b4);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d= orientation1d(b0+1, b4+1);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d= orientation3d(c0, c1, c3, c4);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d= orientation2d(c0+1, c1+1, c4+1);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d=-orientation2d(b0, b1, b4);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d=-orientation3d(c0, c1, c2, c4);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    d= orientation3d(c0, c1, c2, c3);
    if(d<0) return sign*dbl_min; else if(d>0) return -sign*dbl_min;
    
    // row 3
    return sign*sos_orientation3d(priority1, x1, priority2, x2,
                                  priority3, x3, priority4, x4);
}
