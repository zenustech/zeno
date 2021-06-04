// Released into the public domain by Robert Bridson, 2009.

#include <cassert>
#include <fenv_include.h>
#include <cmath>
#include <limits>
#include <tunicate.h>
#include <expansion.h>
#include <neg.h>

//==============================================================================
double
orientation1d(const double* x0,
              const double* x1)
{
    assert(x0 != NULL && x1 != NULL);

    return x0[0]-x1[0];
}

//==============================================================================
static double
simple_orientation2d(const double* x0,
                     const double* x1,
                     const double* x2)
{
    return x0[0]*x1[1] + neg(x0[0])*x2[1]
    + x1[0]*x2[1] + neg(x1[0])*x0[1]
    + x2[0]*x0[1] + neg(x2[0])*x1[1];
}

//==============================================================================
void
interval_orientation2d(const double* x0,
                       const double* x1,
                       const double* x2,
                       double* lower,
                       double* upper)
{
    fesetround(FE_DOWNWARD);
    *lower=simple_orientation2d(x0, x1, x2);
    fesetround(FE_UPWARD);
    *upper=simple_orientation2d(x0, x1, x2);
    assert(*lower<=*upper);
}

//==============================================================================
// calculates the exact result, as an expansion.
static void
expansion_orientation2d(const double* x0,
                        const double* x1,
                        const double* x2,
                        expansion& result)
{
    expansion product;
    multiply(x0[0],  x1[1], result);
    multiply(x0[0], -x2[1], product); add(result, product, result);
    multiply(x1[0],  x2[1], product); add(result, product, result);
    multiply(x1[0], -x0[1], product); add(result, product, result);
    multiply(x2[0],  x0[1], product); add(result, product, result);
    multiply(x2[0], -x1[1], product); add(result, product, result);
}

//==============================================================================
// returns a sign-accurate result
// (zero if and only if the true answer is zero or underflows too far)
static double
accurate_orientation2d(const double* x0,
                       const double* x1,
                       const double* x2)
{
    expansion result;
    expansion_orientation2d(x0, x1, x2, result);
    return estimate(result);
}

//==============================================================================
double
orientation2d(const double* x0,
              const double* x1,
              const double* x2)
{
    assert(x0 && x1 && x2);
    double lower, upper;
    interval_orientation2d(x0, x1, x2, &lower, &upper);
    fesetround(FE_TONEAREST);
    if(upper<0 || lower>0)
        return 0.5*(lower+upper);
    else if(lower==upper) // and hence are both equal to zero
        return 0;
    else // not an exact zero - we don't know sign for sure
        return accurate_orientation2d(x0, x1, x2);
}

//==============================================================================
// Multiply three numbers together in a way where the rounding mode is
// respected (upward or downward) no matter the signs of the factors.
static double
three_product(double a,
              double b,
              double c)
{
    if(a>0)
        return a*(b*c);
    else
        return neg(a)*(-b*c);
}

//==============================================================================
static double
simple_orientation3d(const double* x0,
                     const double* x1,
                     const double* x2,
                     const double* x3)
{
    return three_product( x0[0], x1[1], x2[2])
    + three_product(-x0[0], x1[1], x3[2])
    + three_product(-x0[0], x2[1], x1[2])
    + three_product( x0[0], x2[1], x3[2])
    + three_product( x0[0], x3[1], x1[2])
    + three_product(-x0[0], x3[1], x2[2])
    
    + three_product(-x1[0], x0[1], x2[2])
    + three_product( x1[0], x0[1], x3[2])
    + three_product( x1[0], x2[1], x0[2])
    + three_product(-x1[0], x2[1], x3[2])
    + three_product(-x1[0], x3[1], x0[2])
    + three_product( x1[0], x3[1], x2[2])
    
    + three_product( x2[0], x0[1], x1[2])
    + three_product(-x2[0], x0[1], x3[2])
    + three_product(-x2[0], x1[1], x0[2])
    + three_product( x2[0], x1[1], x3[2])
    + three_product( x2[0], x3[1], x0[2])
    + three_product(-x2[0], x3[1], x1[2])
    
    + three_product(-x3[0], x0[1], x1[2])
    + three_product( x3[0], x0[1], x2[2])
    + three_product( x3[0], x1[1], x0[2])
    + three_product(-x3[0], x1[1], x2[2])
    + three_product(-x3[0], x2[1], x0[2])
    + three_product( x3[0], x2[1], x1[2]);
}

//==============================================================================
void
interval_orientation3d(const double* x0,
                       const double* x1,
                       const double* x2,
                       const double* x3,
                       double* lower,
                       double* upper)
{
    fesetround(FE_DOWNWARD);
    *lower=simple_orientation3d(x0, x1, x2, x3);
    fesetround(FE_UPWARD);
    *upper=simple_orientation3d(x0, x1, x2, x3);
#ifdef _MSC_VER
    assert( !_isnan(*lower) );
    assert( !_isnan(*upper) );
    assert( _finite(*lower) );
    assert( _finite(*upper) );
#else
    assert( !std::isnan(*lower) );
    assert( !std::isnan(*upper) );
    assert( !std::isinf(*lower) );
    assert( !std::isinf(*upper) );
#endif
    assert(*lower<=*upper);
}

//==============================================================================
// calculates the exact result, as an expansion.
static void
expansion_orientation3d(const double* x0,
                        const double* x1,
                        const double* x2,
                        const double* x3,
                        expansion& result)
{
    expansion d, p;
    expansion_orientation2d(x1, x2, x3, d);
    multiply(x0[2], d, result);
    
    expansion_orientation2d(x0, x2, x3, d);
    multiply(-x1[2], d, p); 
    add(result, p, result);
    
    expansion_orientation2d(x0, x1, x3, d);
    multiply(x2[2], d, p); 
    add(result, p, result);
    
    expansion_orientation2d(x0, x1, x2, d);
    multiply(-x3[2], d, p); 
    add(result, p, result);
    
}

//==============================================================================
// returns a sign-accurate result
// (zero if and only if the true answer is zero or underflows too far)
static double
accurate_orientation3d(const double* x0,
                       const double* x1,
                       const double* x2,
                       const double* x3)
{
    expansion result;
    expansion_orientation3d(x0, x1, x2, x3, result);
    return estimate(result);
}

//==============================================================================
double
orientation3d(const double* x0,
              const double* x1,
              const double* x2,
              const double* x3)
{
    assert(x0 && x1 && x2 && x3);
    double lower, upper;
    interval_orientation3d(x0, x1, x2, x3, &lower, &upper);
    fesetround(FE_TONEAREST);
    if(upper<0 || lower>0)
        return 0.5*(lower+upper);
    else if(lower==upper) // and hence exactly zero
        return 0;
    else // not an exact zero - we don't know sign for sure
        return accurate_orientation3d(x0, x1, x2, x3);
}

//==============================================================================
static void
interval_orientation_time3d(const double* x0, int time0,
                            const double* x1, int time1,
                            const double* x2, int time2,
                            const double* x3, int time3,
                            const double* x4, int time4,
                            double* lower,
                            double* upper)
{
    double lower1234=0, upper1234=0;
    if(time0) interval_orientation3d(x1, x2, x3, x4, &lower1234, &upper1234);
    double lower0234=0, upper0234=0;
    if(time1) interval_orientation3d(x0, x2, x3, x4, &lower0234, &upper0234);
    double lower0134=0, upper0134=0;
    if(time2) interval_orientation3d(x0, x1, x3, x4, &lower0134, &upper0134);
    double lower0124=0, upper0124=0;
    if(time3) interval_orientation3d(x0, x1, x2, x4, &lower0124, &upper0124);
    double lower0123=0, upper0123=0;
    if(time4) interval_orientation3d(x0, x1, x2, x3, &lower0123, &upper0123);
    fesetround(FE_DOWNWARD);
    *lower=-upper1234+lower0234-upper0134+lower0124-upper0123;
    fesetround(FE_UPWARD);
    *upper=-lower1234+upper0234-lower0134+upper0124-lower0123;
    assert(*lower<=*upper);
}

//==============================================================================
// calculates the exact result, as an expansion.
static void
expansion_orientation_time3d(const double* x0, int time0,
                             const double* x1, int time1,
                             const double* x2, int time2,
                             const double* x3, int time3,
                             const double* x4, int,
                             expansion& result)
{
    make_zero(result);
    expansion d;
    if(time0){
        expansion_orientation3d(x1, x2, x3, x4, d);
        negative(d, result);
    }
    if(time1){
        expansion_orientation3d(x0, x2, x3, x4, d);
        add(result, d, result);
    }
    if(time2){
        expansion_orientation3d(x0, x1, x3, x4, d);
        subtract(result, d, result);
    }
    if(time3){
        expansion_orientation3d(x0, x1, x2, x4, d);
        add(result, d, result);
    }
    if(time3){
        expansion_orientation3d(x0, x1, x2, x3, d);
        subtract(result, d, result);
    }
}

//==============================================================================
// returns a sign-accurate result
// (zero if and only if the true answer is zero or underflows too far)
static double
accurate_orientation_time3d(const double* x0, int time0,
                            const double* x1, int time1,
                            const double* x2, int time2,
                            const double* x3, int time3,
                            const double* x4, int time4)
{
    expansion result;
    expansion_orientation_time3d(x0, time0, x1, time1, x2, time2,
                                 x3, time3, x4, time4, result);
    return estimate(result);
}

//==============================================================================
double
orientation_time3d(const double* x0, int time0,
                   const double* x1, int time1,
                   const double* x2, int time2,
                   const double* x3, int time3,
                   const double* x4, int time4)
{
    assert(x0 && x1 && x2 && x3);
    double lower, upper;
    interval_orientation_time3d(x0, time0, x1, time1, x2, time2, x3, time3,
                                x4, time4, &lower, &upper);
    fesetround(FE_TONEAREST);
    if(upper<0 || lower>0)
        return 0.5*(lower+upper);
    else if(lower==upper) // and hence exactly zero
        return 0;
    else // not an exact zero - we don't know sign for sure
        return accurate_orientation_time3d(x0, time0, x1, time1, x2, time2,
                                           x3, time3, x4, time4);
}

//==============================================================================
static void
interval_orientation4d(const double* x0,
                       const double* x1,
                       const double* x2,
                       const double* x3,
                       const double* x4,
                       double* lower,
                       double* upper)
{
    double lower1234, upper1234;
    interval_orientation3d(x1+1, x2+1, x3+1, x4+1, &lower1234, &upper1234);
    double lower0234, upper0234;
    interval_orientation3d(x0+1, x2+1, x3+1, x4+1, &lower0234, &upper0234);
    double lower0134, upper0134;
    interval_orientation3d(x0+1, x1+1, x3+1, x4+1, &lower0134, &upper0134);
    double lower0124, upper0124;
    interval_orientation3d(x0+1, x1+1, x2+1, x4+1, &lower0124, &upper0124);
    double lower0123, upper0123;
    interval_orientation3d(x0+1, x1+1, x2+1, x3+1, &lower0123, &upper0123);
    fesetround(FE_DOWNWARD);
    *lower= x0[0]*(x0[0]<0 ? upper1234 : lower1234)
    +neg(x1[0])*(x1[0]>0 ? upper0234 : lower0234)
    +x2[0]*(x2[0]<0 ? upper0134 : lower0134)
    +neg(x3[0])*(x3[0]>0 ? upper0124 : lower0124)
    +x4[0]*(x4[0]<0 ? upper0123 : lower0123);
    fesetround(FE_UPWARD);
    *upper= x0[0]*(x0[0]>0 ? upper1234 : lower1234)
    +neg(x1[0])*(x1[0]<0 ? upper0234 : lower0234)
    +x2[0]*(x2[0]>0 ? upper0134 : lower0134)
    +neg(x3[0])*(x3[0]<0 ? upper0124 : lower0124)
    +x4[0]*(x4[0]>0 ? upper0123 : lower0123);
    assert(*lower<=*upper);
}

//==============================================================================
// calculates the exact result, as an expansion.
static void
expansion_orientation4d(const double* x0,
                        const double* x1,
                        const double* x2,
                        const double* x3,
                        const double* x4,
                        expansion& result)
{
    // do the 2d subsets
    expansion d012, d013, d014, d023, d024, d034, d123, d124, d134, d234;
    expansion_orientation2d(x0+2, x1+2, x2+2, d012);
    expansion_orientation2d(x0+2, x1+2, x3+2, d013);
    expansion_orientation2d(x0+2, x1+2, x4+2, d014);
    expansion_orientation2d(x0+2, x2+2, x3+2, d023);
    expansion_orientation2d(x0+2, x2+2, x4+2, d024);
    expansion_orientation2d(x0+2, x3+2, x4+2, d034);
    expansion_orientation2d(x1+2, x2+2, x3+2, d123);
    expansion_orientation2d(x1+2, x2+2, x4+2, d124);
    expansion_orientation2d(x1+2, x3+2, x4+2, d134);
    expansion_orientation2d(x2+2, x3+2, x4+2, d234);
    
    // then the 3d subsets
    expansion d0123, d0124, d0134, d0234, d1234, product;
    // d0123=x0[1]*d123+(-x1[1])*d023+x2[1]*d013+(-x3[1])*d012,
    multiply( x0[1], d123, d0123);
    multiply(-x1[1], d023, product); add(d0123, product, d0123);
    multiply( x2[1], d013, product); add(d0123, product, d0123);
    multiply(-x3[1], d012, product); add(d0123, product, d0123);
    // d0124=x0[1]*d124+(-x1[1])*d024+x2[1]*d014+(-x4[1])*d012,
    multiply( x0[1], d124, d0124);
    multiply(-x1[1], d024, product); add(d0124, product, d0124);
    multiply( x2[1], d014, product); add(d0124, product, d0124);
    multiply(-x4[1], d012, product); add(d0124, product, d0124);
    
    // d0134=x0[1]*d134+(-x1[1])*d034+x3[1]*d014+(-x4[1])*d013,
    multiply( x0[1], d134, d0134);
    multiply(-x1[1], d034, product); add(d0134, product, d0134);
    multiply( x3[1], d014, product); add(d0134, product, d0134);
    multiply(-x4[1], d013, product); add(d0134, product, d0134);
    
    // d0234=x0[1]*d234+(-x2[1])*d034+x3[1]*d024+(-x4[1])*d023,
    multiply( x0[1], d234, d0234);
    multiply(-x2[1], d034, product); add(d0234, product, d0234);
    multiply( x3[1], d024, product); add(d0234, product, d0234);
    multiply(-x4[1], d023, product); add(d0234, product, d0234);
    
    // d1234=x1[1]*d234+(-x2[1])*d134+x3[1]*d124+(-x4[1])*d123;
    multiply( x1[1], d234, d1234);
    multiply(-x2[1], d134, product); add(d1234, product, d1234);
    multiply( x3[1], d124, product); add(d1234, product, d1234);
    multiply(-x4[1], d123, product); add(d1234, product, d1234);
    
    // and finally get the 4d answer
    // x0[0]*d1234+(-x1[0])*d0234+x2[0]*d0134+(-x3[0])*d0124+x4[0]*d0123;
    multiply( x0[0], d1234, result);
    multiply(-x1[0], d0234, product); add(result, product, result);
    multiply( x2[0], d0134, product); add(result, product, result);
    multiply(-x3[0], d0124, product); add(result, product, result);
    multiply( x4[0], d0123, product); add(result, product, result);
}

//==============================================================================
// returns a sign-accurate result
// (zero if and only if the true answer is zero or underflows too far)
static double
accurate_orientation4d(const double* x0,
                       const double* x1,
                       const double* x2,
                       const double* x3,
                       const double* x4)
{
    expansion result;
    expansion_orientation4d(x0, x1, x2, x3, x4, result);
    return estimate(result);
}

//==============================================================================
double
orientation4d(const double* x0,
              const double* x1,
              const double* x2,
              const double* x3,
              const double* x4)
{
    assert(x0 && x1 && x2 && x3 && x4);
    double lower, upper;
    interval_orientation4d(x0, x1, x2, x3, x4, &lower, &upper);
    fesetround(FE_TONEAREST);
    if(upper<0 || lower>0)
        return 0.5*(lower+upper);
    else if(lower==upper) // and hence exactly zero
        return 0;
    else // not an exact zero - we don't know sign for sure
        return accurate_orientation4d(x0, x1, x2, x3, x4);
}
