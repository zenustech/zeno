#include <float.h>
#include <math.h>
#include <stdio.h>

float smach(char *cmach)
{
/*  -- SuperLU auxiliary routine (version 5.0) --   
    This uses C99 standard constants, and is thread safe.

    Must be compiled with "-std=c99" flag.


    Purpose   
    =======   

    SMACH returns single precision machine parameters.   

    Arguments   
    =========   

    CMACH   (input) CHARACTER*1   
            Specifies the value to be returned by SMACH:   
            = 'E' or 'e',   SMACH := eps   
            = 'S' or 's ,   SMACH := sfmin   
            = 'B' or 'b',   SMACH := base   
            = 'P' or 'p',   SMACH := eps*base   
            = 'N' or 'n',   SMACH := t   
            = 'R' or 'r',   SMACH := rnd   
            = 'M' or 'm',   SMACH := emin   
            = 'U' or 'u',   SMACH := rmin   
            = 'L' or 'l',   SMACH := emax   
            = 'O' or 'o',   SMACH := rmax   

            where   

            eps   = relative machine precision   
            sfmin = safe minimum, such that 1/sfmin does not overflow   
            base  = base of the machine   
            prec  = eps*base   
            t     = number of (base) digits in the mantissa   
            rnd   = 1.0 when rounding occurs in addition, 0.0 otherwise   
            emin  = minimum exponent before (gradual) underflow   
            rmin  = underflow threshold - base**(emin-1)   
            emax  = largest exponent before overflow   
            rmax  = overflow threshold  - (base**emax)*(1-eps)   

   ===================================================================== 
*/

    float sfmin, small, rmach;
    extern int lsame_(char *, char *);

    if (lsame_(cmach, "E")) {
	rmach = FLT_EPSILON * 0.5;
    } else if (lsame_(cmach, "S")) {
	sfmin = FLT_MIN;
	small = 1. / FLT_MAX;
	if (small >= sfmin) {
	    /* Use SMALL plus a bit, to avoid the possibility of rounding   
	       causing overflow when computing  1/sfmin. */
	    sfmin = small * (FLT_EPSILON*0.5 + 1.);
	}
	rmach = sfmin;
    } else if (lsame_(cmach, "B")) {
	rmach = FLT_RADIX;
    } else if (lsame_(cmach, "P")) {
	rmach = FLT_EPSILON * 0.5 * FLT_RADIX;
    } else if (lsame_(cmach, "N")) {
	rmach = FLT_MANT_DIG;
    } else if (lsame_(cmach, "R")) {
	rmach = FLT_ROUNDS;
    } else if (lsame_(cmach, "M")) {
	rmach = FLT_MIN_EXP;
    } else if (lsame_(cmach, "U")) {
	rmach = FLT_MIN;
    } else if (lsame_(cmach, "L")) {
	rmach = FLT_MAX_EXP;
    } else if (lsame_(cmach, "O")) {
	rmach = FLT_MAX;
    }

    return rmach;

} /* end smach */
