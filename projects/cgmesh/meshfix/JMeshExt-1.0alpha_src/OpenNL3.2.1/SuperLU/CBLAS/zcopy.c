
/*  -- translated by f2c (version 19940927).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

#include "f2c.h"

/* Subroutine */ int zcopy_(integer *n, doublecomplex *zx, integer *incx, 
	doublecomplex *zy, integer *incy)
{


    /* System generated locals */


    /* Local variables */
    integer i, ix, iy;


/*     copies a vector, x, to a vector, y.   
       jack dongarra, linpack, 4/11/78.   
       modified 12/3/93, array(1) declarations changed to array(*)   


    
   Parameter adjustments   
       Function Body */
#define ZY(I) zy[(I)-1]
#define ZX(I) zx[(I)-1]


    if (*n <= 0) {
	return 0;
    }
    if (*incx == 1 && *incy == 1) {
	goto L20;
    }

/*        code for unequal increments or equal increments   
            not equal to 1 */

    ix = 1;
    iy = 1;
    if (*incx < 0) {
	ix = (-(*n) + 1) * *incx + 1;
    }
    if (*incy < 0) {
	iy = (-(*n) + 1) * *incy + 1;
    }
    for (i = 1; i <= *n; ++i) {
	ZY(iy).r = ZX(ix).r, ZY(iy).i = ZX(ix).i;
	ix += *incx;
	iy += *incy;
/* L10: */
    }
    return 0;

/*        code for both increments equal to 1 */

L20:
    for (i = 1; i <= *n; ++i) {
	ZY(i).r = ZX(i).r, ZY(i).i = ZX(i).i;
/* L30: */
    }
    return 0;
} /* zcopy_ */

