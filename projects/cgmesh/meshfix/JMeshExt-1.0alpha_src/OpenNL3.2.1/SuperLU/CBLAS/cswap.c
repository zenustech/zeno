#include "f2c.h"
/*#include "blaswrap.h"*/

/* Subroutine */ int cswap_(integer *n, complex *cx, integer *incx, complex *
	cy, integer *incy)
{
    /* System generated locals */
    

    /* Local variables */
    integer i__, ix, iy;
    complex ctemp;

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*     interchanges two vectors. */
/*     jack dongarra, linpack, 3/11/78. */
/*     modified 12/3/93, array(1) declarations changed to array(*) */


/*     .. Local Scalars .. */
/*     .. */
    /* Parameter adjustments */
    --cy;
    --cx;

    /* Function Body */
    if (*n <= 0) {
	return 0;
    }
    if (*incx == 1 && *incy == 1) {
	goto L20;
    }

/*       code for unequal increments or equal increments not equal */
/*         to 1 */

    ix = 1;
    iy = 1;
    if (*incx < 0) {
	ix = (-(*n) + 1) * *incx + 1;
    }
    if (*incy < 0) {
	iy = (-(*n) + 1) * *incy + 1;
    }
    for (i__ = 1; i__ <= *n; ++i__) {
	ctemp.r = cx[ix].r, ctemp.i = cx[ix].i;
	cx[ix].r = cy[iy].r, cx[ix].i = cy[iy].i;
	cy[iy].r = ctemp.r, cy[iy].i = ctemp.i;
	ix += *incx;
	iy += *incy;
/* L10: */
    }
    return 0;

/*       code for both increments equal to 1 */
L20:
    for (i__ = 1; i__ <= *n; ++i__) {
	ctemp.r = cx[i__].r, ctemp.i = cx[i__].i;
	cx[i__].r = cy[i__].r, cx[i__].i = cy[i__].i;
	cy[i__].r = ctemp.r, cy[i__].i = ctemp.i;
/* L30: */
    }
    return 0;
} /* cswap_ */
