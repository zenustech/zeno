#include "f2c.h"
/*#include "blaswrap.h"*/

/* Subroutine */ int zswap_(integer *n, doublecomplex *zx, integer *incx, 
	doublecomplex *zy, integer *incy)
{
    /* System generated locals */
    

    /* Local variables */
    integer i__, ix, iy;
    doublecomplex ztemp;

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*     interchanges two vectors. */
/*     jack dongarra, 3/11/78. */
/*     modified 12/3/93, array(1) declarations changed to array(*) */


/*     .. Local Scalars .. */
/*     .. */
    /* Parameter adjustments */
    --zy;
    --zx;

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
	ztemp.r = zx[ix].r, ztemp.i = zx[ix].i;
	zx[ix].r = zy[iy].r, zx[ix].i = zy[iy].i;
	zy[iy].r = ztemp.r, zy[iy].i = ztemp.i;
	ix += *incx;
	iy += *incy;
/* L10: */
    }
    return 0;

/*       code for both increments equal to 1 */
L20:
    for (i__ = 1; i__ <= *n; ++i__) {
	ztemp.r = zx[i__].r, ztemp.i = zx[i__].i;
	zx[i__].r = zy[i__].r, zx[i__].i = zy[i__].i;
	zy[i__].r = ztemp.r, zy[i__].i = ztemp.i;
/* L30: */
    }
    return 0;
} /* zswap_ */
