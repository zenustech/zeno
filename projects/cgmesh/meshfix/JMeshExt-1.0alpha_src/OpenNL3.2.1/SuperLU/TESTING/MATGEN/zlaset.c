#include <string.h>
#include "f2c.h"

/* Subroutine */ int zlaset_slu(char *uplo, integer *m, integer *n, 
	doublecomplex *alpha, doublecomplex *beta, doublecomplex *a, integer *
	lda)
{
/*  -- LAPACK auxiliary routine (version 2.0) --   
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,   
       Courant Institute, Argonne National Lab, and Rice University   
       October 31, 1992   


    Purpose   
    =======   

    ZLASET initializes a 2-D array A to BETA on the diagonal and   
    ALPHA on the offdiagonals.   

    Arguments   
    =========   

    UPLO    (input) CHARACTER*1   
            Specifies the part of the matrix A to be set.   
            = 'U':      Upper triangular part is set. The lower triangle 
  
                        is unchanged.   
            = 'L':      Lower triangular part is set. The upper triangle 
  
                        is unchanged.   
            Otherwise:  All of the matrix A is set.   

    M       (input) INTEGER   
            On entry, M specifies the number of rows of A.   

    N       (input) INTEGER   
            On entry, N specifies the number of columns of A.   

    ALPHA   (input) COMPLEX*16   
            All the offdiagonal array elements are set to ALPHA.   

    BETA    (input) COMPLEX*16   
            All the diagonal array elements are set to BETA.   

    A       (input/output) COMPLEX*16 array, dimension (LDA,N)   
            On entry, the m by n matrix A.   
            On exit, A(i,j) = ALPHA, 1 <= i <= m, 1 <= j <= n, i.ne.j;   
                     A(i,i) = BETA , 1 <= i <= min(m,n)   

    LDA     (input) INTEGER   
            The leading dimension of the array A.  LDA >= max(1,M).   

    ===================================================================== 
  


    
   Parameter adjustments   
       Function Body */
    /* System generated locals */

    /* Local variables */
    static integer i, j;


#define A(I,J) a[(I)-1 + ((J)-1)* ( *lda)]

    if (strncmp(uplo, "U", 1)==0) {
/*        Set the diagonal to BETA and the strictly upper triangular 
  
          part of the array to ALPHA. */
	for (j = 2; j <= *n; ++j) {
/* Computing MIN */
	    for (i = 1; i <= min(j-1,*m); ++i) {
		A(i,j).r = alpha->r, A(i,j).i = alpha->i;
/* L10: */
	    }
/* L20: */
	}

	for (i = 1; i <= min(*n,*m); ++i) {
	    A(i,i).r = beta->r, A(i,i).i = beta->i;
/* L30: */
	}
    } else if (strncmp(uplo, "L", 1)==0) {
/*        Set the diagonal to BETA and the strictly lower triangular 
  
          part of the array to ALPHA. */
	for (j = 1; j <= min(*m,*n); ++j) {

	    for (i = j + 1; i <= *m; ++i) {
		A(i,j).r = alpha->r, A(i,j).i = alpha->i;
/* L40: */
	    }
/* L50: */
	}

	for (i = 1; i <= min(*n,*m); ++i) {
	    A(i,i).r = beta->r, A(i,i).i = beta->i;
/* L60: */
	}
    } else {
/*        Set the array to BETA on the diagonal and ALPHA on the   
          offdiagonal. */
	for (j = 1; j <= *n; ++j) {

	    for (i = 1; i <= *m; ++i) {
		A(i,j).r = alpha->r, A(i,j).i = alpha->i;
/* L70: */
	    }
/* L80: */
	}

	for (i = 1; i <= min(*m,*n); ++i) {
	    A(i,i).r = beta->r, A(i,i).i = beta->i;
/* L90: */
	}
    }

    return 0;

/*     End of ZLASET */

} /* zlaset_slu */

