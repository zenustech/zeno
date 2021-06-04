#ifndef MATLAPACK_H
#define MATLAPACK_H

#include "lapack_wrapper.h"
#include "mat.h"

template<unsigned int M, unsigned int N, class T>
void invert(Mat<M,N,T>& matrix) {
	assert(M==N);
    static int ipiv[M];
	static T work[M];
	int lwork = M;
	int info;
    
	LAPACK::invert_general_matrix(M, matrix.a, M, ipiv, info);
}


template<unsigned int M, unsigned int N, class T>
void least_squares(Mat<M,N,T>&matrix, Vec<M,T>&rhs) {
    int info = 0;
    LAPACK::solve_least_squares('N', M, N, 1, matrix.a, M, rhs.v, max(M,N), info);
}
#endif
