#ifndef BLAS_WRAPPER_H
#define BLAS_WRAPPER_H

#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

#define USE_CBLAS 1

// Useful dense kernels from BLAS, with readable, overloaded, cross-platform names and some simplified calling
//  dot    (dot-product of vectors)
//  nrm2   (2-norm of a vector)
//  asum   (1-norm of a vector)
//  amax   (index of maximum absolute value in a vector)
//  swap   (exchanging values in two vectors)
//  copy   (copying values from one vector to another)
//  axpy   (adding a scalar times a vector to another vector)
//  scal   (multiplying a vector by a scalar)
//  gemv   (multiplying a matrix times a vector, scaling, and adding result to another vector))
//  gemm   (multiplying two matrices, scaling, and adding result to another matrix)
// In addition:
//  set_zero  (zero out all entries in a vector)
//  abs_max   (return the infinity norm of a vector, i.e. the magnitude of its largest element)
// There are also version using std::vector for convenience.

// Matrices are always assumed to be in column-major format.

// You can #define one of:
//    USE_FORTRAN_BLAS (if your BLAS calls should look like dgemm_ with FORTRAN calling conventions, as in GOTO BLAS)
//    USE_AMD_BLAS     (if using the AMD Math Library)
//    USE_CBLAS        (if instead you have calls like cblas_dgemm, and have a file "cblas.h" available)
// or, if you're on the Mac, it will default to the vecLib CBLAS if none of these are specified.

namespace BLAS{
    
    template<class T>
    inline void set_zero(int n, T *x)
    { std::memset(x, 0, n*sizeof(T)); }
    
}

//============================================================================
#ifdef USE_FORTRAN_BLAS

extern "C" {
    double dsdot_(const int*, const float*, const int*, const float*, const int*);
    double sdot_(const int*, const float*, const int*, const float*, const int*);
    double ddot_(const int*, const double*, const int*, const double*, const int*);
    float snrm2_(const int*, const float*, const int*);
    double dnrm2_(const int*, const double*, const int*);
    float sasum_(const int*, const float*, const int*);
    double dasum_(const int*, const double*, const int*);
    int isamax_(const int*, const float*, const int*);
    int idamax_(const int*, const double*, const int*);
    void sswap_(const int*, float*, const int*, float*, const int*);
    void dswap_(const int*, double*, const int*, double*, const int*);
    void scopy_(const int*, const float*, const int*, float*, const int*);
    void dcopy_(const int*, const double*, const int*, double*, const int*);
    void saxpy_(const int*, const float*, const float*, const int*, float*, const int*);
    void daxpy_(const int*, const double*, const double*, const int*, double*, const int*);
    void sscal_(const int*, const float*, float*, const int*);
    void dscal_(const int*, const double*, double*, const int*);
    void sgemv_(const char*, const int*, const int*, const float*, const float*, const int*, const float*, const int*, const float*, float*, const int*);
    void dgemv_(const char*, const int*, const int*, const double*, const double*, const int*, const double*, const int*, const double*, double*, const int*);
    void sgemm_(const char*, const char*, const int*, const int*, const int*, const float*, const float*, const int*, const float*, const int*, const float*, float*, const int*);
    void dgemm_(const char*, const char*, const int*, const int*, const int*, const double*, const double*, const int*, const double*, const int*, const double*, double*, const int*);
}

namespace BLAS{
    
    enum Transpose {NoTrans='N', Trans='T'};
    enum UpperLower {Upper='U', Lower='L'};
    enum UnitDiag {NonUnit='N', Unit='U'};
    enum Side {Left='L', Right='R'};
    
    // dot products
    
    inline double dot(int n, const float *x, int incx, const float *y, int incy=1)
    { return dsdot_(&n, x, &incx, y, &incy); }
    
    inline double dot(int n, const float *x, const float *y, int incy=1)
    { const int one=1; return dsdot_(&n, x, &one, y, &incy); }
    
    inline float dotf(int n, const float *x, int incx, const float *y, int incy=1)
    { return (float)sdot_(&n, x, &incx, y, &incy); }
    
    inline float dotf(int n, const float *x, const float *y, int incy=1)
    { const int one=1; return (float)sdot_(&n, x, &one, y, &incy); }
    
    inline double dot(int n, const double *x, int incx, const double *y, int incy=1)
    { return ddot_(&n, x, &incx, y, &incy); }
    
    inline double dot(int n, const double *x, const double *y, int incy=1)
    { const int one=1; return ddot_(&n, x, &one, y, &incy); }
    
    // 2-norm 
    
    inline float norm2(int n, const float *x, int incx=1)
    { return snrm2_(&n, x, &incx); }
    
    inline double norm2(int n, const double *x, int incx=1)
    { return dnrm2_(&n, x, &incx); }
    
    // 1-norm (sum of absolute values)
    
    inline float abs_sum(int n, const float *x, int incx=1)
    { return sasum_(&n, x, &incx); }
    
    inline double abs_sum(int n, const double *x, int incx=1)
    { return dasum_(&n, x, &incx); }
    
    // inf-norm (maximum absolute value: index of max returned)
    
    inline int index_abs_max(int n, const float *x, int incx=1)
    { return isamax_(&n, x, &incx)-1; }
    
    inline int index_abs_max(int n, const double *x, int incx=1)
    { return idamax_(&n, x, &incx)-1; }
    
    inline float abs_max(int n, const float *x, int incx=1)
    { return std::fabs(x[isamax_(&n, x, &incx)-1]); }
    
    inline double abs_max(int n, const double *x, int incx=1)
    { return std::fabs(x[idamax_(&n, x, &incx)-1]); }
    
    // swap (actual data exchanged, not just pointers)
    
    inline void swap(int n, float *x, int incx, float *y, int incy=1)
    { sswap_(&n, x, &incx, y, &incy); }
    
    inline void swap(int n, float *x, float *y, int incy=1)
    { const int one=1; sswap_(&n, x, &one, y, &incy); }
    
    inline void swap(int n, double *x, int incx, double *y, int incy=1)
    { dswap_(&n, x, &incx, y, &incy); }
    
    inline void swap(int n, double *x, double *y, int incy=1)
    { const int one=1; dswap_(&n, x, &one, y, &incy); }
    
    // copy (y=x)
    
    inline void copy(int n, const float *x, int incx, float *y, int incy=1)
    { scopy_(&n, x, &incx, y, &incy); }
    
    inline void copy(int n, const float *x, float *y, int incy=1)
    { const int one=1; scopy_(&n, x, &one, y, &incy); }
    
    inline void copy(int n, const double *x, int incx, double *y, int incy=1)
    { dcopy_(&n, x, &incx, y, &incy); }
    
    inline void copy(int n, const double *x, double *y, int incy=1)
    { const int one=1; dcopy_(&n, x, &one, y, &incy); }
    
    // saxpy (y=alpha*x+y)
    
    inline void add_scaled(int n, float alpha, const float *x, int incx, float *y, int incy=1)
    { saxpy_(&n, &alpha, x, &incx, y, &incy); }
    
    inline void add_scaled(int n, float alpha, const float *x, float *y, int incy=1)
    { const int one=1; saxpy_(&n, &alpha, x, &one, y, &incy); }
    
    inline void add_scaled(int n, double alpha, const double *x, int incx, double *y, int incy=1)
    { daxpy_(&n, &alpha, x, &incx, y, &incy); }
    
    inline void add_scaled(int n, double alpha, const double *x, double *y, int incy=1)
    { const int one=1; daxpy_(&n, &alpha, x, &one, y, &incy); }
    
    // scale (x=alpha*x)
    
    inline void scale(int n, float alpha, float *x, int incx=1)
    { sscal_(&n, &alpha, x, &incx); }
    
    inline void scale(int n, double alpha, double *x, int incx=1)
    { dscal_(&n, &alpha, x, &incx); }
    
    // gemv (y=alpha*A*x+beta*y, or using A^T)
    // The matrix is always m*n; the size of x and y depend on if A is transposed.
    
    inline void multiply_matrix_vector(Transpose transpose,
                                       int m, int n, float alpha, const float *A, int lda,
                                       const float *x, int incx, float beta, float *y, int incy=1)
    { sgemv_((const char*)&transpose, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy); }
    
    inline void multiply_matrix_vector(int m, int n, const float *A, const float *x, float *y, int incy=1) // y=A*x
    { const int onei=1; const float zero=0, onef=1; sgemv_("N", &m, &n, &onef, A, &m, x, &onei, &zero, y, &incy); }
    
    inline void multiply_matrix_vector(Transpose transpose,
                                       int m, int n, double alpha, const double *A, int lda,
                                       const double *x, int incx, double beta, double *y, int incy=1)
    { dgemv_((const char*)&transpose, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy); }
    
    inline void multiply_matrix_vector(int m, int n, const double *A, const double *x, double *y, int incy=1) // y=A*x
    { const int onei=1; const double zero=0, onef=1; dgemv_("N", &m, &n, &onef, A, &m, x, &onei, &zero, y, &incy); }
    
    // gemm (C=alpha*A*B+beta*C)
    
    inline void multiply_matrix_matrix(Transpose transA, Transpose transB,
                                       int m, int n, int k, float alpha, const float *A, int lda,
                                       const float *B, int ldb, float beta, float *C, int ldc)
    { sgemm_((const char*)&transA, (const char*)&transB, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc); }
    
    inline void multiply_matrix_matrix(int m, int n, int k, const float *A, const float *B, float *C) 
    { const float zero=0, one=1; sgemm_("N", "N", &m, &n, &k, &one, A, &m, B, &k, &zero, C, &m); } // C=A*B
    
    inline void multiply_matrix_matrix(Transpose transA, Transpose transB,
                                       int m, int n, int k, double alpha, const double *A, int lda,
                                       const double *B, int ldb, double beta, double *C, int ldc)
    { dgemm_((const char*)&transA, (const char*)&transB, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc); }
    
    inline void multiply_matrix_matrix(int m, int n, int k, const double *A, const double *B, double *C) 
    { const double zero=0, one=1; dgemm_("N", "N", &m, &n, &k, &one, A, &m, B, &k, &zero, C, &m); } // C=A*B
    
}

//============================================================================
#elif defined USE_AMD_BLAS

#include <acml.h>

namespace BLAS{
    
    enum Transpose {NoTrans='N', Trans='T'};
    enum UpperLower {Upper='U', Lower='L'};
    enum UnitDiag {NonUnit='N', Unit='U'};
    enum Side {Left='L', Right='R'};
    
    // dot products
    
    inline double dot(int n, const float *x, int incx, const float *y, int incy=1)
    { return dsdot(n, (float*)x, incx, (float*)y, incy); }
    
    inline double dot(int n, const float *x, const float *y, int incy=1)
    { return dsdot(n, (float*)x, 1, (float*)y, incy); }
    
    inline float dotf(int n, const float *x, int incx, const float *y, int incy=1)
    { return sdot(n, (float*)x, incx, (float*)y, incy); }
    
    inline float dotf(int n, const float *x, const float *y, int incy=1)
    { return sdot(n, (float*)x, 1, (float*)y, incy); }
    
    inline double dot(int n, const double *x, int incx, const double *y, int incy=1)
    { return ddot(n, (double *)x, incx, (double *)y, incy); }
    
    inline double dot(int n, const double *x, const double *y, int incy=1)
    { return ddot(n, (double *)x, 1, (double *)y, incy); }
    
    // 2-norm 
    
    inline float norm2(int n, const float *x, int incx=1)
    { return snrm2(n, (float*)x, incx); }
    
    inline double norm2(int n, const double *x, int incx=1)
    { return dnrm2(n, (double*)x, incx); }
    
    // 1-norm (sum of absolute values)
    
    inline float abs_sum(int n, const float *x, int incx=1)
    { return sasum(n, (float*)x, incx); }
    
    inline double abs_sum(int n, const double *x, int incx=1)
    { return dasum(n, (double*)x, incx); }
    
    // inf-norm (maximum absolute value: index of max returned)
    
    inline int index_abs_max(int n, const float *x, int incx=1)
    { return isamax(n, (float*)x, incx)-1; }
    
    inline int index_abs_max(int n, const double *x, int incx=1)
    { return idamax(n, (double*)x, incx)-1; }
    
    inline float abs_max(int n, const float *x, int incx=1)
    { return std::fabs(x[isamax(n, (float*)x, incx)]-1); }
    
    inline double abs_max(int n, const double *x, int incx=1)
    { return std::fabs(x[idamax(n, (double*)x, incx)]-1); }
    
    // swap (actual data exchanged, not just pointers)
    
    inline void swap(int n, float *x, int incx, float *y, int incy=1)
    { sswap(n, x, incx, y, incy); }
    
    inline void swap(int n, float *x, float *y, int incy=1)
    { sswap(n, x, 1, y, incy); }
    
    inline void swap(int n, double *x, int incx, double *y, int incy=1)
    { dswap(n, x, incx, y, incy); }
    
    inline void swap(int n, double *x, double *y, int incy=1)
    { dswap(n, x, 1, y, incy); }
    
    // copy (y=x)
    
    inline void copy(int n, const float *x, int incx, float *y, int incy=1)
    { scopy(n, (float*)x, incx, y, incy); }
    
    inline void copy(int n, const float *x, float *y, int incy=1)
    { scopy(n, (float*)x, 1, y, incy); }
    
    inline void copy(int n, const double *x, int incx, double *y, int incy=1)
    { dcopy(n, (double *)x, incx, y, incy); }
    
    inline void copy(int n, const double *x, double *y, int incy=1)
    { dcopy(n, (double *)x, 1, y, incy); }
    
    // saxpy (y=alpha*x+y)
    
    inline void add_scaled(int n, float alpha, const float *x, int incx, float *y, int incy=1)
    { saxpy(n, alpha, (float*)x, incx, y, incy); }
    
    inline void add_scaled(int n, float alpha, const float *x, float *y, int incy=1)
    { saxpy(n, alpha, (float*)x, 1, y, incy); }
    
    inline void add_scaled(int n, double alpha, const double *x, int incx, double *y, int incy=1)
    { daxpy(n, alpha, (double*)x, incx, y, incy); }
    
    inline void add_scaled(int n, double alpha, const double *x, double *y, int incy=1)
    { daxpy(n, alpha, (double*)x, 1, y, incy); }
    
    // scale (x=alpha*x)
    
    inline void scale(int n, float alpha, float *x, int incx=1)
    { sscal(n, alpha, x, incx); }
    
    inline void scale(int n, double alpha, double *x, int incx=1)
    { dscal(n, alpha, x, incx); }
    
    // gemv (y=alpha*A*x+beta*y, or using A^T)
    // The matrix is always m*n; the size of x and y depend on if A is transposed.
    
    inline void multiply_matrix_vector(Transpose transpose,
                                       int m, int n, float alpha, const float *A, int lda,
                                       const float *x, int incx, float beta, float *y, int incy=1)
    { sgemv(transpose, m, n, alpha, (float*)A, lda, (float*)x, incx, beta, y, incy); }
    
    inline void multiply_matrix_vector(int m, int n, const float *A, const float *x, float *y, int incy=1) // y=A*x
    { sgemv(NoTrans, m, n, 1.f, (float*)A, m, (float*)x, 1, 0.f, y, incy); }
    
    inline void multiply_matrix_vector(Transpose transpose,
                                       int m, int n, double alpha, const double *A, int lda,
                                       const double *x, int incx, double beta, double *y, int incy=1)
    { dgemv(transpose, m, n, alpha, (double*)A, lda, (double*)x, incx, beta, y, incy); }
    
    inline void multiply_matrix_vector(int m, int n, const double *A, const double *x, double *y, int incy=1) // y=A*x
    { dgemv(NoTrans, m, n, 1., (double*)A, m, (double*)x, 1, 0., y, incy); }
    
    // gemm (C=alpha*A*B+beta*C)
    
    inline void multiply_matrix_matrix(Transpose transA, Transpose transB,
                                       int m, int n, int k, float alpha, const float *A, int lda,
                                       const float *B, int ldb, float beta, float *C, int ldc)
    { sgemm(transA, transB, m, n, k, alpha, (float*)A, lda, (float*)B, ldb, beta, C, ldc); }
    
    inline void multiply_matrix_matrix(int m, int n, int k, const float *A, const float *B, float *C) 
    { sgemm(NoTrans, NoTrans, m, n, k, 1.f, (float*)A, m, (float*)B, k, 0.f, C, m); } // C=A*B
    
    inline void multiply_matrix_matrix(Transpose transA, Transpose transB,
                                       int m, int n, int k, double alpha, const double *A, int lda,
                                       const double *B, int ldb, double beta, double *C, int ldc)
    { dgemm(transA, transB, m, n, k, alpha, (double*)A, lda, (double*)B, ldb, beta, C, ldc); }
    
    inline void multiply_matrix_matrix(int m, int n, int k, const double *A, const double *B, double *C) 
    { dgemm(NoTrans, NoTrans, m, n, k, 1., (double*)A, m, (double*)B, k, 0., C, m); } // C=A*B
    
};

//============================================================================
#elif defined USE_CBLAS || defined __APPLE__

#ifdef USE_CBLAS
#include <cblas.h>
#elif defined __APPLE__
#include <vecLib/cblas.h>
#endif

namespace BLAS{
    
    enum Transpose {NoTrans=CblasNoTrans, Trans=CblasTrans};
    enum UpperLower {Upper=CblasUpper, Lower=CblasLower};
    enum UnitDiag {NonUnit=CblasNonUnit, Unit=CblasUnit};
    enum Side {Left=CblasLeft, Right=CblasRight};
    
    // dot products
    
    inline float dotf(int n, const float *x, int incx, const float *y, int incy=1)
    { return cblas_sdot(n, x, incx, y, incy); }
    
    inline float dotf(int n, const float *x, const float *y, int incy=1)
    { return cblas_sdot(n, x, 1, y, incy); }
    
    inline double dot(int n, const float *x, int incx, const float *y, int incy=1)
    { return cblas_dsdot(n, x, incx, y, incy); }
    
    inline double dot(int n, const float *x, const float *y, int incy=1)
    { return cblas_dsdot(n, x, 1, y, incy); }
    
    inline double dot(int n, const double *x, int incx, const double *y, int incy=1)
    { return cblas_ddot(n, x, incx, y, incy); }
    
    inline double dot(int n, const double *x, const double *y, int incy=1)
    { return cblas_ddot(n, x, 1, y, incy); }
    
    // 2-norm
    
    inline float norm2(int n, const float *x, int incx=1)
    { return cblas_snrm2(n, x, incx); }
    
    inline double norm2(int n, const double *x, int incx=1)
    { return cblas_dnrm2(n, x, incx); }
    
    // 1-norm (sum of absolute values)
    
    inline float abs_sum(int n, const float *x, int incx=1)
    { return cblas_sasum(n, x, incx); }
    
    inline double abs_sum(int n, const double *x, int incx=1)
    { return cblas_dasum(n, x, incx); }
    
    // inf-norm (maximum absolute value)
    
    inline int index_abs_max(int n, const float *x, int incx=1)
    { return cblas_isamax(n, x, incx); }
    
    inline int index_abs_max(int n, const double *x, int incx=1)
    { return cblas_idamax(n, x, incx); }
    
    inline float abs_max(int n, const float *x, int incx=1)
    { return std::fabs(x[cblas_isamax(n, x, incx)]); }
    
    inline double abs_max(int n, const double *x, int incx=1)
    { return std::fabs(x[cblas_idamax(n, x, incx)]); }
    
    // swap (actual data exchanged, not just pointers)
    
    inline void swap(int n, float *x, int incx, float *y, int incy=1)
    { cblas_sswap(n, x, incx, y, incy); }
    
    inline void swap(int n, float *x, float *y, int incy=1)
    { cblas_sswap(n, x, 1, y, incy); }
    
    inline void swap(int n, double *x, int incx, double *y, int incy=1)
    { cblas_dswap(n, x, incx, y, incy); }
    
    inline void swap(int n, double *x, double *y, int incy=1)
    { cblas_dswap(n, x, 1, y, incy); }
    
    // copy (y=x)
    
    inline void copy(int n, const float *x, int incx, float *y, int incy=1)
    { cblas_scopy(n, x, incx, y, incy); }
    
    inline void copy(int n, const float *x, float *y, int incy=1)
    { cblas_scopy(n, x, 1, y, incy); }
    
    inline void copy(int n, const double *x, int incx, double *y, int incy=1)
    { cblas_dcopy(n, x, incx, y, incy); }
    
    inline void copy(int n, const double *x, double *y, int incy=1)
    { cblas_dcopy(n, x, 1, y, incy); }
    
    // saxpy (y=alpha*x+y)
    
    inline void add_scaled(int n, float alpha, const float *x, int incx, float *y, int incy=1)
    { cblas_saxpy(n, alpha, x, incx, y, incy); }
    
    inline void add_scaled(int n, float alpha, const float *x, float *y, int incy=1)
    { cblas_saxpy(n, alpha, x, 1, y, incy); }
    
    inline void add_scaled(int n, double alpha, const double *x, int incx, double *y, int incy=1)
    { cblas_daxpy(n, alpha, x, incx, y, incy); }
    
    inline void add_scaled(int n, double alpha, const double *x, double *y, int incy=1)
    { cblas_daxpy(n, alpha, x, 1, y, incy); }
    
    // scale (x=alpha*x)
    
    inline void scale(int n, float alpha, float *x, int incx=1)
    { cblas_sscal(n, alpha, x, incx); }
    
    inline void scale(int n, double alpha, double *x, int incx=1)
    { cblas_dscal(n, alpha, x, incx); }
    
    // gemv (y=alpha*A*x+beta*y, or using A^T)
    // The matrix is always m*n; the size of x and y depend on if A is transposed.
    
    inline void multiply_matrix_vector(Transpose transpose,
                                       int m, int n, float alpha, const float *A, int lda,
                                       const float *x, int incx,
                                       float beta, float *y, int incy=1)
    { cblas_sgemv(CblasColMajor, (CBLAS_TRANSPOSE)transpose, m, n, alpha, A, lda, x, incx, beta, y, incy); }
    
    inline void multiply_matrix_vector(int m, int n, const float *A, const float *x, float *y, int incy=1) // y=A*x
    { cblas_sgemv(CblasColMajor, CblasNoTrans, m, n, 1.f, A, m, x, 1, 0.f, y, incy); }
    
    inline void multiply_matrix_vector(Transpose transpose,
                                       int m, int n, double alpha, const double *A, int lda,
                                       const double *x, int incx, double beta, double *y, int incy=1)
    { cblas_dgemv(CblasColMajor, (CBLAS_TRANSPOSE)transpose, m, n, alpha, A, lda, x, incx, beta, y, incy); }
    
    inline void multiply_matrix_vector(int m, int n, const double *A, const double *x, double *y, int incy=1) // y=A*x
    { cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, 1., A, m, x, 1, 0., y, incy); }
    
    // gemm (C=alpha*A*B+beta*C)
    
    inline void multiply_matrix_matrix(Transpose transA, Transpose transB,
                                       int m, int n, int k, float alpha, const float *A, int lda,
                                       const float *B, int ldb, float beta, float *C, int ldc)
    { cblas_sgemm(CblasColMajor, (CBLAS_TRANSPOSE)transA, (CBLAS_TRANSPOSE)transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc); }
    
    inline void multiply_matrix_matrix(int m, int n, int k, const float *A, const float *B, float *C) 
    { cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.f, A, m, B, k, 0.f, C, m); } // C=A*B
    
    inline void multiply_matrix_matrix(Transpose transA, Transpose transB,
                                       int m, int n, int k, double alpha, const double *A, int lda,
                                       const double *B, int ldb, double beta, double *C, int ldc)
    { cblas_dgemm(CblasColMajor, (CBLAS_TRANSPOSE)transA, (CBLAS_TRANSPOSE)transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc); }
    
    inline void multiply_matrix_matrix(int m, int n, int k, const double *A, const double *B, double *C) 
    { cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1., A, m, B, k, 0., C, m); } // C=A*B
    
    inline void rank_one_update( int m, int n, double alpha, const double* x, const double* y, double* A )
    {
        cblas_dger( CblasColMajor, m, n, alpha, x, 1, y, 1, A, m );
    }
    
}  // namespace BLAS


#endif

// std::vector calls =========================================================
namespace BLAS{
    
    template<class T>
    inline void set_zero(std::vector<T> &x)
    { set_zero((int)x.size(), &x[0]); }
    
    inline float dotf(const std::vector<float> &x, const std::vector<float> &y)
    { assert(x.size()==y.size()); return dotf((int)x.size(), &x[0], &y[0]); }
    
    inline double dot(const std::vector<float> &x, const std::vector<float> &y)
    { assert(x.size()==y.size()); return dot((int)x.size(), &x[0], &y[0]); }
    
    inline double dot(const std::vector<double> &x, const std::vector<double> &y)
    { assert(x.size()==y.size()); return dot((int)x.size(), &x[0], &y[0]); }
    
    inline float norm2(const std::vector<float> &x)
    { return norm2((int)x.size(), &x[0]); }
    
    inline double norm2(const std::vector<double> &x)
    { return norm2((int)x.size(), &x[0]); }
    
    inline float abs_sum(const std::vector<float> &x)
    { return abs_sum((int)x.size(), &x[0]); }
    
    inline double abs_sum(const std::vector<double> &x)
    { return abs_sum((int)x.size(), &x[0]); }
    
    inline int index_abs_max(const std::vector<float> &x)
    { return index_abs_max((int)x.size(), &x[0]); }
    
    inline int index_abs_max(const std::vector<double> &x)
    { return index_abs_max((int)x.size(), &x[0]); }
    
    inline float abs_max(const std::vector<float> &x)
    { return abs_max((int)x.size(), &x[0]); }
    
    inline double abs_max(const std::vector<double> &x)
    { return abs_max((int)x.size(), &x[0]); }
    
    inline void swap(std::vector<float> &x, std::vector<float> &y)
    { assert(x.size()==y.size()); swap((int)x.size(), &x[0], &y[0]); }
    
    inline void swap(std::vector<double> &x, std::vector<double> &y)
    { assert(x.size()==y.size()); swap((int)x.size(), &x[0], &y[0]); }
    
    inline void copy(const std::vector<float> &x, std::vector<float> &y)
    { assert(x.size()==y.size()); copy((int)x.size(), &x[0], &y[0]); }
    
    inline void copy(const std::vector<double> &x, std::vector<double> &y)
    { assert(x.size()==y.size()); copy((int)x.size(), &x[0], &y[0]); }
    
    inline void add_scaled(float alpha, const std::vector<float> &x, std::vector<float> &y)
    { assert(x.size()==y.size()); add_scaled((int)x.size(), alpha, &x[0], &y[0]); }
    
    inline void add_scaled(double alpha, const std::vector<double> &x, std::vector<double> &y)
    { assert(x.size()==y.size()); add_scaled((int)x.size(), alpha, &x[0], &y[0]); }
    
    inline void scale(float alpha, std::vector<float> &x)
    { scale((int)x.size(), alpha, &x[0]); }
    
    inline void scale(float alpha, std::vector<double> &x)
    { scale((int)x.size(), alpha, &x[0]); }
    
    // I'm not sure if it makes sense to include level 2 or level 3 std::vector versions,
    // since there isn't an STL matrix type...
    
}  // namespace BLAS

#endif
