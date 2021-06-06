#ifndef LAPACK_WRAPPER_H
#define LAPACK_WRAPPER_H

// Simplified LAPACK wrapper (overloaded readable names, standard across platforms)
// as well as versions with stride==1 assumed.
// For the moment, no complex number support, and most routines have been dropped.

#include <cmath>
#include <iostream>

//////////////
// includes necessary to use Accelerate framework with GCC on OS Yosemite+ (http://stackoverflow.com/questions/26527077/compiling-with-accelerate-framework-on-osx-yosemite ). Thanks to Herve Turlier for pointing out.
#ifndef __has_extension
#define __has_extension(x) 0
#endif

#define vImage_Utilities_h
#define vImage_CVUtilities_h
//////////////

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
//#include <clapack.h>
#endif

using std::max;
using std::min;

namespace LAPACK{
    
    // ---------------------------------------------------------
    //  Function declarations
    // ---------------------------------------------------------
    
    int solve_general_system(int &n, int &nrhs, float *a, int lda, int *ipiv, float *b, int ldb, int &info);
    int solve_general_system(int n, int nrhs, double *a, int lda, int *ipiv, double *b, int ldb, int &info);
    int factor_general_matrix(int m, int n, double *a, int lda, int *ipiv, int &info);
    void invert_general_matrix(int n, double *a, int lda, int *ipiv, double *work, int lwork, int &info);
    void get_eigen_decomposition( int *n, double *a, int* /*lda*/, double *eigenvalues, double *work, int *lwork, int *info );
    int svd( int* m, int* n, double* a, int* lda, double *s, double *u, int* ldu, double* vt, int *ldvt, double *work, int* lwork, int* iwork, int* info );
    int least_squares_svd( int* m, int* n, int* nrhs,  double* a, int* lda,  double* rhs_sol, int* ldb,  double* s, double* rcond, int* rank, double* work, int* lwork, int* iwork, int *info );
    void simple_least_squares_svd( int m, int n, int nrhs, double* a, int lda, double* rhs_sol, int& info, double rcond, int& rank );
    void solve_least_squares(char trans, int m, int n, int nrhs, double*a, int lda, double*b, int ldb, int& info, double rcond, int& rank);
    
    // ---------------------------------------------------------
    //  Inline function definitions --- common for all platforms
    // ---------------------------------------------------------
    
    inline void simple_least_squares_svd( int m, int n, int nrhs, double* a, int lda, double* rhs_sol, int& info, double rcond, int& rank )
    {
        int lapack_m = m;
        int lapack_n = n;
        int lapack_nrhs = nrhs;
        int lapack_lda = lda;   
        int lapack_ldb = max( lapack_m, lapack_n );
        double* s = new double[ min(lapack_m, lapack_n) ];
        double optimal_work_size;
        int lwork = -1;
        int nlvl = 26;
        int liwork = 2*min(lapack_m,lapack_n)*nlvl + 11*min(lapack_m,lapack_n);
        int *iwork = new int[liwork];
        
        // query for optimal work size
        unsigned int lapack_status = LAPACK::least_squares_svd( &lapack_m, 
                                                               &lapack_n, 
                                                               &lapack_nrhs, 
                                                               a, 
                                                               &lapack_lda, 
                                                               rhs_sol, 
                                                               &lapack_ldb, 
                                                               s, 
                                                               &rcond, 
                                                               &rank, 
                                                               &optimal_work_size, 
                                                               &lwork, 
                                                               iwork, 
                                                               &info );
        
        
        if ( info != 0 ) 
        {
            std::cout << "lapack info: " << info << "; lapack_status: " << lapack_status << std::endl;
        }
        
        assert( info == 0 );
        
        lwork = (int)ceil(optimal_work_size);
        
        double *work = new double[lwork];
        
        lapack_status = LAPACK::least_squares_svd( &lapack_m, 
                                                  &lapack_n, 
                                                  &lapack_nrhs, 
                                                  a, 
                                                  &lapack_lda, 
                                                  rhs_sol, 
                                                  &lapack_ldb, 
                                                  s, 
                                                  &rcond, 
                                                  &rank, 
                                                  work, 
                                                  &lwork, 
                                                  iwork, 
                                                  &info );
        
        assert( info == 0 );
        
        delete[] s;
        delete[] iwork;
        delete[] work;
        
    }
    
    inline void solve_least_squares(char, int m, int n, int nrhs, double*a, int lda, double*b, int, int& info, double rcond, int& rank ) 
    {
        simple_least_squares_svd( m, n, nrhs, a, lda, b, info, rcond, rank );
    }
    
    // end of platform-independent code
    
    // ---------------------------------------------------------
    //  Inline function definitions --- Apple
    // ---------------------------------------------------------
    
#ifdef __APPLE__
    
    inline int solve_general_system(int &n, int &nrhs, float *a, int lda, int *ipiv, float *b, int ldb, int &info)
    { return sgesv_( (__CLPK_integer*) &n, 
                    (__CLPK_integer*) &nrhs, 
                    a, 
                    (__CLPK_integer*) &lda, 
                    (__CLPK_integer*) ipiv, 
                    b, 
                    (__CLPK_integer*) &ldb, 
                    (__CLPK_integer*) &info); }
    
    inline int solve_general_system(int n, int nrhs, double *a, int lda, int *ipiv, double *b, int ldb, int &info)
    { return dgesv_( (__CLPK_integer*) &n, 
                    (__CLPK_integer*) &nrhs, 
                    a, 
                    (__CLPK_integer*) &lda, 
                    (__CLPK_integer*) ipiv, 
                    b, 
                    (__CLPK_integer*) &ldb, 
                    (__CLPK_integer*) &info); }
    
    
    inline int factor_general_matrix(int m, int n, double *a, int lda, int *ipiv, int &info)
    { return dgetrf_( (__CLPK_integer*)&m, (__CLPK_integer*)&n, a, (__CLPK_integer*)&lda, (__CLPK_integer*)ipiv, (__CLPK_integer*)&info); }
    
    
    inline void invert_general_matrix(int n, double *a, int lda, int *ipiv, double *work, int lwork, int &info)
    {
        factor_general_matrix( n, n, a, lda, ipiv, info);
        assert( info == 0 );
        dgetri_( (__CLPK_integer*)&n, a, (__CLPK_integer*)&lda, (__CLPK_integer*)ipiv, work, (__CLPK_integer*)&lwork, (__CLPK_integer*)&info);  
    }
    
    inline void get_eigen_decomposition( int *n, double *a, int* /*lda*/, double *eigenvalues, double *work, int *lwork, int *info )
    {
        static char char_v = 'V';
        static char char_l = 'L';
        dsyev_( &char_v, &char_l, (__CLPK_integer*)n, a, (__CLPK_integer*)n, eigenvalues, work, (__CLPK_integer*)lwork, (__CLPK_integer*)info );      
    }
    
    
    inline int svd( int* m, int* n, double* a, int* lda, double *s, double *u, int* ldu, double* vt, int *ldvt, double *work, int* lwork, int* iwork, int* info )
    {
        char char_a = 'A';
        return dgesdd_( &char_a, (__CLPK_integer*)m, (__CLPK_integer*)n, a, (__CLPK_integer*)lda, s, u, (__CLPK_integer*)ldu, vt, (__CLPK_integer*)ldvt, work, (__CLPK_integer*)lwork, (__CLPK_integer*)iwork, (__CLPK_integer*)info ); 
    }
    
    inline int least_squares_svd( int* m, int* n, int* nrhs,  double* a, int* lda,  double* rhs_sol, int* ldb,  double* s, double* rcond, int* rank, double* work, int* lwork, int* iwork, int *info )
    {
        return dgelsd_(  (__CLPK_integer*)m,  (__CLPK_integer*)n, (__CLPK_integer*)nrhs, a, (__CLPK_integer*)lda, rhs_sol, (__CLPK_integer*)ldb, s, rcond, (__CLPK_integer*) rank, work, (__CLPK_integer*) lwork, (__CLPK_integer*) iwork , (__CLPK_integer*) info );
    }
    
#endif // end of __APPLE__ version of this file
    
    
    // ---------------------------------------------------------
    //  Inline function definitions --- Linux and Windows
    // ---------------------------------------------------------
    
#ifndef __APPLE__ 
    
}     // namespace LAPACK

extern "C" {
    int   sgesv_(const int *N, const int *nrhs, float *A, const int *lda, int *ipiv, float *b, const int *ldb, int *info);   
    int 	dgesv_ (int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b, int *ldb, int *info);
    int 	dgeev_ (char *jobvl, char *jobvr, int *n, double *a, int *lda, double *wr, double *wi, double *vl, int *ldvl, double *vr, int *ldvr, double *work, int *lwork, int *info);
    int 	dsygv_ (int *itype, char *jobz, char *uplo, int *n, double *a, int *lda, double *b, int *ldb, double *w, double *work, int *lwork, int *info);
    int 	dgelss_ (int *m, int *n, int *nrhs, double *a, int *lda, double *b, int *ldb, double *s, double *rcond, int *rank, double *work, int *lwork, int *info);
    int 	dgelsd_ (int *m, int *n, int *nrhs, double *a, int *lda, double *b, int *ldb, double *s, double *rcond, int *rank, double *work, int *lwork, int *iwork, int *info);
    int 	dsyev_ (char *jobz, char *uplo, int *n, double *fa, int *lda, double *w, double *work, int *lwork, int *info);
    int 	dsysv_ (char *uplo, int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b, int *ldb, double *work, int *lwork, int *info);
    int 	dgetrf_ (int *m, int *n, double *a, int *lda, int *ipiv, int *info);
    int 	dgetri_ (int *n, double *a, int *lda, int *ipiv, int *info);
    void 	dgebrd_ (int *m, int *n, double *a, int *lda, double *d, double *e, double *tauq, double *taup, double *work, int *lwork, int *info);
    void 	dorgbr_ (char *vect, int *m, int *n, int *k, double *a, int *lda, double *tau, double *work, int *lwork, int *info);
    void 	dbdsqr_ (char *uplo, int *n, int *ncvt, int *nru, int *ncc, double *d, double *e, double *vt, int *ldvt, double *u, int *ldu, double *c, int *ldc, double *work, int *info);
    void 	dgetrs_ (char *trans, int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b, int *ldb, int *info);
    void 	dpotrf_ (char *uplo, int *n, double *a, int *lda, int *info);
    void 	dgeqpf_ (int *m, int *n, double *a, int *lda, int *jpvt, double *tau, double *work, int *info);   
    int dgesdd_( char* jobz, int* m, int* n, double* a, int* lda, double *s, double *u, int* ldu, double* vt, int *ldvt, double *work, int* lwork, int* iwork, int* info );
}

namespace LAPACK
{
    
    inline void solve_general_system(int n, int nrhs, float *a, int lda, int *ipiv, float *b, int ldb, int &info)
    { 
        sgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info); 
    }
    
    inline int solve_general_system(int n, int nrhs, double *a, int lda, int *ipiv, double *b, int ldb, int &info)
    { 
        return dgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info); 
    }
    
    inline int factor_general_matrix(int m, int n, double *a, int lda, int *ipiv, int &info)
    { return dgetrf_(&m, &n, a, &lda, ipiv, &info); }
    
    inline void invert_general_matrix(int n, double *a, int lda, int *ipiv, double* /*work*/, int /*lwork*/, int &info)
    {
        factor_general_matrix(n, n, a, lda, ipiv, info);
        dgetri_(&n, a, &lda, ipiv, &info);  
    }
    
    inline void get_eigen_decomposition( int *n, double *a, int* /*lda*/, double *eigenvalues, double *work, int *lwork, int *info )
    {
        static char char_v = 'V';
        static char char_l = 'L';
        dsyev_( &char_v, &char_l, n, a, n, eigenvalues, work, lwork, info );      
    }
    
    inline int svd( int* m, int* n, double* a, int* lda, double *s, double *u, int* ldu, double* vt, int *ldvt, double *work, int* lwork, int* iwork, int* info )
    {
        char char_a = 'A';
        return dgesdd_( &char_a, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info ); 
    }
    
    inline int least_squares_svd( int* m, int* n, int* nrhs,  double* a, int* lda,  double* rhs_sol, int* ldb,  double* s, double* rcond, int* rank, double* work, int* lwork, int* iwork, int *info )
    {
        return dgelsd_( m,  n, nrhs, a, lda, rhs_sol, ldb, s, rcond, rank, work, lwork, iwork, info );
    }   
    
    
#endif //End of Windows and Linux version of this file
    
} // end of namespace LAPACK

#endif
