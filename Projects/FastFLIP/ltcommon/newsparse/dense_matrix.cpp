#include <dense_matrix.h>
#include <blas_wrapper.h>
#include <cstring>

void DenseMatrix::
clear(void)
{
    m=n=0;
    value.clear();
}

void DenseMatrix::
set_zero(void)
{
    BLAS::set_zero(value);
}

void DenseMatrix::
resize(int m_, int n_)
{
    m=m_; n=n_;
    value.resize(m*n, 0);
}

void DenseMatrix::
apply(const double *x, double *y) const
{
    assert(x && y);
    BLAS::multiply_matrix_vector(m, n, &value[0], x, y);
}

void DenseMatrix::
apply_and_subtract(const double *x, const double *y, double *z) const
{
    assert(x && y);
    if(y!=z) BLAS::copy(m, y, z);
    BLAS::multiply_matrix_vector(BLAS::NoTrans, m, n, -1, &value[0], m, x, 1, 1, z);
}

void DenseMatrix::
apply_transpose(const double *x, double *y) const
{
    assert(x && y);
    BLAS::multiply_matrix_vector(BLAS::Trans, m, n, 1, &value[0], m, x, 1, 0, y);
}

void DenseMatrix::
apply_transpose_and_subtract(const double *x, const double *y, double *z) const
{
    assert(x && y);
    if(y!=z) BLAS::copy(n, y, z);
    BLAS::multiply_matrix_vector(BLAS::Trans, m, n, -1, &value[0], m, x, 1, 1, z);
}

void DenseMatrix::
write_matlab(std::ostream &output, const char *variable_name) const
{
    output<<variable_name<<"=[";
    std::streamsize old_precision=output.precision();
    output.precision(18);
    for(int i=0; i<m; ++i){
        if(i>0) output<<" ";
        for(int j=0; j<n-1; ++j) output<<value[i+j*m]<<" ";
        output<<value[i+(n-1)*m];
        if(i<m-1) output<<std::endl;
        else      output<<"];"<<std::endl;
    }
    output.precision(old_precision);
}

void transpose(const DenseMatrix &A, DenseMatrix &Atranspose)
{
    Atranspose.resize(A.n, A.m);
    for(int j=0; j<A.n; ++j) for(int i=0; i<A.m; ++i){
        Atranspose(j,i)=A(i,j);
    }
}

void multiply(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &C)
{
    assert(A.n==B.m);
    C.resize(A.m, B.n);
    BLAS::multiply_matrix_matrix(A.m, A.n, B.m, &A.value[0], &B.value[0], &C.value[0]);
}

void multiply_with_transpose(const DenseMatrix &A, DenseMatrix &ATA)
{
    ATA.resize(A.n, A.n);
    BLAS::multiply_matrix_matrix(BLAS::Trans, BLAS::NoTrans, A.n, A.n, A.m, 1, &A.value[0], A.m, &A.value[0], A.m,
                                 0, &ATA.value[0], A.n);
}

