#ifndef DENSE_MATRIX_H
#define DENSE_MATRIX_H

#include <linear_operator.h>

//============================================================================
struct DenseMatrix: public LinearOperator
{
    std::vector<double> value;
    
    DenseMatrix(int m_=0)
    : LinearOperator(m_, m_), value(m_*m_, 0)
    {}
    
    DenseMatrix(int m_, int n_)
    : LinearOperator(m_, n_), value(m_*n_, 0)
    {}
    
    void clear(void);
    void set_zero(void);
    void resize(int m_, int n_);
    
    const double &operator()(int i, int j) const
    {
        assert(i>=0 && i<m && j>=0 && j<n);
        return value[i+j*m];
    }
    
    double &operator()(int i, int j)
    {
        assert(i>=0 && i<m && j>=0 && j<n);
        return value[i+j*m];
    }
    
    using LinearOperator::apply;
    using LinearOperator::apply_and_subtract;
    using LinearOperator::apply_transpose;
    using LinearOperator::apply_transpose_and_subtract;
    
    virtual void apply(const double *input_vector, double *output_vector) const;
    virtual void apply_and_subtract(const double *x, const double *y, double *z) const;
    virtual void apply_transpose(const double *input_vector, double *output_vector) const;
    virtual void apply_transpose_and_subtract(const double *x, const double *y, double *z) const;
    virtual void write_matlab(std::ostream &output, const char *variable_name) const;
};

void transpose(const DenseMatrix &A, DenseMatrix &Atranspose);
void multiply(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &C);
void multiply_with_transpose(const DenseMatrix &A, DenseMatrix &ATA);

#endif
