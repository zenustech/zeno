#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

// Definitions for a variety of sparse matrix formats.
// By default everything keeps entries in sorted order, with no duplicates.

#include <blas_wrapper.h>
#include <linear_operator.h>
#include <list>

namespace LosTopos {
//============================================================================
// useful for dynamic data structures
struct SparseEntry
{
    int index;
    double value;
    
    SparseEntry(void) : index(~0), value(1e+30) {}
    SparseEntry(int index_, double value_) : index(index_), value(value_) {}
    bool operator<(const SparseEntry &e) const { return index<e.index; }
};

typedef std::list<SparseEntry> DynamicSparseVector;

//============================================================================
// Dynamic Compressed Sparse Row
// This probably should only be used when constructing a new matrix; for
// iteratively solving etc. you probably want to convert it to a more efficienct
// static structure.
struct SparseMatrixDynamicCSR: public LinearOperator
{
    std::vector<DynamicSparseVector> row;
    const double zero; // may need to return a reference to zero for convenience in operator()
    
    SparseMatrixDynamicCSR(int m_=0) : LinearOperator(m_), row(m_), zero(0) {}
    SparseMatrixDynamicCSR(int m_, int n_) : LinearOperator(m_, n_), row(m_), zero(0) {}
    void clear(void);
    void set_zero(void);
    void resize(int m_, int n_); // note: eliminates extra rows, but not extra columns
    const double &operator()(int i, int j) const;
    double &operator()(int i, int j); // will create a new zero entry at (i,j) if one doesn't exist already; this may invalidate other references in row i
    void add_sparse_row(int i, const DynamicSparseVector &x, double multiplier=1);
    using LinearOperator::apply;
    using LinearOperator::apply_and_subtract;
    using LinearOperator::apply_transpose;
    using LinearOperator::apply_transpose_and_subtract;
    virtual void apply(const double *x, double *y) const;
    virtual void apply_and_subtract(const double *x, const double *y, double *z) const;
    virtual void apply_transpose(const double *x, double *y) const;
    virtual void apply_transpose_and_subtract(const double *x, const double *y, double *z) const;
    virtual void write_matlab(std::ostream &output, const char *variable_name) const;
};

//============================================================================
// Static Compressed Sparse Row
struct SparseMatrixStaticCSR: public LinearOperator
{
    std::vector<int> rowstart;
    std::vector<int> colindex;
    std::vector<double> value;
    
    SparseMatrixStaticCSR(int m_=0) : LinearOperator(m_), rowstart(m_+1, 0), colindex(0), value(0) {}
    SparseMatrixStaticCSR(int m_, int n_) : LinearOperator(m_, n_), rowstart(m_+1, 0), colindex(0), value(0) {}
    SparseMatrixStaticCSR(const SparseMatrixDynamicCSR &matrix);
    void clear(void);
    void set_zero(void);
    void resize(int m_, int n_); // note: eliminates extra rows, but not extra columns
    double operator()(int i, int j) const;
    using LinearOperator::apply;
    using LinearOperator::apply_and_subtract;
    using LinearOperator::apply_transpose;
    using LinearOperator::apply_transpose_and_subtract;
    virtual void apply(const double *x, double *y) const;
    virtual void apply_and_subtract(const double *x, const double *y, double *z) const;
    virtual void apply_transpose(const double *x, double *y) const;
    virtual void apply_transpose_and_subtract(const double *x, const double *y, double *z) const;
    virtual void write_matlab(std::ostream &output, const char *variable_name) const;
};

inline void SparseMatrixStaticCSR::
apply(const double *x, double *y) const
{
    assert(x != NULL && y != NULL);
    if(x != NULL && y != NULL) {
       for(int i=0, k=rowstart[0]; i<m; ++i){
           double d=0;
           for(; k<rowstart[i+1]; ++k) d+=value[k]*x[colindex[k]];
           y[i]=d;
       }
    }
}

inline void SparseMatrixStaticCSR::
apply_transpose(const double *x, double *y) const
{
    assert(x && y);
    if(x != NULL && y != NULL) {
       BLAS::set_zero(n, y);
       for(int i=0, k=rowstart[0]; i<m; ++i){
           double xi=x[i];
           for(; k<rowstart[i+1]; ++k) y[colindex[k]]+=value[k]*xi;
       }
    }
}

}

#endif
