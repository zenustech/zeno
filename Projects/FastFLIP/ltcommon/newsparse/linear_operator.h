#ifndef LINEAR_OPERATOR_H
#define LINEAR_OPERATOR_H

#include <cassert>
#include <iostream>
#include <vector>

// virtual base class for linear operators

struct LinearOperator
{
    int m, n; // like an m*n matrix
    
    LinearOperator(int m_=0) : m(m_), n(m_) {}
    LinearOperator(int m_, int n_) : m(m_), n(n_) {}
    virtual ~LinearOperator(void) {}
    
    // y=A*x: x should have dimension n, y dimension m
    virtual void apply(const double *x, double *y) const = 0;
    
    // z=y-A*x: x should have dimension n, y and z dimension m
    // z and y may be aliased (y==z)
    virtual void apply_and_subtract(const double *x, const double *y, double *z) const = 0;
    
    // y=A^T*x: x should have dimension m, y dimension n
    virtual void apply_transpose(const double *x, double *y) const = 0;
    
    // z=y-A^T*x: x should have dimension m, y and z dimension n
    // z and y may be aliased (y==z)
    virtual void apply_transpose_and_subtract(const double *x, const double *y, double *z) const = 0;
    
    // shortcuts if you're using std::vector
    virtual void apply(const std::vector<double> &x, std::vector<double> &y) const
    {
        assert(x.size()>=(size_t)n);
        y.resize(m);
        apply(&x[0], &y[0]);
    }
    
    virtual void apply_and_subtract(const std::vector<double> &x, const std::vector<double> &y, std::vector<double> &z) const
    {
        assert(x.size()>=(size_t)n && y.size()>=(size_t)m);
        z.resize(m);
        apply_and_subtract(&x[0], &y[0], &z[0]);
    }
    
    virtual void apply_transpose(const std::vector<double> &x, std::vector<double> &y) const
    {
        assert(x.size()>=(size_t)m);
        y.resize(n);
        apply_transpose(&x[0], &y[0]);
    }
    
    virtual void apply_transpose_and_subtract(const std::vector<double> &x, const std::vector<double> &y, std::vector<double> &z) const
    {
        assert(x.size()>=(size_t)m && y.size()>=(size_t)n);
        z.resize(n);
        apply_transpose_and_subtract(&x[0], &y[0], &z[0]);
    }
    
    // the following might not be implemented depending on the operator
    virtual void write_matlab(std::ostream &output, const char *variable_name) const
    {
        output<<variable_name<<"='unimplemented LinearOperator output';"<<std::endl;
    }
};

// useful extra: compositions of linear operators (e.g. for factored preconditioners)
struct FactoredLinearOperator: public LinearOperator
{
    std::vector<const LinearOperator*> factors;
    std::vector<bool> transpose;
    
    FactoredLinearOperator(int m_=0) : LinearOperator(m_), factors(0), transpose(0), temp(0) {}
    FactoredLinearOperator(const LinearOperator *A, bool Atranspose=false);
    FactoredLinearOperator(const LinearOperator *A, bool Atranspose, const LinearOperator *B, bool Btranspose);
    FactoredLinearOperator(const LinearOperator *A, bool Atranspose, const LinearOperator *B, bool Btranspose, const LinearOperator *C, bool Ctranspose);
    bool check_dimensions(void);
    using LinearOperator::apply;
    using LinearOperator::apply_and_subtract;
    using LinearOperator::apply_transpose;
    using LinearOperator::apply_transpose_and_subtract;
    virtual void apply(const double *x, double *y) const;
    virtual void apply_and_subtract(const double *x, const double *y, double *z) const;
    virtual void apply_transpose(const double *x, double *y) const;
    virtual void apply_transpose_and_subtract(const double *x, const double *y, double *z) const;
    
private:
    std::vector<double> temp; // intermediate vector for use in the apply functions if needed
};

#endif
