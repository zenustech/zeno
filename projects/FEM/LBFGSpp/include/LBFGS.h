// Copyright (C) 2016-2021 Yixuan Qiu <yixuan.qiu@cos.name>
// Under MIT license

#ifndef LBFGSPP_LBFGS_H
#define LBFGSPP_LBFGS_H

#include <Eigen/Core>
#include "LBFGSpp/Param.h"
#include "LBFGSpp/BFGSMat.h"
#include "LBFGSpp/LineSearchBacktracking.h"
#include "LBFGSpp/LineSearchBracketing.h"
#include "LBFGSpp/LineSearchNocedalWright.h"


namespace LBFGSpp {


///
/// L-BFGS solver for unconstrained numerical optimization
///
template < typename Scalar,
           template<class> class LineSearch = LineSearchBacktracking >
class LBFGSSolver
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Map<Vector> MapVec;

    const LBFGSParam<Scalar>& m_param;  // Parameters to control the LBFGS algorithm
    BFGSMat<Scalar>           m_bfgs;   // Approximation to the Hessian matrix
    Vector                    m_fx;     // History of the objective function values
    Vector                    m_xp;     // Old x
    Vector                    m_grad;   // New gradient
    Vector                    m_gradp;  // Old gradient
    Vector                    m_drt;    // Moving direction

    // Reset internal variables
    // n: dimension of the vector to be optimized
    inline void reset(int n)
    {
        const int m = m_param.m;
        m_bfgs.reset(n, m);
        m_xp.resize(n);
        m_grad.resize(n);
        m_gradp.resize(n);
        m_drt.resize(n);
        if(m_param.past > 0)
            m_fx.resize(m_param.past);
    }

public:
    ///
    /// Constructor for the L-BFGS solver.
    ///
    /// \param param An object of \ref LBFGSParam to store parameters for the
    ///        algorithm
    ///
    LBFGSSolver(const LBFGSParam<Scalar>& param) :
        m_param(param)
    {
        m_param.check_param();
    }

    ///
    /// Minimizing a multivariate function using the L-BFGS algorithm.
    /// Exceptions will be thrown if error occurs.
    ///
    /// \param f  A function object such that `f(x, grad)` returns the
    ///           objective function value at `x`, and overwrites `grad` with
    ///           the gradient.
    /// \param x  In: An initial guess of the optimal point. Out: The best point
    ///           found.
    /// \param fx Out: The objective function value at `x`.
    ///
    /// \return Number of iterations used.
    ///
    template <typename Foo,
        typename HinvOp>
    inline int minimize(Foo&& f, Vector& x, Scalar& fx,HinvOp&& h,bool customed_Hinv = false)
    {
        using std::abs;

        // Dimension of the vector
        const int n = x.size();
        reset(n);

        // The length of lag for objective function value to test convergence
        const int fpast = m_param.past;

        // Evaluate function and compute gradient
        fx = f(x, m_grad);
        Scalar gnorm = m_grad.norm();
        if(fpast > 0)
            m_fx[0] = fx;

        // Early exit if the initial x is already a minimizer
        if(gnorm <= m_param.epsilon || gnorm <= m_param.epsilon_rel * x.norm())
        {
            return 1;
        }

        // Initial direction
        m_drt.noalias() = -m_grad;
        // Initial step size
        Scalar step = Scalar(1) / m_drt.norm();

        // Number of iterations used
        int k = 1;
        for( ; ; )
        {
            // Save the curent x and gradient
            m_xp.noalias() = x;
            m_gradp.noalias() = m_grad;

            // Line search to update x, fx and gradient
            LineSearch<Scalar>::LineSearch(f, fx, x, m_grad, step, m_drt, m_xp, m_param);

            // New gradient norm
            gnorm = m_grad.norm();

            // Convergence test -- gradient
            if(gnorm <= m_param.epsilon || gnorm <= m_param.epsilon_rel * x.norm())
            {
                return k;
            }
            // Convergence test -- objective function value
            if(fpast > 0)
            {
                const Scalar fxd = m_fx[k % fpast];
                if(k >= fpast && abs(fxd - fx) <= m_param.delta * std::max(std::max(abs(fx), abs(fxd)), Scalar(1)))
                    return k;

                m_fx[k % fpast] = fx;
            }
            // Maximum number of iterations
            if(m_param.max_iterations != 0 && k >= m_param.max_iterations)
            {
                return k;
            }

            // Update s and y
            // s_{k+1} = x_{k+1} - x_k
            // y_{k+1} = g_{k+1} - g_k
            m_bfgs.add_correction(x - m_xp, m_grad - m_gradp);

            // Recursive formula to compute d = -H * g
            if(!customed_Hinv)
                m_bfgs.apply_Hv(m_grad, -Scalar(1), m_drt);
            else
                m_bfgs.apply_customed_Hv(m_grad,-Scalar(1),m_drt,h);

            // Reset step = 1.0 as initial guess for the next line search
            step = Scalar(1);
            k++;
        }

        return k;
    }
};


} // namespace LBFGSpp

#endif // LBFGSPP_LBFGS_H
