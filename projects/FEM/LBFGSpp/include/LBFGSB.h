// Copyright (C) 2020-2021 Yixuan Qiu <yixuan.qiu@cos.name>
// Under MIT license

#ifndef LBFGSPP_LBFGSB_H
#define LBFGSPP_LBFGSB_H

#include <stdexcept>  // std::invalid_argument
#include <vector>
#include <Eigen/Core>
#include "LBFGSpp/Param.h"
#include "LBFGSpp/BFGSMat.h"
#include "LBFGSpp/Cauchy.h"
#include "LBFGSpp/SubspaceMin.h"
#include "LBFGSpp/LineSearchMoreThuente.h"


namespace LBFGSpp {


///
/// L-BFGS-B solver for box-constrained numerical optimization
///
template < typename Scalar,
           template<class> class LineSearch = LineSearchMoreThuente >
class LBFGSBSolver
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Map<Vector> MapVec;
    typedef std::vector<int> IndexSet;

    const LBFGSBParam<Scalar>& m_param;  // Parameters to control the LBFGS algorithm
    BFGSMat<Scalar, true>      m_bfgs;   // Approximation to the Hessian matrix
    Vector                     m_fx;     // History of the objective function values
    Vector                     m_xp;     // Old x
    Vector                     m_grad;   // New gradient
    Vector                     m_gradp;  // Old gradient
    Vector                     m_drt;    // Moving direction

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

    // Project the vector x to the bound constraint set
    static void force_bounds(Vector& x, const Vector& lb, const Vector& ub)
    {
        x.noalias() = x.cwiseMax(lb).cwiseMin(ub);
    }

    // Norm of the projected gradient
    // ||P(x-g, l, u) - x||_inf
    static Scalar proj_grad_norm(const Vector& x, const Vector& g, const Vector& lb, const Vector& ub)
    {
        return ((x - g).cwiseMax(lb).cwiseMin(ub) - x).cwiseAbs().maxCoeff();
    }

    // The maximum step size alpha such that x0 + alpha * d stays within the bounds
    static Scalar max_step_size(const Vector& x0, const Vector& drt, const Vector& lb, const Vector& ub)
    {
        const int n = x0.size();
        Scalar step = std::numeric_limits<Scalar>::infinity();

        for(int i = 0; i < n; i++)
        {
            if(drt[i] > Scalar(0))
            {
                step = std::min(step, (ub[i] - x0[i]) / drt[i]);
            } else if(drt[i] < Scalar(0)) {
                step = std::min(step, (lb[i] - x0[i]) / drt[i]);
            }
        }

        return step;
    }

public:
    ///
    /// Constructor for the L-BFGS-B solver.
    ///
    /// \param param An object of \ref LBFGSParam to store parameters for the
    ///        algorithm
    ///
    LBFGSBSolver(const LBFGSBParam<Scalar>& param) :
        m_param(param)
    {
        m_param.check_param();
    }

    ///
    /// Minimizing a multivariate function subject to box constraints, using the L-BFGS-B algorithm.
    /// Exceptions will be thrown if error occurs.
    ///
    /// \param f  A function object such that `f(x, grad)` returns the
    ///           objective function value at `x`, and overwrites `grad` with
    ///           the gradient.
    /// \param x  In: An initial guess of the optimal point. Out: The best point
    ///           found.
    /// \param fx Out: The objective function value at `x`.
    /// \param lb Lower bounds for `x`.
    /// \param ub Upper bounds for `x`.
    ///
    /// \return Number of iterations used.
    ///
    template <typename Foo>
    inline int minimize(Foo& f, Vector& x, Scalar& fx, const Vector& lb, const Vector& ub)
    {
        using std::abs;

        // Dimension of the vector
        const int n = x.size();
        if(lb.size() != n || ub.size() != n)
            throw std::invalid_argument("'lb' and 'ub' must have the same size as 'x'");

        // Check whether the initial vector is within the bounds
        // If not, project to the feasible set
        force_bounds(x, lb, ub);

        // Initialization
        reset(n);

        // The length of lag for objective function value to test convergence
        const int fpast = m_param.past;

        // Evaluate function and compute gradient
        fx = f(x, m_grad);
        Scalar projgnorm = proj_grad_norm(x, m_grad, lb, ub);
        if(fpast > 0)
            m_fx[0] = fx;

        // std::cout << "x0 = " << x.transpose() << std::endl;
        // std::cout << "f(x0) = " << fx << ", ||proj_grad|| = " << projgnorm << std::endl << std::endl;

        // Early exit if the initial x is already a minimizer
        if(projgnorm <= m_param.epsilon || projgnorm <= m_param.epsilon_rel * x.norm())
        {
            return 1;
        }

        // Compute generalized Cauchy point
        Vector xcp(n), vecc;
        IndexSet newact_set, fv_set;
        Cauchy<Scalar>::get_cauchy_point(m_bfgs, x, m_grad, lb, ub, xcp, vecc, newact_set, fv_set);

        /* Vector gcp(n);
        Scalar fcp = f(xcp, gcp);
        Scalar projgcpnorm = proj_grad_norm(xcp, gcp, lb, ub);
        std::cout << "xcp = " << xcp.transpose() << std::endl;
        std::cout << "f(xcp) = " << fcp << ", ||proj_grad|| = " << projgcpnorm << std::endl << std::endl; */

        // Initial direction
        m_drt.noalias() = xcp - x;
        m_drt.normalize();
        // Tolerance for s'y >= eps * (y'y)
        const Scalar eps = std::numeric_limits<Scalar>::epsilon();
        // s and y vectors
        Vector vecs(n), vecy(n);
        // Number of iterations used
        int k = 1;
        for( ; ; )
        {
            // Save the curent x and gradient
            m_xp.noalias() = x;
            m_gradp.noalias() = m_grad;

            // Line search to update x, fx and gradient
            Scalar step_max = max_step_size(x, m_drt, lb, ub);
            step_max = std::min(m_param.max_step, step_max);
            Scalar step = Scalar(1);
            step = std::min(step, step_max);
            LineSearch<Scalar>::LineSearch(f, fx, x, m_grad, step, step_max, m_drt, m_xp, m_param);

            // New projected gradient norm
            projgnorm = proj_grad_norm(x, m_grad, lb, ub);

            /* std::cout << "** Iteration " << k << std::endl;
            std::cout << "   x = " << x.transpose() << std::endl;
            std::cout << "   f(x) = " << fx << ", ||proj_grad|| = " << projgnorm << std::endl << std::endl; */

            // Convergence test -- gradient
            if(projgnorm <= m_param.epsilon || projgnorm <= m_param.epsilon_rel * x.norm())
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
            vecs.noalias() = x - m_xp;
            vecy.noalias() = m_grad - m_gradp;
            if(vecs.dot(vecy) > eps * vecy.squaredNorm())
                m_bfgs.add_correction(vecs, vecy);

            force_bounds(x, lb, ub);
            Cauchy<Scalar>::get_cauchy_point(m_bfgs, x, m_grad, lb, ub, xcp, vecc, newact_set, fv_set);

            /*Vector gcp(n);
            Scalar fcp = f(xcp, gcp);
            Scalar projgcpnorm = proj_grad_norm(xcp, gcp, lb, ub);
            std::cout << "xcp = " << xcp.transpose() << std::endl;
            std::cout << "f(xcp) = " << fcp << ", ||proj_grad|| = " << projgcpnorm << std::endl << std::endl;*/

            SubspaceMin<Scalar>::subspace_minimize(m_bfgs, x, xcp, m_grad, lb, ub,
                vecc, newact_set, fv_set, m_param.max_submin, m_drt);

            /*Vector gsm(n);
            Scalar fsm = f(x + m_drt, gsm);
            Scalar projgsmnorm = proj_grad_norm(x + m_drt, gsm, lb, ub);
            std::cout << "xsm = " << (x + m_drt).transpose() << std::endl;
            std::cout << "f(xsm) = " << fsm << ", ||proj_grad|| = " << projgsmnorm << std::endl << std::endl;*/

            k++;
        }

        return k;
    }
};


} // namespace LBFGSpp

#endif // LBFGSPP_LBFGSB_H
