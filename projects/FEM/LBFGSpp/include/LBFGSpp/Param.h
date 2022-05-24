// Copyright (C) 2016-2021 Yixuan Qiu <yixuan.qiu@cos.name>
// Under MIT license

#ifndef LBFGSPP_PARAM_H
#define LBFGSPP_PARAM_H

#include <Eigen/Core>
#include <stdexcept>  // std::invalid_argument


namespace LBFGSpp {


///
/// \defgroup Enumerations
///
/// Enumeration types for line search.
///

///
/// \ingroup Enumerations
///
/// The enumeration of line search termination conditions.
///
enum LINE_SEARCH_TERMINATION_CONDITION
{
    ///
    /// Backtracking method with the Armijo condition.
    /// The backtracking method finds the step length such that it satisfies
    /// the sufficient decrease (Armijo) condition,
    /// \f$f(x + a \cdot d) \le f(x) + \beta' \cdot a \cdot g(x)^T d\f$,
    /// where \f$x\f$ is the current point, \f$d\f$ is the current search direction,
    /// \f$a\f$ is the step length, and \f$\beta'\f$ is the value specified by
    /// \ref LBFGSParam::ftol. \f$f\f$ and \f$g\f$ are the function
    /// and gradient values respectively.
    ///
    LBFGS_LINESEARCH_BACKTRACKING_ARMIJO = 1,

    ///
    /// The backtracking method with the defualt (regular Wolfe) condition.
    /// An alias of `LBFGS_LINESEARCH_BACKTRACKING_WOLFE`.
    ///
    LBFGS_LINESEARCH_BACKTRACKING = 2,

    ///
    /// Backtracking method with regular Wolfe condition.
    /// The backtracking method finds the step length such that it satisfies
    /// both the Armijo condition (`LBFGS_LINESEARCH_BACKTRACKING_ARMIJO`)
    /// and the curvature condition,
    /// \f$g(x + a \cdot d)^T d \ge \beta \cdot g(x)^T d\f$, where \f$\beta\f$
    /// is the value specified by \ref LBFGSParam::wolfe.
    ///
    LBFGS_LINESEARCH_BACKTRACKING_WOLFE = 2,

    ///
    /// Backtracking method with strong Wolfe condition.
    /// The backtracking method finds the step length such that it satisfies
    /// both the Armijo condition (`LBFGS_LINESEARCH_BACKTRACKING_ARMIJO`)
    /// and the following condition,
    /// \f$\vert g(x + a \cdot d)^T d\vert \le \beta \cdot \vert g(x)^T d\vert\f$,
    /// where \f$\beta\f$ is the value specified by \ref LBFGSParam::wolfe.
    ///
    LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE = 3
};


///
/// Parameters to control the L-BFGS algorithm.
///
template <typename Scalar = double>
class LBFGSParam
{
public:
    ///
    /// The number of corrections to approximate the inverse Hessian matrix.
    /// The L-BFGS routine stores the computation results of previous \ref m
    /// iterations to approximate the inverse Hessian matrix of the current
    /// iteration. This parameter controls the size of the limited memories
    /// (corrections). The default value is \c 6. Values less than \c 3 are
    /// not recommended. Large values will result in excessive computing time.
    ///
    int    m;
    ///
    /// Absolute tolerance for convergence test.
    /// This parameter determines the absolute accuracy \f$\epsilon_{abs}\f$
    /// with which the solution is to be found. A minimization terminates when
    /// \f$||g|| < \max\{\epsilon_{abs}, \epsilon_{rel}||x||\}\f$,
    /// where \f$||\cdot||\f$ denotes the Euclidean (L2) norm. The default value is
    /// \c 1e-5.
    ///
    Scalar epsilon;
    ///
    /// Relative tolerance for convergence test.
    /// This parameter determines the relative accuracy \f$\epsilon_{rel}\f$
    /// with which the solution is to be found. A minimization terminates when
    /// \f$||g|| < \max\{\epsilon_{abs}, \epsilon_{rel}||x||\}\f$,
    /// where \f$||\cdot||\f$ denotes the Euclidean (L2) norm. The default value is
    /// \c 1e-5.
    ///
    Scalar epsilon_rel;
    ///
    /// Distance for delta-based convergence test.
    /// This parameter determines the distance \f$d\f$ to compute the
    /// rate of decrease of the objective function,
    /// \f$f_{k-d}(x)-f_k(x)\f$, where \f$k\f$ is the current iteration
    /// step. If the value of this parameter is zero, the delta-based convergence
    /// test will not be performed. The default value is \c 0.
    ///
    int    past;
    ///
    /// Delta for convergence test.
    /// The algorithm stops when the following condition is met,
    /// \f$|f_{k-d}(x)-f_k(x)|<\delta\cdot\max(1, |f_k(x)|, |f_{k-d}(x)|)\f$, where \f$f_k(x)\f$ is
    /// the current function value, and \f$f_{k-d}(x)\f$ is the function value
    /// \f$d\f$ iterations ago (specified by the \ref past parameter).
    /// The default value is \c 0.
    ///
    Scalar delta;
    ///
    /// The maximum number of iterations.
    /// The optimization process is terminated when the iteration count
    /// exceeds this parameter. Setting this parameter to zero continues an
    /// optimization process until a convergence or error. The default value
    /// is \c 0.
    ///
    int    max_iterations;
    ///
    /// The line search termination condition.
    /// This parameter specifies the line search termination condition that will be used
    /// by the LBFGS routine. The default value is `LBFGS_LINESEARCH_BACKTRACKING_ARMIJO`.
    ///
    int    linesearch;
    ///
    /// The maximum number of trials for the line search.
    /// This parameter controls the number of function and gradients evaluations
    /// per iteration for the line search routine. The default value is \c 20.
    ///
    int    max_linesearch;
    ///
    /// The minimum step length allowed in the line search.
    /// The default value is \c 1e-20. Usually this value does not need to be
    /// modified.
    ///
    Scalar min_step;
    ///
    /// The maximum step length allowed in the line search.
    /// The default value is \c 1e+20. Usually this value does not need to be
    /// modified.
    ///
    Scalar max_step;
    ///
    /// A parameter to control the accuracy of the line search routine.
    /// The default value is \c 1e-4. This parameter should be greater
    /// than zero and smaller than \c 0.5.
    ///
    Scalar ftol;
    ///
    /// The coefficient for the Wolfe condition.
    /// This parameter is valid only when the line-search
    /// algorithm is used with the Wolfe condition.
    /// The default value is \c 0.9. This parameter should be greater
    /// the \ref ftol parameter and smaller than \c 1.0.
    ///
    Scalar wolfe;

public:
    ///
    /// Constructor for L-BFGS parameters.
    /// Default values for parameters will be set when the object is created.
    ///
    LBFGSParam()
    {
        m              = 6;
        epsilon        = Scalar(1e-5);
        epsilon_rel    = Scalar(1e-5);
        past           = 0;
        delta          = Scalar(0);
        max_iterations = 0;
        linesearch     = LBFGS_LINESEARCH_BACKTRACKING_ARMIJO;
        max_linesearch = 20;
        min_step       = Scalar(1e-20);
        max_step       = Scalar(1e+20);
        ftol           = Scalar(1e-4);
        wolfe          = Scalar(0.9);
    }

    ///
    /// Checking the validity of L-BFGS parameters.
    /// An `std::invalid_argument` exception will be thrown if some parameter
    /// is invalid.
    ///
    inline void check_param() const
    {
        if(m <= 0)
            throw std::invalid_argument("'m' must be positive");
        if(epsilon < 0)
            throw std::invalid_argument("'epsilon' must be non-negative");
        if(epsilon_rel < 0)
            throw std::invalid_argument("'epsilon_rel' must be non-negative");
        if(past < 0)
            throw std::invalid_argument("'past' must be non-negative");
        if(delta < 0)
            throw std::invalid_argument("'delta' must be non-negative");
        if(max_iterations < 0)
            throw std::invalid_argument("'max_iterations' must be non-negative");
        if(linesearch < LBFGS_LINESEARCH_BACKTRACKING_ARMIJO ||
           linesearch > LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE)
           throw std::invalid_argument("unsupported line search termination condition");
        if(max_linesearch <= 0)
            throw std::invalid_argument("'max_linesearch' must be positive");
        if(min_step < 0)
            throw std::invalid_argument("'min_step' must be positive");
        if(max_step < min_step )
            throw std::invalid_argument("'max_step' must be greater than 'min_step'");
        if(ftol <= 0 || ftol >= 0.5)
            throw std::invalid_argument("'ftol' must satisfy 0 < ftol < 0.5");
        if(wolfe <= ftol || wolfe >= 1)
            throw std::invalid_argument("'wolfe' must satisfy ftol < wolfe < 1");
    }
};


///
/// Parameters to control the L-BFGS-B algorithm.
///
template <typename Scalar = double>
class LBFGSBParam
{
public:
    ///
    /// The number of corrections to approximate the inverse Hessian matrix.
    /// The L-BFGS-B routine stores the computation results of previous \ref m
    /// iterations to approximate the inverse Hessian matrix of the current
    /// iteration. This parameter controls the size of the limited memories
    /// (corrections). The default value is \c 6. Values less than \c 3 are
    /// not recommended. Large values will result in excessive computing time.
    ///
    int    m;
    ///
    /// Absolute tolerance for convergence test.
    /// This parameter determines the absolute accuracy \f$\epsilon_{abs}\f$
    /// with which the solution is to be found. A minimization terminates when
    /// \f$||Pg||_{\infty} < \max\{\epsilon_{abs}, \epsilon_{rel}||x||\}\f$,
    /// where \f$||x||\f$ denotes the Euclidean (L2) norm of \f$x\f$, and
    /// \f$Pg=P(x-g,l,u)-x\f$ is the projected gradient. The default value is
    /// \c 1e-5.
    ///
    Scalar epsilon;
    ///
    /// Relative tolerance for convergence test.
    /// This parameter determines the relative accuracy \f$\epsilon_{rel}\f$
    /// with which the solution is to be found. A minimization terminates when
    /// \f$||Pg||_{\infty} < \max\{\epsilon_{abs}, \epsilon_{rel}||x||\}\f$,
    /// where \f$||x||\f$ denotes the Euclidean (L2) norm of \f$x\f$, and
    /// \f$Pg=P(x-g,l,u)-x\f$ is the projected gradient. The default value is
    /// \c 1e-5.
    ///
    Scalar epsilon_rel;
    ///
    /// Distance for delta-based convergence test.
    /// This parameter determines the distance \f$d\f$ to compute the
    /// rate of decrease of the objective function,
    /// \f$f_{k-d}(x)-f_k(x)\f$, where \f$k\f$ is the current iteration
    /// step. If the value of this parameter is zero, the delta-based convergence
    /// test will not be performed. The default value is \c 1.
    ///
    int    past;
    ///
    /// Delta for convergence test.
    /// The algorithm stops when the following condition is met,
    /// \f$|f_{k-d}(x)-f_k(x)|<\delta\cdot\max(1, |f_k(x)|, |f_{k-d}(x)|)\f$, where \f$f_k(x)\f$ is
    /// the current function value, and \f$f_{k-d}(x)\f$ is the function value
    /// \f$d\f$ iterations ago (specified by the \ref past parameter).
    /// The default value is \c 1e-10.
    ///
    Scalar delta;
    ///
    /// The maximum number of iterations.
    /// The optimization process is terminated when the iteration count
    /// exceeds this parameter. Setting this parameter to zero continues an
    /// optimization process until a convergence or error. The default value
    /// is \c 0.
    ///
    int    max_iterations;
    ///
    /// The maximum number of iterations in the subspace minimization.
    /// This parameter controls the number of iterations in the subspace
    /// minimization routine. The default value is \c 10.
    ///
    int    max_submin;
    ///
    /// The maximum number of trials for the line search.
    /// This parameter controls the number of function and gradients evaluations
    /// per iteration for the line search routine. The default value is \c 20.
    ///
    int    max_linesearch;
    ///
    /// The minimum step length allowed in the line search.
    /// The default value is \c 1e-20. Usually this value does not need to be
    /// modified.
    ///
    Scalar min_step;
    ///
    /// The maximum step length allowed in the line search.
    /// The default value is \c 1e+20. Usually this value does not need to be
    /// modified.
    ///
    Scalar max_step;
    ///
    /// A parameter to control the accuracy of the line search routine.
    /// The default value is \c 1e-4. This parameter should be greater
    /// than zero and smaller than \c 0.5.
    ///
    Scalar ftol;
    ///
    /// The coefficient for the Wolfe condition.
    /// This parameter is valid only when the line-search
    /// algorithm is used with the Wolfe condition.
    /// The default value is \c 0.9. This parameter should be greater
    /// the \ref ftol parameter and smaller than \c 1.0.
    ///
    Scalar wolfe;

public:
    ///
    /// Constructor for L-BFGS-B parameters.
    /// Default values for parameters will be set when the object is created.
    ///
    LBFGSBParam()
    {
        m              = 6;
        epsilon        = Scalar(1e-5);
        epsilon_rel    = Scalar(1e-5);
        past           = 1;
        delta          = Scalar(1e-10);
        max_iterations = 0;
        max_submin     = 10;
        max_linesearch = 20;
        min_step       = Scalar(1e-20);
        max_step       = Scalar(1e+20);
        ftol           = Scalar(1e-4);
        wolfe          = Scalar(0.9);
    }

    ///
    /// Checking the validity of L-BFGS-B parameters.
    /// An `std::invalid_argument` exception will be thrown if some parameter
    /// is invalid.
    ///
    inline void check_param() const
    {
        if(m <= 0)
            throw std::invalid_argument("'m' must be positive");
        if(epsilon < 0)
            throw std::invalid_argument("'epsilon' must be non-negative");
        if(epsilon_rel < 0)
            throw std::invalid_argument("'epsilon_rel' must be non-negative");
        if(past < 0)
            throw std::invalid_argument("'past' must be non-negative");
        if(delta < 0)
            throw std::invalid_argument("'delta' must be non-negative");
        if(max_iterations < 0)
            throw std::invalid_argument("'max_iterations' must be non-negative");
        if(max_submin < 0)
            throw std::invalid_argument("'max_submin' must be non-negative");
        if(max_linesearch <= 0)
            throw std::invalid_argument("'max_linesearch' must be positive");
        if(min_step < 0)
            throw std::invalid_argument("'min_step' must be positive");
        if(max_step < min_step )
            throw std::invalid_argument("'max_step' must be greater than 'min_step'");
        if(ftol <= 0 || ftol >= 0.5)
            throw std::invalid_argument("'ftol' must satisfy 0 < ftol < 0.5");
        if(wolfe <= ftol || wolfe >= 1)
            throw std::invalid_argument("'wolfe' must satisfy ftol < wolfe < 1");
    }
};


} // namespace LBFGSpp

#endif // LBFGSPP_PARAM_H
