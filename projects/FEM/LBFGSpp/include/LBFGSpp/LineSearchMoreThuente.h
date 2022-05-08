// Copyright (C) 2020-2021 Yixuan Qiu <yixuan.qiu@cos.name>
// Under MIT license

#ifndef LBFGSPP_LINE_SEARCH_MORE_THUENTE_H
#define LBFGSPP_LINE_SEARCH_MORE_THUENTE_H

#include <stdexcept>  // std::invalid_argument, std::runtime_error
#include <Eigen/Core>
#include "LBFGSpp/Param.h"


namespace LBFGSpp {


///
/// The line search algorithm by Moré and Thuente (1994), currently used for the L-BFGS-B algorithm.
///
/// The target of this line search algorithm is to find a step size \f$\alpha\f$ that satisfies the strong Wolfe condition
/// \f$f(x+\alpha d) \le f(x) + \alpha\mu g(x)^T d\f$ and \f$|g(x+\alpha d)^T d| \le \eta|g(x)^T d|\f$.
/// Our implementation is a simplified version of the algorithm in [1]. We assume that \f$0<\mu<\eta<1\f$, while in [1]
/// they do not assume \f$\eta>\mu\f$. As a result, the algorithm in [1] has two stages, but in our implementation we
/// only need the first stage to guarantee the convergence.
///
/// Reference:
/// [1] Moré, J. J., & Thuente, D. J. (1994). Line search algorithms with guaranteed sufficient decrease.
///
template <typename Scalar>
class LineSearchMoreThuente
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

    // Mininum of a quadratic function that interpolates fa, ga, and fb
    static Scalar quadratic_interp(const Scalar& a, const Scalar& b, const Scalar& fa, const Scalar& ga, const Scalar& fb)
    {
        const Scalar ba = b - a;
        return a + Scalar(0.5) * ba * ba * ga / (fa - fb + ba * ga);
    }

    // Mininum of a quadratic function that interpolates ga and gb
    // Assume that ga != gb
    static Scalar quadratic_interp(const Scalar& a, const Scalar& b, const Scalar& ga, const Scalar& gb)
    {
        return b + (b - a) * gb / (ga - gb);
    }

    // Mininum of a cubic function that interpolates fa, ga, fb and gb
    // Assume that a != b
    static Scalar cubic_interp(const Scalar& a, const Scalar& b, const Scalar& fa, const Scalar& fb, const Scalar& ga, const Scalar& gb)
    {
        using std::abs;
        using std::sqrt;

        if(a == b)
            return a;

        const Scalar ba = b - a;
        const Scalar ba2 = ba * ba;
        const Scalar ba3 = ba2 * ba;
        const Scalar fba = fb - fa;
        const Scalar z = (ga + gb) * ba - Scalar(2) * fba;
        const Scalar w = fba * ba - ga * ba2;

        // If c3 = z/(b-a)^3 == 0, reduce to quadratic problem
        const Scalar endmin = (fa < fb) ? a : b;
        if(abs(z) < std::numeric_limits<Scalar>::epsilon())
        {
            const Scalar c2 = fba / ba2 - ga / ba;
            const Scalar c1 = fba / ba - (a + b) * c2;
            // Global minimum, can be infinity
            const Scalar globmin = -c1 / (Scalar(2) * c2);
            // If c2 <= 0, or globmin is outside [a, b], then the minimum is achieved at one end point
            return (c2 > Scalar(0) && globmin >= a && globmin <= b) ? globmin : endmin;
        }

        // v = c1 / c2
        const Scalar v = (-Scalar(2) * a * w + ga * ba3 + a * (a + Scalar(2) * b) * z) /
            (w - (Scalar(2) * a + b) * z);
        // u = c2 / (3 * c3), may be very large if c3 ~= 0
        const Scalar u = (w / z - (Scalar(2) * a + b)) / Scalar(3);
        // q'(x) = c1 + 2 * c2 * x + 3 * c3 * x^2 = 0
        // x1 = -u * (1 + sqrt(1 - v/u))
        // x2 = -u * (1 - sqrt(1 - v/u)) = -v / (1 + sqrt(1 - v/u))

        // If q'(x) = 0 has no solution in [a, b], q(x) is monotone in [a, b]
        // Case I: no solution globally, 1 - v/u <= 0
        if(v / u >= Scalar(1))
            return endmin;
        // Case II: no solution in [a, b]
        const Scalar vu = Scalar(1) + sqrt(Scalar(1) - v / u);
        const Scalar sol1 = -u * vu;
        const Scalar sol2 = -v / vu;
        if( (sol1 - a) * (sol1 - b) >= Scalar(0) && (sol2 - a) * (sol2 - b) >= Scalar(0) )
            return endmin;

        // Now at least one solution is in (a, b)
        // Check the second derivative
        // q''(x) = 2 * c2 + 6 * c3 * x;
        const Scalar c3 = z / ba3;
        const Scalar c2 = Scalar(3) * c3 * u;
        const Scalar qpp1 = Scalar(2) * c2 + Scalar(6) * c3 * sol1;
        const Scalar sol = (qpp1 > Scalar(0)) ? sol1 : sol2;
        // If the local minimum is not in [a, b], return one of the end points
        if((sol - a) * (sol - b) >= Scalar(0))
            return endmin;

        // Compare the local minimum with the end points
        const Scalar c1 = v * c2;
        const Scalar fsol = fa + c1 * (sol- a) + c2 * (sol * sol - a * a) +
            c3 * (sol * sol * sol - a * a * a);
        return (fsol < std::min(fa, fb)) ? sol : endmin;
    }

    static Scalar step_selection(
        const Scalar& al, const Scalar& au, const Scalar& at,
        const Scalar& fl, const Scalar& fu, const Scalar& ft,
        const Scalar& gl, const Scalar& gu, const Scalar& gt
    )
    {
        if(al == au)
            return al;

        // ac: cubic interpolation of fl, ft, gl, gt
        // aq: quadratic interpolation of fl, gl, ft
        const Scalar ac = cubic_interp(al, at, fl, ft, gl, gt);
        const Scalar aq = quadratic_interp(al, at, fl, gl, ft);
        // Case 1: ft > fl
        if(ft > fl)
            return (std::abs(ac - al) < std::abs(aq - al)) ?
                   ac :
                   ((aq + ac) / Scalar(2));

        // as: quadratic interpolation of gl and gt
        const Scalar as = quadratic_interp(al, at, gl, gt);
        // Case 2: ft <= fl, gt * gl < 0
        if(gt * gl < Scalar(0))
            return (std::abs(ac - at) >= std::abs(as - at)) ? ac : as;

        // Case 3: ft <= fl, gt * gl >= 0, |gt| < |gl|
        const Scalar delta = Scalar(0.66);
        if(std::abs(gt) < std::abs(gl))
        {
            const Scalar res = (std::abs(ac - at) < std::abs(as - at)) ? ac : as;
            return (at > al) ?
                   std::min(at + delta * (au - at), res) :
                   std::max(at + delta * (au - at), res);
        }

        // ae: cubic interpolation of ft, fu, gt, gu
        const Scalar ae = cubic_interp(at, au, ft, fu, gt, gu);
        // Case 4: ft <= fl, gt * gl >= 0, |gt| >= |gl|
        return (at > al) ?
               std::min(at + delta * (au - at), ae) :
               std::max(at + delta * (au - at), ae);
    }

public:
    ///
    /// Line search by Moré and Thuente (1994).
    ///
    /// \param f        A function object such that `f(x, grad)` returns the
    ///                 objective function value at `x`, and overwrites `grad` with
    ///                 the gradient.
    /// \param fx       In: The objective function value at the current point.
    ///                 Out: The function value at the new point.
    /// \param x        Out: The new point moved to.
    /// \param grad     In: The current gradient vector. Out: The gradient at the
    ///                 new point.
    /// \param step     In: The initial step length. Out: The calculated step length.
    /// \param step_max The upper bound for the step size.
    /// \param drt      The current moving direction.
    /// \param xp       The current point.
    /// \param param    Parameters for the LBFGS algorithm
    ///
    template <typename Foo>
    static void LineSearch(Foo& f, Scalar& fx, Vector& x, Vector& grad,
                           Scalar& step, const Scalar& step_max,
                           const Vector& drt, const Vector& xp,
                           const LBFGSBParam<Scalar>& param)
    {
        // std::cout << "========================= Entering line search =========================\n\n";

        // Check the value of step
        if(step <= Scalar(0))
            throw std::invalid_argument("'step' must be positive");
        if(step > step_max)
            throw std::invalid_argument("'step' exceeds 'step_max'");

        // Save the function value at the current x
        const Scalar fx_init = fx;
        // Projection of gradient on the search direction
        const Scalar dg_init = grad.dot(drt);

        // std::cout << "fx_init = " << fx_init << ", dg_init = " << dg_init << std::endl << std::endl;

        // Make sure d points to a descent direction
        if(dg_init >= 0)
            throw std::logic_error("the moving direction does not decrease the objective function value");

        // Tolerance for convergence test
        // Sufficient decrease
        const Scalar test_decr = param.ftol * dg_init;
        // Curvature
        const Scalar test_curv = -param.wolfe * dg_init;

        // The bracketing interval
        Scalar  I_lo = Scalar(0),                           I_hi = std::numeric_limits<Scalar>::infinity();
        Scalar fI_lo = Scalar(0),                          fI_hi = std::numeric_limits<Scalar>::infinity();
        Scalar gI_lo = (Scalar(1) - param.ftol) * dg_init, gI_hi = std::numeric_limits<Scalar>::infinity();
        // Function value and gradient at the current step size
        x.noalias() = xp + step * drt;
        fx = f(x, grad);
        Scalar dg = grad.dot(drt);

        // std::cout << "max_step = " << step_max << ", step = " << step << ", fx = " << fx << ", dg = " << dg << std::endl;

        // Convergence test
        if(fx <= fx_init + step * test_decr && std::abs(dg) <= test_curv)
        {
            // std::cout << "** Criteria met\n\n";
            // std::cout << "========================= Leaving line search =========================\n\n";
            return;
        }

        // Extrapolation factor
        const Scalar delta = Scalar(1.1);
        int iter;
        for(iter = 0; iter < param.max_linesearch; iter++)
        {
            // ft = psi(step) = f(xp + step * drt) - f(xp) - step * test_decr
            // gt = psi'(step) = dg - mu * dg_init
            // mu = param.ftol
            const Scalar ft = fx - fx_init - step * test_decr;
            const Scalar gt = dg - param.ftol * dg_init;

            // Update bracketing interval and step size
            Scalar new_step;
            if(ft > fI_lo)
            {
                // Case 1: ft > fl
                new_step = step_selection( I_lo,  I_hi, step,
                                          fI_lo, fI_hi, ft,
                                          gI_lo, gI_hi, gt);
                I_hi = step;
                fI_hi = ft;
                gI_hi = gt;

                // std::cout << "Case 1: new step = " << new_step;

            } else if(gt * (fI_lo - step) > Scalar(0)) {
                // Case 2: ft <= fl, gt * (al - at) > 0
                new_step = std::min(step_max, step + delta * (step - I_lo));

                I_lo = step;
                fI_lo = ft;
                gI_lo = gt;

                // std::cout << "Case 2: new step = " << new_step;

            } else {
                // Case 3: ft <= fl, gt * (al - at) <= 0
                new_step = step_selection( I_lo,  I_hi, step,
                                          fI_lo, fI_hi, ft,
                                          gI_lo, gI_hi, gt);
                I_hi = I_lo;
                fI_hi = fI_lo;
                gI_hi = gI_lo;

                I_lo = step;
                fI_lo = ft;
                gI_lo = gt;

                // std::cout << "Case 3: new step = " << new_step;
            }

            // In case step, new_step, and step_max are equal, directly return the computed x and fx
            if(step == step_max && new_step >= step_max)
            {
                // std::cout << "** Maximum step size reached\n\n";
                // std::cout << "========================= Leaving line search =========================\n\n";
                return;
            }
            // Otherwise, recompute x and fx based on new_step
            step = new_step;

            if(step < param.min_step)
                throw std::runtime_error("the line search step became smaller than the minimum value allowed");

            if(step > param.max_step)
                throw std::runtime_error("the line search step became larger than the maximum value allowed");

            // Update parameter, function value, and gradient
            x.noalias() = xp + step * drt;
            fx = f(x, grad);
            dg = grad.dot(drt);

            // std::cout << ", fx = " << fx << std::endl;

            // Convergence test
            if(fx <= fx_init + step * test_decr && std::abs(dg) <= test_curv)
            {
                // std::cout << "** Criteria met\n\n";
                // std::cout << "========================= Leaving line search =========================\n\n";
                return;
            }
            if(step >= step_max)
            {
                // std::cout << "** Maximum step size reached\n\n";
                // std::cout << "========================= Leaving line search =========================\n\n";
                return;
            }
        }

        if(iter >= param.max_linesearch)
            throw std::runtime_error("the line search routine reached the maximum number of iterations");
    }
};


} // namespace LBFGSpp

#endif // LBFGSPP_LINE_SEARCH_MORE_THUENTE_H
