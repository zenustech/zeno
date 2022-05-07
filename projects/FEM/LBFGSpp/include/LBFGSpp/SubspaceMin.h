// Copyright (C) 2020-2021 Yixuan Qiu <yixuan.qiu@cos.name>
// Under MIT license

#ifndef LBFGSPP_SUBSPACE_MIN_H
#define LBFGSPP_SUBSPACE_MIN_H

#include <stdexcept>
#include <vector>
#include <Eigen/Core>
#include "BFGSMat.h"


/// \cond

namespace LBFGSpp {


//
// Subspace minimization procedure of the L-BFGS-B algorithm,
// mainly for internal use.
//
// The target of subspace minimization is to minimize the quadratic function m(x)
// over the free variables, subject to the bound condition.
// Free variables stand for coordinates that are not at the boundary in xcp,
// the generalized Cauchy point.
//
// In the classical implementation of L-BFGS-B [1], the minimization is done by first
// ignoring the box constraints, followed by a line search. Our implementation is
// an exact minimization subject to the bounds, based on the BOXCQP algorithm [2].
//
// Reference:
// [1] R. H. Byrd, P. Lu, and J. Nocedal (1995). A limited memory algorithm for bound constrained optimization.
// [2] C. Voglis and I. E. Lagaris (2004). BOXCQP: An algorithm for bound constrained convex quadratic problems.
//
template <typename Scalar>
class SubspaceMin
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef std::vector<int> IndexSet;

    // v[ind]
    static Vector subvec(const Vector& v, const IndexSet& ind)
    {
        const int nsub = ind.size();
        Vector res(nsub);
        for(int i = 0; i < nsub; i++)
            res[i] = v[ind[i]];
        return res;
    }

    // v[ind] = rhs
    static void subvec_assign(Vector& v, const IndexSet& ind, const Vector& rhs)
    {
        const int nsub = ind.size();
        for(int i = 0; i < nsub; i++)
            v[ind[i]] = rhs[i];
    }

    // Check whether the vector is within the bounds
    static bool in_bounds(const Vector& x, const Vector& lb, const Vector& ub)
    {
        const int n = x.size();
        for(int i = 0; i < n; i++)
        {
            if(x[i] < lb[i] || x[i] > ub[i])
                return false;
        }
        return true;
    }

    // Test convergence of P set
    static bool P_converged(const IndexSet& yP_set, const Vector& vecy, const Vector& vecl, const Vector& vecu)
    {
        const int nP = yP_set.size();
        for(int i = 0; i < nP; i++)
        {
            const int coord = yP_set[i];
            if(vecy[coord] < vecl[coord] || vecy[coord] > vecu[coord])
                return false;
        }
        return true;
    }

    // Test convergence of L set
    static bool L_converged(const IndexSet& yL_set, const Vector& lambda)
    {
        const int nL = yL_set.size();
        for(int i = 0; i < nL; i++)
        {
            const int coord = yL_set[i];
            if(lambda[coord] < Scalar(0))
                return false;
        }
        return true;
    }

    // Test convergence of L set
    static bool U_converged(const IndexSet& yU_set, const Vector& mu)
    {
        const int nU = yU_set.size();
        for(int i = 0; i < nU; i++)
        {
            const int coord = yU_set[i];
            if(mu[coord] < Scalar(0))
                return false;
        }
        return true;
    }

public:
    // bfgs:       An object that represents the BFGS approximation matrix.
    // x0:         Current parameter vector.
    // xcp:        Computed generalized Cauchy point.
    // g:          Gradient at x0.
    // lb:         Lower bounds for x.
    // ub:         Upper bounds for x.
    // Wd:         W'(xcp - x0)
    // newact_set: Coordinates that newly become active during the GCP procedure.
    // fv_set:     Free variable set.
    // maxit:      Maximum number of iterations.
    // drt:        The output direction vector, drt = xsm - x0.
    static void subspace_minimize(
        const BFGSMat<Scalar, true>& bfgs, const Vector& x0, const Vector& xcp, const Vector& g,
        const Vector& lb, const Vector& ub, const Vector& Wd, const IndexSet& newact_set, const IndexSet& fv_set, int maxit,
        Vector& drt
    )
    {
        // std::cout << "========================= Entering subspace minimization =========================\n\n";

        // d = xcp - x0
        drt.noalias() = xcp - x0;
        // Size of free variables
        const int nfree = fv_set.size();
        // If there is no free variable, simply return drt
        if(nfree < 1)
        {
            // std::cout << "========================= (Early) leaving subspace minimization =========================\n\n";
            return;
        }

        // std::cout << "New active set = [ "; for(std::size_t i = 0; i < newact_set.size(); i++)  std::cout << newact_set[i] << " "; std::cout << "]\n";
        // std::cout << "Free variable set = [ "; for(std::size_t i = 0; i < fv_set.size(); i++)  std::cout << fv_set[i] << " "; std::cout << "]\n\n";

        // Extract the rows of W in the free variable set
        Matrix WF = bfgs.Wb(fv_set);
        // Compute F'BAb = -F'WMW'AA'd
        Vector vecc(nfree);
        bfgs.compute_FtBAb(WF, fv_set, newact_set, Wd, drt, vecc);
        // Set the vector c=F'BAb+F'g for linear term, and vectors l and u for the new bounds
        Vector vecl(nfree), vecu(nfree);
        for(int i = 0; i < nfree; i++)
        {
            const int coord = fv_set[i];
            vecl[i] = lb[coord] - x0[coord];
            vecu[i] = ub[coord] - x0[coord];
            vecc[i] += g[coord];
        }
        // Solve y = -inv(B[F, F]) * c
        Vector vecy(nfree);
        bfgs.solve_PtBP(WF, -vecc, vecy);
        // Test feasibility
        // If yes, then the solution has been found
        if(in_bounds(vecy, vecl, vecu))
        {
            subvec_assign(drt, fv_set, vecy);
            return;
        }
        // Otherwise, enter the iterations

        // Make a copy of y as a fallback solution
        Vector yfallback = vecy;
        // Dual variables
        Vector lambda = Vector::Zero(nfree), mu = Vector::Zero(nfree);

        // Iterations
        IndexSet L_set, U_set, P_set, yL_set, yU_set, yP_set;
        L_set.reserve(nfree / 3); yL_set.reserve(nfree / 3);
        U_set.reserve(nfree / 3); yU_set.reserve(nfree / 3);
        P_set.reserve(nfree); yP_set.reserve(nfree);
        int k;
        for(k = 0; k < maxit; k++)
        {
            // Construct the L, U, and P sets, and then update values
            // Indices in original drt vector
            L_set.clear();
            U_set.clear();
            P_set.clear();
            // Indices in y
            yL_set.clear();
            yU_set.clear();
            yP_set.clear();
            for(int i = 0; i < nfree; i++)
            {
                const int coord = fv_set[i];
                const Scalar li = vecl[i], ui = vecu[i];
                if( (vecy[i] < li) || (vecy[i] == li && lambda[i] >= Scalar(0)) )
                {
                    L_set.push_back(coord);
                    yL_set.push_back(i);
                    vecy[i] = li;
                    mu[i] = Scalar(0);
                } else if( (vecy[i] > ui) || (vecy[i] == ui && mu[i] >= Scalar(0)) ) {
                    U_set.push_back(coord);
                    yU_set.push_back(i);
                    vecy[i] = ui;
                    lambda[i] = Scalar(0);
                } else {
                    P_set.push_back(coord);
                    yP_set.push_back(i);
                    lambda[i] = Scalar(0);
                    mu[i] = Scalar(0);
                }
            }

            /* std::cout << "** Iter " << k << " **\n";
            std::cout << "   L = [ "; for(std::size_t i = 0; i < L_set.size(); i++)  std::cout << L_set[i] << " "; std::cout << "]\n";
            std::cout << "   U = [ "; for(std::size_t i = 0; i < U_set.size(); i++)  std::cout << U_set[i] << " "; std::cout << "]\n";
            std::cout << "   P = [ "; for(std::size_t i = 0; i < P_set.size(); i++)  std::cout << P_set[i] << " "; std::cout << "]\n\n"; */

            // Extract the rows of W in the P set
            Matrix WP = bfgs.Wb(P_set);
            // Solve y[P] = -inv(B[P, P]) * (B[P, L] * l[L] + B[P, U] * u[U] + c[P])
            const int nP = P_set.size();
            if(nP > 0)
            {
                Vector rhs = subvec(vecc, yP_set);
                Vector lL = subvec(vecl, yL_set);
                Vector uU = subvec(vecu, yU_set);
                Vector tmp(nP);
                bool nonzero = bfgs.apply_PtBQv(WP, L_set, lL, tmp, true);
                if(nonzero)
                    rhs.noalias() += tmp;
                nonzero = bfgs.apply_PtBQv(WP, U_set, uU, tmp, true);
                if(nonzero)
                    rhs.noalias() += tmp;

                bfgs.solve_PtBP(WP, -rhs, tmp);
                subvec_assign(vecy, yP_set, tmp);
            }

            // Solve lambda[L] = B[L, F] * y + c[L]
            const int nL = L_set.size();
            const int nU = U_set.size();
            Vector Fy;
            if(nL > 0 || nU > 0)
                bfgs.apply_WtPv(fv_set, vecy, Fy);
            if(nL > 0)
            {
                Vector res;
                bfgs.apply_PtWMv(L_set, Fy, res, Scalar(-1));
                res.noalias() += subvec(vecc, yL_set);
                subvec_assign(lambda, yL_set, res);
            }

            // Solve mu[U] = -B[U, F] * y - c[U]
            if(nU > 0)
            {
                Vector res;
                bfgs.apply_PtWMv(U_set, Fy, res, Scalar(-1));
                res.noalias() = -res - subvec(vecc, yU_set);
                subvec_assign(mu, yU_set, res);
            }

            // Test convergence
            if( L_converged(yL_set, lambda) && U_converged(yU_set, mu) && P_converged(yP_set, vecy, vecl, vecu) )
                break;
        }

        // If the iterations do not converge, try the projection
        if(k >= maxit)
        {
            vecy.noalias() = vecy.cwiseMax(vecl).cwiseMin(vecu);
            subvec_assign(drt, fv_set, vecy);
            // Test whether drt is a descent direction
            Scalar dg = drt.dot(g);
            // If yes, return the result
            if(dg <= -std::numeric_limits<Scalar>::epsilon())
                return;

            // If not, fall back to the projected unconstrained solution
            vecy.noalias() = yfallback.cwiseMax(vecl).cwiseMin(vecu);
            subvec_assign(drt, fv_set, vecy);
            dg = drt.dot(g);
            if(dg <= -std::numeric_limits<Scalar>::epsilon())
                return;

            // If still not, fall back to the unconstrained solution
            subvec_assign(drt, fv_set, yfallback);
            return;
        }

        // std::cout << "** Minimization finished in " << k + 1 << " iteration(s) **\n\n";
        // std::cout << "========================= Leaving subspace minimization =========================\n\n";

        subvec_assign(drt, fv_set, vecy);
    }
};


} // namespace LBFGSpp

/// \endcond

#endif // LBFGSPP_SUBSPACE_MIN_H
