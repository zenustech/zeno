// Copyright (C) 2020-2021 Yixuan Qiu <yixuan.qiu@cos.name>
// Under MIT license

#ifndef LBFGSPP_CAUCHY_H
#define LBFGSPP_CAUCHY_H

#include <vector>
#include <Eigen/Core>
#include "BFGSMat.h"


/// \cond

namespace LBFGSpp {


//
// Class to compute the generalized Cauchy point (GCP) for the L-BFGS-B algorithm,
// mainly for internal use.
//
// The target of the GCP procedure is to find a step size t such that
// x(t) = x0 - t * g is a local minimum of the quadratic function m(x),
// where m(x) is a local approximation to the objective function.
//
// First determine a sequence of break points t0=0, t1, t2, ..., tn.
// On each interval [t[i-1], t[i]], x is changing linearly.
// After passing a break point, one or more coordinates of x will be fixed at the bounds.
// We search the first local minimum of m(x) by examining the intervals [t[i-1], t[i]] sequentially.
//
// Reference:
// [1] R. H. Byrd, P. Lu, and J. Nocedal (1995). A limited memory algorithm for bound constrained optimization.
//
template <typename Scalar>
class ArgSort
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef std::vector<int> IndexSet;

    const Scalar* values;

public:
    ArgSort(const Vector& value_vec) :
        values(value_vec.data())
    {}

    inline bool operator()(int key1, int key2) { return values[key1] < values[key2]; }
    inline void sort_key(IndexSet& key_vec) const
    {
        std::sort(key_vec.begin(), key_vec.end(), *this);
    }
};

template <typename Scalar>
class Cauchy
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Matrix<int, Eigen::Dynamic, 1> IntVector;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef std::vector<int> IndexSet;

    // Find the smallest index i such that brk[ord[i]] > t, assuming brk[ord] is already sorted.
    // If the return value equals n, then all values are <= t.
    static int search_greater(const Vector& brk, const IndexSet& ord, const Scalar& t, int start = 0)
    {
        const int nord = ord.size();
        int i;
        for(i = start; i < nord; i++)
        {
            if(brk[ord[i]] > t)
                break;
        }

        return i;
    }

public:
    // bfgs:       An object that represents the BFGS approximation matrix.
    // x0:         Current parameter vector.
    // g:          Gradient at x0.
    // lb:         Lower bounds for x.
    // ub:         Upper bounds for x.
    // xcp:        The output generalized Cauchy point.
    // vecc:       c = W'(xcp - x0), used in the subspace minimization routine.
    // newact_set: Coordinates that newly become active during the GCP procedure.
    // fv_set:     Free variable set.
    static void get_cauchy_point(
        const BFGSMat<Scalar, true>& bfgs, const Vector& x0, const Vector& g, const Vector& lb, const Vector& ub,
        Vector& xcp, Vector& vecc, IndexSet& newact_set, IndexSet& fv_set
    )
    {
        // std::cout << "========================= Entering GCP search =========================\n\n";

        // Initialization
        const int n = x0.size();
        xcp.resize(n);
        xcp.noalias() = x0;
        vecc.resize(2 * bfgs.num_corrections());
        vecc.setZero();
        newact_set.clear();
        newact_set.reserve(n);
        fv_set.clear();
        fv_set.reserve(n);

        // Construct break points
        Vector brk(n), vecd(n);
        // If brk[i] == 0, i belongs to active set
        // If brk[i] == Inf, i belongs to free variable set
        // Others are currently undecided
        IndexSet ord;
        ord.reserve(n);
        const Scalar inf = std::numeric_limits<Scalar>::infinity();
        for(int i = 0; i < n; i++)
        {
            if(lb[i] == ub[i])
                brk[i] = Scalar(0);
            else if(g[i] < Scalar(0))
                brk[i] = (x0[i] - ub[i]) / g[i];
            else if(g[i] > Scalar(0))
                brk[i] = (x0[i] - lb[i]) / g[i];
            else
                brk[i] = inf;

            const bool iszero = (brk[i] == Scalar(0));
            vecd[i] = iszero ? Scalar(0) : -g[i];

            if(brk[i] == inf)
                fv_set.push_back(i);
            else if(!iszero)
                ord.push_back(i);
        }

        // Sort indices of break points
        ArgSort<Scalar> sorting(brk);
        sorting.sort_key(ord);

        // Break points `brko := brk[ord]` are in increasing order
        // `ord` contains the coordinates that define the corresponding break points
        // brk[i] == 0 <=> The i-th coordinate is on the boundary
        const int nord = ord.size();
        const int nfree = fv_set.size();
        if( (nfree < 1) && (nord < 1) )
        {
            /* std::cout << "** All coordinates at boundary **\n";
            std::cout << "\n========================= Leaving GCP search =========================\n\n"; */
            return;
        }

        // First interval: [il=0, iu=brk[ord[0]]]
        // In case ord is empty, we take iu=Inf

        // p = W'd, c = 0
        Vector vecp;
        bfgs.apply_Wtv(vecd, vecp);
        // f' = -d'd
        Scalar fp = -vecd.squaredNorm();
        // f'' = -theta * f' - p'Mp
        Vector cache;
        bfgs.apply_Mv(vecp, cache);  // cache = Mp
        Scalar fpp = -bfgs.theta() * fp - vecp.dot(cache);

        // Theoretical step size to move
        Scalar deltatmin = -fp / fpp;

        // Limit on the current interval
        Scalar il = Scalar(0);
        // We have excluded the case that max(brk) <= 0
        int b = 0;
        Scalar iu = (nord < 1) ? inf : brk[ord[b]];
        Scalar deltat = iu - il;

        /* int iter = 0;
        std::cout << "** Iter " << iter << " **\n";
        std::cout << "   fp = " << fp << ", fpp = " << fpp << ", deltatmin = " << deltatmin << std::endl;
        std::cout << "   il = " << il << ", iu = " << iu << ", deltat = " << deltat << std::endl; */

        // If deltatmin >= deltat, we need to do the following things:
        // 1. Update vecc
        // 2. Since we are going to cross iu, the coordinates that define iu become active
        // 3. Update some quantities on these new active coordinates (xcp, vecd, vecp)
        // 4. Move to the next interval and compute the new deltatmin
        bool crossed_all = false;
        const int ncorr = bfgs.num_corrections();
        Vector wact(2 * ncorr);
        while(deltatmin >= deltat)
        {
            // Step 1
            vecc.noalias() += deltat * vecp;

            // Step 2
            // First check how many coordinates will be active when we cross the previous iu
            // b is the smallest number such that brko[b] == iu
            // Let bp be the largest number such that brko[bp] == iu
            // Then coordinates ord[b] to ord[bp] will be active
            const int act_begin = b;
            const int act_end = search_greater(brk, ord, iu, b) - 1;

            // If nfree == 0 and act_end == nord-1, then we have crossed all coordinates
            // We only need to update xcp from ord[b] to ord[bp], and then exit
            if( (nfree == 0) && (act_end == nord - 1) )
            {
                // std::cout << "** [ ";
                for(int i = act_begin; i <= act_end; i++)
                {
                    const int act = ord[i];
                    xcp[act] = (vecd[act] > Scalar(0)) ? ub[act] : lb[act];
                    newact_set.push_back(act);
                    // std::cout << act + 1 << " ";
                }
                // std::cout << "] become active **\n\n";
                // std::cout << "** All break points visited **\n\n";

                crossed_all = true;
                break;
            }

            // Step 3
            // Update xcp and d on active coordinates
            // std::cout << "** [ ";
            fp += deltat * fpp;
            for(int i = act_begin; i <= act_end; i++)
            {
                const int act = ord[i];
                xcp[act] = (vecd[act] > Scalar(0)) ? ub[act] : lb[act];
                // z = xcp - x0
                const Scalar zact = xcp[act] - x0[act];
                const Scalar gact = g[act];
                const Scalar ggact = gact * gact;
                wact.noalias() = bfgs.Wb(act);
                bfgs.apply_Mv(wact, cache);  // cache = Mw
                fp += ggact + bfgs.theta() * gact * zact - gact * cache.dot(vecc);
                fpp -= (bfgs.theta() * ggact + 2 * gact * cache.dot(vecp) + ggact * cache.dot(wact));
                vecp.noalias() += gact * wact;
                vecd[act] = Scalar(0);
                newact_set.push_back(act);
                // std::cout << act + 1 << " ";
            }
            // std::cout << "] become active **\n\n";

            // Step 4
            // Theoretical step size to move
            deltatmin = -fp / fpp;
            // Update interval bound
            il = iu;
            b = act_end + 1;
            // If we have visited all finite-valued break points, and have not exited earlier,
            // then the next iu will be infinity. Simply exit the loop now
            if(b >= nord)
                break;
            iu = brk[ord[b]];
            // Width of the current interval
            deltat = iu - il;

            /* iter++;
            std::cout << "** Iter " << iter << " **\n";
            std::cout << "   fp = " << fp << ", fpp = " << fpp << ", deltatmin = " << deltatmin << std::endl;
            std::cout << "   il = " << il << ", iu = " << iu << ", deltat = " << deltat << std::endl; */
        }

        // In some rare cases fpp is numerically zero, making deltatmin equal to Inf
        // If this happens, force fpp to be the machine precision
        const Scalar eps = std::numeric_limits<Scalar>::epsilon();
        if(fpp < eps)
            deltatmin = -fp / eps;

        // Last step
        if(!crossed_all)
        {
            deltatmin = std::max(deltatmin, Scalar(0));
            vecc.noalias() += deltatmin * vecp;
            const Scalar tfinal = il + deltatmin;
            // Update xcp on free variable coordinates
            for(int i = 0; i < nfree; i++)
            {
                const int coord = fv_set[i];
                xcp[coord] = x0[coord] + tfinal * vecd[coord];
            }
            for(int i = b; i < nord; i++)
            {
                const int coord = ord[i];
                xcp[coord] = x0[coord] + tfinal * vecd[coord];
                fv_set.push_back(coord);
            }
        }
        // std::cout << "\n========================= Leaving GCP search =========================\n\n";
    }
};


} // namespace LBFGSpp

/// \endcond

#endif // LBFGSPP_CAUCHY_H
