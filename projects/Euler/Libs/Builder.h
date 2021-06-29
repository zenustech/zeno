#ifndef LINEAR_BUILDER_H
#define LINEAR_BUILDER_H

#include "Assembler.h"
#include <Bow/Math/LinearSolver/Amgcl.h>

namespace Bow {
namespace LinearProjection {

template <class T, int dim, class StorageIndex>
class Builder {
public:
    using IJK = Eigen::Triplet<T>;
#ifdef BOW_AMGCL
    Bow::Math::LinearSolver::AmgclSolver<Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>, false> solver;
#else
    Eigen::ConjugateGradient<Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>, Eigen::Lower | Eigen::Upper> solver;
#endif
    Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex> A;
    Vector<T, Eigen::Dynamic> RHS;
    Vector<T, Eigen::Dynamic> x;

    T dx;

    Builder(T dx_)
        : dx(dx_)
    {
#ifdef BOW_AMGCL
        solver.prm.put("solver.type", "cg");
        solver.prm.get_child("solver").erase("M");
#endif
    };

    // Helper Functions:
    void fill_block(int row_offset, int col_offset, const Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>& block, std::vector<IJK>& A_coeffs)
    {
        for (int k = 0; k < block.outerSize(); ++k)
            for (typename Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>::InnerIterator it(block, k); it; ++it)
                A_coeffs.push_back(IJK(row_offset + it.row(), col_offset + it.col(), it.value()));
    };

    void build(T dt, const MatrixOperators<T, StorageIndex>& operators, bool build_RHS_only)
    {
        BOW_TIMER_FLAG("Build");
        Logging::info("Build");

        StorageIndex n_P = operators.Linear_P.size(), n_Y = operators.Linear_Y.size(), n_H = operators.Linear_H.size();
        StorageIndex offset_0 = 0, offset_1 = n_P, offset_2 = n_P + n_Y;

        if (!build_RHS_only) {
            std::vector<IJK> A_coeffs;

            // block 11
            // GT G + a diag
            fill_block(offset_0, offset_0, operators.GT * operators.invM * operators.G + dx * dx / dt / dt * operators.invS, A_coeffs);
            // block 12
            // GT B
            fill_block(offset_0, offset_1, operators.GT * operators.invM * operators.Y, A_coeffs);
            // block 13
            // GT H
            fill_block(offset_0, offset_2, operators.GT * operators.invM * operators.H, A_coeffs);

            // block 21
            // BT G
            fill_block(offset_1, offset_0, operators.YT * operators.invM * operators.G, A_coeffs);
            // block 22
            // BT B
            fill_block(offset_1, offset_1, operators.YT * operators.invM * operators.Y, A_coeffs);
            // block 23
            // BT H
            fill_block(offset_1, offset_2, operators.YT * operators.invM * operators.H, A_coeffs);

            // block 31
            // HT G
            fill_block(offset_2, offset_0, operators.HT * operators.invM * operators.G, A_coeffs);
            // block 32
            // HT B
            fill_block(offset_2, offset_1, operators.HT * operators.invM * operators.Y, A_coeffs);
            // block 33
            // HT H
            fill_block(offset_2, offset_2, operators.HT * operators.invM * operators.H, A_coeffs);

            A.resize(n_P + n_Y + n_H, n_P + n_Y + n_H);
            A.setZero();
            A.setFromTriplets(A_coeffs.begin(), A_coeffs.end());
        }
        // RHS
        RHS.resize(n_P + n_Y + n_H);
        RHS.setZero();
        RHS << dx * dx / dt / dt * operators.RHS_invS_P + dx / dt * operators.RHS_GT_U, dx / dt * operators.RHS_YT_U, dx / dt * operators.RHS_HT_U;
    };

    void solve(int it_limit, T converge_cretiria, bool compute)
    {
        BOW_TIMER_FLAG("Solve");
        Logging::info("Solve");
        if (compute) {
#ifdef BOW_AMGCL
            solver.prm.put("solver.maxiter", it_limit);
            solver.prm.put("solver.tol", converge_cretiria);
#else
            solver.setMaxIterations(it_limit);
            solver.setTolerance(converge_cretiria);
#endif
            solver.compute(A);
        }
        x = solver.solve(RHS);
#ifndef BOW_AMGCL
        Logging::info("#iterations:     ", solver.iterations());
        Logging::info("estimated error: ", solver.error());
#endif
    };

    void copy_to_linear(MatrixOperators<T, StorageIndex>& operators, StorageIndex offsetP, StorageIndex offsetY, StorageIndex offsetH)
    {
        StorageIndex len_P = operators.Linear_P.size(), len_Y = operators.Linear_Y.size(), len_H = operators.Linear_H.size();
        StorageIndex len_tot = x.size();
        // proper assert to avoid copy out of range
        BOW_ASSERT_INFO(offsetP + len_P <= len_tot, "copy out of range P");
        BOW_ASSERT_INFO(offsetY + len_Y <= len_tot, "copy out of range Y");
        BOW_ASSERT_INFO(offsetH + len_H <= len_tot, "copy out of range H");
        tbb::parallel_for(offsetP, offsetP + len_P, [&](int i) {
            operators.Linear_P[i - offsetP] = x[i];
        });
        tbb::parallel_for(offsetY, offsetY + len_Y, [&](int i) {
            operators.Linear_Y[i - offsetY] = x[i];
        });
        tbb::parallel_for(offsetH, offsetH + len_H, [&](int i) {
            operators.Linear_H[i - offsetH] = x[i];
        });
    };

    void build_and_solve(T dt, MatrixOperators<T, StorageIndex>& operators, bool build_RHS_only, int it_limit, T converge_cretiria)
    {
        StorageIndex n_P = operators.Linear_P.size(), n_Y = operators.Linear_Y.size(), n_H = operators.Linear_H.size();
        StorageIndex offset_0 = 0, offset_1 = n_P, offset_2 = n_P + n_Y;
        build(dt, operators, build_RHS_only);
        solve(it_limit, converge_cretiria, !build_RHS_only);
        copy_to_linear(operators, offset_0, offset_1, offset_2);
    };
};

template <class T, int dim, class StorageIndex>
class CoupledBuilder : virtual public Builder<T, dim, StorageIndex> {
public:
    using IJK = Eigen::Triplet<T>;
    using Base = Builder<T, dim, StorageIndex>;
    using Base::A;
    using Base::dx;
    using Base::RHS;
    using Base::x;

    using Base::copy_to_linear;
    using Base::fill_block;
    using Base::solve;

    CoupledBuilder(T dx_)
        : Base(dx_){};

    void coupled_build(T dt, const MatrixOperators<T, StorageIndex>& operators1, const MatrixOperators<T, StorageIndex>& operators2, bool build_RHS_only)
    {
        BOW_TIMER_FLAG("Build");
        Logging::info("Build");

        StorageIndex n_P1 = operators1.Linear_P.size(), n_Y1 = operators1.Linear_Y.size(), n_H1 = operators1.Linear_H.size();
        StorageIndex n_P2 = operators2.Linear_P.size(), n_Y2 = operators2.Linear_Y.size(), n_H2 = operators2.Linear_H.size();
        BOW_ASSERT_INFO(n_H1 == n_H2, "wrong number of interface constraint nodes");
        StorageIndex offset_0 = 0, offset_1 = n_P1, offset_2 = n_P1 + n_Y1, offset_3 = n_P1 + n_Y1 + n_H1, offset_4 = n_P1 + n_Y1 + n_H1 + n_P2;
        StorageIndex tot_dof = n_P1 + n_Y1 + n_H1 + n_P2 + n_Y2;

        if (!build_RHS_only) {
            std::vector<IJK> A_coeffs;

            // block 11
            // GT G + a diag
            fill_block(offset_0, offset_0, operators1.GT * operators1.invM * operators1.G + dx * dx / dt / dt * operators1.invS, A_coeffs);
            // block 12
            // GT B
            fill_block(offset_0, offset_1, operators1.GT * operators1.invM * operators1.Y, A_coeffs);
            // block 13
            // GT H
            fill_block(offset_0, offset_2, operators1.GT * operators1.invM * operators1.H, A_coeffs);

            // block 21
            // BT G
            fill_block(offset_1, offset_0, operators1.YT * operators1.invM * operators1.G, A_coeffs);
            // block 22
            // BT B
            fill_block(offset_1, offset_1, operators1.YT * operators1.invM * operators1.Y, A_coeffs);
            // block 23
            // BT H
            fill_block(offset_1, offset_2, operators1.YT * operators1.invM * operators1.H, A_coeffs);

            // block 31
            // HT G
            fill_block(offset_2, offset_0, operators1.HT * operators1.invM * operators1.G, A_coeffs);
            // block 32
            // HT B
            fill_block(offset_2, offset_1, operators1.HT * operators1.invM * operators1.Y, A_coeffs);

            // block 33
            // HT H + HTH
            fill_block(offset_2, offset_2, operators1.HT * operators1.invM * operators1.H + operators2.HT * operators2.invM * operators2.H, A_coeffs);
            // block 34
            // HTG
            fill_block(offset_2, offset_3, operators2.HT * operators2.invM * operators2.G, A_coeffs);
            // block 35
            // HTB
            fill_block(offset_2, offset_4, operators2.HT * operators2.invM * operators2.Y, A_coeffs);

            //43
            fill_block(offset_3, offset_2, operators2.GT * operators2.invM * operators2.H, A_coeffs);
            //44
            fill_block(offset_3, offset_3, operators2.GT * operators2.invM * operators2.G + dx * dx / dt / dt * operators2.invS, A_coeffs);
            //45
            fill_block(offset_3, offset_4, operators2.GT * operators2.invM * operators2.Y, A_coeffs);

            //53
            fill_block(offset_4, offset_2, operators2.YT * operators2.invM * operators2.H, A_coeffs);
            //54
            fill_block(offset_4, offset_3, operators2.YT * operators2.invM * operators2.G, A_coeffs);
            //55
            fill_block(offset_4, offset_4, operators2.YT * operators2.invM * operators2.Y, A_coeffs);

            A.resize(tot_dof, tot_dof);
            A.setZero();
            A.setFromTriplets(A_coeffs.begin(), A_coeffs.end());
        }
        // RHS
        RHS.resize(tot_dof);
        RHS.setZero();
        RHS << dx * dx / dt / dt * operators1.RHS_invS_P + dx / dt * operators1.RHS_GT_U, dx / dt * operators1.RHS_YT_U, dx / dt * operators1.RHS_HT_U + dx / dt * operators2.RHS_HT_U, dx * dx / dt / dt * operators2.RHS_invS_P + dx / dt * operators2.RHS_GT_U, dx / dt * operators2.RHS_YT_U;
    };

    void coupled_build_and_solve(T dt, MatrixOperators<T, StorageIndex>& operators1, MatrixOperators<T, StorageIndex>& operators2, bool build_RHS_only, int it_limit, T converge_cretiria)
    {
        StorageIndex n_P1 = operators1.Linear_P.size(), n_Y1 = operators1.Linear_Y.size(), n_H1 = operators1.Linear_H.size();
        StorageIndex n_P2 = operators2.Linear_P.size(), n_Y2 = operators2.Linear_Y.size(), n_H2 = operators2.Linear_H.size();
        BOW_ASSERT_INFO(n_H1 == n_H2, "wrong number of interface constraint nodes");
        StorageIndex offset_0 = 0, offset_1 = n_P1, offset_2 = n_P1 + n_Y1, offset_3 = n_P1 + n_Y1 + n_H1, offset_4 = n_P1 + n_Y1 + n_H1 + n_P2;
        coupled_build(dt, operators1, operators2, build_RHS_only);
        solve(it_limit, converge_cretiria, !build_RHS_only);
        copy_to_linear(operators1, offset_0, offset_1, offset_2);
        copy_to_linear(operators2, offset_3, offset_4, offset_2);
    };
};
}
} // namespace Bow::LinearProjection

#endif