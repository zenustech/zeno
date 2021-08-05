#ifndef LINEAR_ASSEMBLERBASE_H
#define LINEAR_ASSEMBLERBASE_H

#include "StateDense.h"
#include "Types.h"
#include <Eigen/Sparse>
#include <tbb/tbb.h>

namespace ZenEulerGas {
namespace LinearProjection {
enum BSplineDegree : int { B1B0,
    B2B1,
    B2B0 };

template <class T, int BSplineDegree_>
class IntBSplineWeights {
public:
    // if cut-cell is needed, compute coefficients on the fly will
    // then we also need GS-LD integration
    Matrix<T, 3, 2> IntBTestdxBTrial, IntBTestxBTrial;
    Vector<T, 3> IntB;

    IntBSplineWeights()
    {
        if constexpr (BSplineDegree_ == BSplineDegree::B1B0) {
            IntBTestdxBTrial << -.5, 0, .5, -.5, 0, .5;
            IntBTestxBTrial << .125, 0, .375, .375, 0, .125;
            IntB << .125, .75, .125;
        }
        else if constexpr (BSplineDegree_ == BSplineDegree::B2B0) {
            {
                std::cout << "not implemented" << std::endl;
                exit(1);
            }
        }
        else if constexpr (BSplineDegree_ == BSplineDegree::B2B1) {
            IntBTestdxBTrial << -(T)1 / (T)3, -(T)1 / (T)6, (T)1 / (T)6, -(T)1 / (T)6,
                (T)1 / (T)6, (T)1 / (T)3;
            IntBTestxBTrial << .125, (T)1 / (T)24, (T)1 / (T)3, (T)1 / (T)3,
                (T)1 / (T)24, .125;
            IntB << (T)1 / (T)6, (T)2 / (T)3, (T)1 / (T)6;
        }
        else {
            {
                std::cout << "not implemented" << std::endl;
                exit(1);
            }
        }
    }
};

template <typename T, class StorageIndex>
class MatrixOperators {
public:
    // global operators
    Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex> G;
    Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex> Y;
    Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex> H;
    Eigen::SparseMatrix<T, Eigen::RowMajor, StorageIndex> GT;
    Eigen::SparseMatrix<T, Eigen::RowMajor, StorageIndex> YT;
    Eigen::SparseMatrix<T, Eigen::RowMajor, StorageIndex> HT;
    // mass and stiffness
    Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex> invM;
    Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex> invS;
    // linear vec and RHS terms
    Vector<T, Eigen::Dynamic> Linear_U;
    Vector<T, Eigen::Dynamic> Linear_P;
    Vector<T, Eigen::Dynamic> Linear_Y;
    Vector<T, Eigen::Dynamic> Linear_H;
    Vector<T, Eigen::Dynamic> RHS_invS_P;
    Vector<T, Eigen::Dynamic> RHS_GT_U;
    Vector<T, Eigen::Dynamic> RHS_YT_U;
    Vector<T, Eigen::Dynamic> RHS_HT_U;
};

template <class T, int dim, class StorageIndex, bool XFastestSweep,
    int BSplineDegree_>
class AssemblerBase {
public:
    using IJK = Eigen::Triplet<T>;
    using TV = Vector<T, dim>;
    using IV = Vector<int, dim>;
    using FieldHelper = ZenEulerGas::FieldHelperDense<T, dim, StorageIndex, XFastestSweep>;
    // Inputs:
    // name used for logging
    std::string sys_name = "_";
    // Operators
    MatrixOperators<T, StorageIndex> global_operators;

    // Helper Functions:
    // kernel width and offset
    static inline int global_operator_width()
    {
        if constexpr (BSplineDegree_ == BSplineDegree::B1B0)
            return 2;
        else if constexpr (BSplineDegree_ == BSplineDegree::B2B0)
            return 3;
        else if constexpr (BSplineDegree_ == BSplineDegree::B2B1)
            return 4;
        else {
            std::cout << "not implemented" << std::endl;
            exit(1);
        }
    }
    static inline int global_operator_u2P_offset()
    {
        if constexpr (BSplineDegree_ == BSplineDegree::B1B0)
            return 0;
        else if constexpr (BSplineDegree_ == BSplineDegree::B2B0)
            return -1;
        else if constexpr (BSplineDegree_ == BSplineDegree::B2B1)
            return -1;
        else {
            std::cout << "not implemented" << std::endl;
            exit(1);
        }
    }
    static inline int global_operator_kernel_size()
    {
        return std::pow(global_operator_width(), dim);
    }
    static inline int global_operator_kernel_offset(const IV& dI_u2P)
    {
        assertm(
            (dI_u2P.array() - global_operator_u2P_offset() >= 0).all() && (dI_u2P.array() - global_operator_u2P_offset() < global_operator_width()).all(),
            "out of assemble kernel");
        if constexpr (dim == 1)
            return (dI_u2P(0) - global_operator_u2P_offset());
        else if constexpr (dim == 2)
            return (dI_u2P(0) - global_operator_u2P_offset()) * global_operator_width() + (dI_u2P(1) - global_operator_u2P_offset());
        else if constexpr (dim == 3)
            return (dI_u2P(0) - global_operator_u2P_offset()) * global_operator_width() * global_operator_width() + (dI_u2P(1) - global_operator_u2P_offset()) * global_operator_width() + (dI_u2P(2) - global_operator_u2P_offset());
        else {
            std::cout << "not implemented" << std::endl;
            exit(1);
        }
    }
    // spatial I -> linear system dof
    virtual bool active_int_cell(FieldHelper& field_helper, const IV& I)
    {
        return false;
    };
    virtual StorageIndex get_idx(FieldHelper& field_helper, const IV& I)
    {
        return -1;
    };
    virtual StorageIndex get_idu(FieldHelper& field_helper, const IV& I)
    {
        return -1;
    };
    virtual StorageIndex get_idP(FieldHelper& field_helper, const IV& I)
    {
        return -1;
    };
    virtual StorageIndex get_idY(FieldHelper& field_helper, const IV& I)
    {
        return -1;
    };
    virtual StorageIndex get_idH(FieldHelper& field_helper, const IV& I)
    {
        return -1;
    };
    // copy from spatial field <-> linear vec
    void copy_from(FieldHelper& field_helper, Field<TV>& u_field,
        Field<T>& P_field, int num_u, int num_P, int num_Y,
        int num_H)
    {
        auto copyP = [&](const IV& I) {
            if (get_idP(field_helper, I) >= 0) {
                StorageIndex idx = get_idx(field_helper, I);
                StorageIndex idP = get_idP(field_helper, I);
                StorageIndex idY = get_idY(field_helper, I);
                StorageIndex idH = get_idH(field_helper, I);
                T P = P_field[idx];
                global_operators.Linear_P[idP] = P;
                if (idY >= 0)
                    global_operators.Linear_Y[idY] = P;
                if (idH >= 0)
                    global_operators.Linear_H[idH] = P;
            }
        };
        auto copyU = [&](const IV& I) {
            if (get_idu(field_helper, I) >= 0) {
                StorageIndex idx = get_idx(field_helper, I);
                StorageIndex idu = get_idu(field_helper, I);
                TV u = u_field[idx];
                for (int d = 0; d < dim; d++) {
                    global_operators.Linear_U[idu * dim + d] = u(d);
                }
            }
        };
        global_operators.Linear_U.resize(num_u * dim);
        global_operators.Linear_U.setZero();
        field_helper.iterateGridParallel(copyU, 1);
        global_operators.Linear_P.resize(num_P);
        global_operators.Linear_P.setZero();
        global_operators.Linear_Y.resize(num_Y);
        global_operators.Linear_Y.setZero();
        global_operators.Linear_H.resize(num_H);
        global_operators.Linear_H.setZero();
        field_helper.iterateGridParallel(copyP, 1);
    };
    void copy_to(FieldHelper& field_helper, Field<TV>& u_field,
        Field<T>& P_field)
    {
        auto copyP = [&](const IV& I) {
            if (get_idP(field_helper, I) >= 0) {
                StorageIndex idx = get_idx(field_helper, I);
                StorageIndex idP = get_idP(field_helper, I);
                P_field[idx] = global_operators.Linear_P[idP];
            }
        };
        auto copyU = [&](const IV& I) {
            if (get_idu(field_helper, I) >= 0) {
                StorageIndex idx = get_idx(field_helper, I);
                StorageIndex idu = get_idu(field_helper, I);
                if constexpr (dim == 1)
                    u_field[idx] = TV{ global_operators.Linear_U[idu * dim] };
                else if constexpr (dim == 2)
                    u_field[idx] = TV{ global_operators.Linear_U[idu * dim],
                        global_operators.Linear_U[idu * dim + 1] };
                else if constexpr (dim == 3)
                    u_field[idx] = TV{ global_operators.Linear_U[idu * dim],
                        global_operators.Linear_U[idu * dim + 1],
                        global_operators.Linear_U[idu * dim + 2] };
                else {
                    std::cout << "not implemented" << std::endl;
                    exit(1);
                }
            }
        };
        field_helper.iterateGridParallel(copyU, 1);
        field_helper.iterateGridParallel(copyP, 1);
    };
    virtual void copy_from_spatial_field(FieldHelper& field_helper){};
    virtual void copy_to_spatial_field(FieldHelper& field_helper){};
    // for diagonal mass/stiffness matrix
    virtual T get_mass(FieldHelper& field_helper, const IV& I) { return 0; };
    virtual T get_stiffness(FieldHelper& field_helper, const IV& I) { return 0; };
    // assemble operators
    void assemble_G(FieldHelper& field_helper)
    {
        StorageIndex n_ualpha = global_operators.Linear_U.size(),
                     n_P = global_operators.Linear_P.size();
        StorageIndex kernel_size = global_operator_kernel_size();
        StorageIndex max_nonzero_len = n_ualpha * kernel_size;

        std::vector<StorageIndex> G_row(max_nonzero_len, 0);
        std::vector<StorageIndex> G_col(max_nonzero_len, 0);
        std::vector<T> G_var(max_nonzero_len, 0);
        std::vector<IJK> G_coeffs(max_nonzero_len);

        auto assemble_G = [&](const IV& I) {
            if (active_int_cell(field_helper, I))
                field_helper.iterateDoubleKernel(
                    [&](const IV& I, const IV& test_I, const IV& trial_I) {
                        bool PU_too_far = (BSplineDegree_ == BSplineDegree::B1B0) && (((trial_I - test_I).array() < 0).any() || ((trial_I - test_I).array() > 1).any());
                        StorageIndex idu = get_idu(field_helper, test_I),
                                     idP = get_idP(field_helper, trial_I);
                        if (!PU_too_far && idu >= 0 && idP >= 0) {
                            auto dI_u2P = trial_I - test_I;
                            int kernel_offset = global_operator_kernel_offset(dI_u2P);
                            IntBSplineWeights<T, BSplineDegree_> IntWeights;
                            for (int d = 0; d < dim; d++) {
                                T dyadic_product = 1;
                                for (int d_ = 0; d_ < dim; d_++)
                                    dyadic_product *= (d == d_
                                            ? IntWeights.IntBTestdxBTrial(
                                                test_I(d_) - I(d_) + 1, trial_I(d_) - I(d_))
                                            : IntWeights.IntBTestxBTrial(test_I(d_) - I(d_) + 1,
                                                trial_I(d_) - I(d_)));

                                G_var[(idu * dim + d) * kernel_size + kernel_offset] -= dyadic_product;
                                G_row[(idu * dim + d) * kernel_size + kernel_offset] = idu * dim + d;
                                G_col[(idu * dim + d) * kernel_size + kernel_offset] = idP;
                            }
                        }
                    },
                    I, -1, 2, 0, 2);
        };
        field_helper.iterateGridColoredParallel(assemble_G, 3);
        for (int i = 0; i < max_nonzero_len; ++i) {
            G_coeffs[i] = IJK(G_row[i], G_col[i], G_var[i]);
        }
        global_operators.G.resize(n_ualpha, n_P);
        global_operators.G.setZero();
        global_operators.G.setFromTriplets(G_coeffs.begin(), G_coeffs.end());
        global_operators.GT = global_operators.G.transpose();
    };
    void assemble_Y(FieldHelper& field_helper,
        Field<std::tuple<IV, int, T>>& Y_interfaces)
    {
        StorageIndex n_ualpha = global_operators.Linear_U.size(),
                     n_Y = global_operators.Linear_Y.size();
        StorageIndex kernel_size = global_operator_kernel_size();
        StorageIndex max_nonzero_len = n_ualpha * kernel_size;

        std::vector<int> Y_row(max_nonzero_len, 0);
        std::vector<int> Y_col(max_nonzero_len, 0);
        std::vector<T> Y_var(max_nonzero_len, 0);
        std::vector<IJK> Y_coeffs;

        for (const auto& it_mark : Y_interfaces) {
            IV I;
            int alpha;
            T normal;
            std::tie(I, alpha, normal) = it_mark;

            field_helper.iterateDoubleKernel(
                [&](const IV& I, const IV& test_I, const IV& trial_I) {
                    bool PU_too_far = (BSplineDegree_ == BSplineDegree::B1B0) && (((trial_I - test_I).array() < 0).any() || ((trial_I - test_I).array() > 1).any());
                    StorageIndex idu = get_idu(field_helper, test_I),
                                 idY = get_idY(field_helper, trial_I);
                    if (!PU_too_far && test_I(alpha) - I(alpha) != 1 && trial_I(alpha) - I(alpha) != 1 && idu >= 0 && idY >= 0) {
                        auto dI_u2P = trial_I - test_I;
                        int kernel_offset = global_operator_kernel_offset(dI_u2P);
                        IntBSplineWeights<T, BSplineDegree_> IntWeights;
                        T dyadic_product_coeff = 0.5 * normal;
                        for (int d_ = 0; d_ < dim; d_++)
                            if (d_ != alpha)
                                dyadic_product_coeff *= IntWeights.IntBTestxBTrial(
                                    test_I(d_) - I(d_) + 1, trial_I(d_) - I(d_));

                        Y_var[(idu * dim + alpha) * kernel_size + kernel_offset] += dyadic_product_coeff;
                        Y_row[(idu * dim + alpha) * kernel_size + kernel_offset] = idu * dim + alpha;
                        Y_col[(idu * dim + alpha) * kernel_size + kernel_offset] = idY;
                    }
                },
                I, -1, 2, 0, 2);
        }

        for (int i = 0; i < n_ualpha * kernel_size; ++i)
            if (Y_var[i] != 0)
                Y_coeffs.push_back(IJK(Y_row[i], Y_col[i], Y_var[i]));

        global_operators.Y.resize(n_ualpha, n_Y);
        global_operators.Y.setZero();
        global_operators.Y.setFromTriplets(Y_coeffs.begin(), Y_coeffs.end());
        global_operators.YT = global_operators.Y.transpose();
    };
    void assemble_H(FieldHelper& field_helper,
        Field<std::tuple<IV, int, T>>& H_interfaces)
    {
        StorageIndex n_ualpha = global_operators.Linear_U.size(),
                     n_H = global_operators.Linear_H.size();
        StorageIndex kernel_size = global_operator_kernel_size();
        StorageIndex max_nonzero_len = n_ualpha * kernel_size;

        std::vector<int> H_row(max_nonzero_len, 0);
        std::vector<int> H_col(max_nonzero_len, 0);
        std::vector<T> H_var(max_nonzero_len, 0);
        std::vector<IJK> H_coeffs;

        for (const auto& it_mark : H_interfaces) {
            IV I;
            int alpha;
            T normal;
            std::tie(I, alpha, normal) = it_mark;

            field_helper.iterateDoubleKernel(
                [&](const IV& I, const IV& test_I, const IV& trial_I) {
                    bool PU_too_far = (BSplineDegree_ == BSplineDegree::B1B0) && (((trial_I - test_I).array() < 0).any() || ((trial_I - test_I).array() > 1).any());
                    StorageIndex idu = get_idu(field_helper, test_I),
                                 idH = get_idH(field_helper, trial_I);
                    if (!PU_too_far && test_I(alpha) - I(alpha) != 1 && trial_I(alpha) - I(alpha) != 1 && idu >= 0 && idH >= 0) {
                        auto dI_u2P = trial_I - test_I;
                        int kernel_offset = global_operator_kernel_offset(dI_u2P);
                        IntBSplineWeights<T, BSplineDegree_> IntWeights;
                        T dyadic_product_coeff = 0.5 * normal;
                        for (int d_ = 0; d_ < dim; d_++)
                            if (d_ != alpha)
                                dyadic_product_coeff *= IntWeights.IntBTestxBTrial(
                                    test_I(d_) - I(d_) + 1, trial_I(d_) - I(d_));

                        H_var[(idu * dim + alpha) * kernel_size + kernel_offset] += dyadic_product_coeff;
                        H_row[(idu * dim + alpha) * kernel_size + kernel_offset] = idu * dim + alpha;
                        H_col[(idu * dim + alpha) * kernel_size + kernel_offset] = idH;
                    }
                },
                I, -1, 2, 0, 2);
        }

        for (int i = 0; i < n_ualpha * kernel_size; ++i)
            if (H_var[i] != 0)
                H_coeffs.push_back(IJK(H_row[i], H_col[i], H_var[i]));

        global_operators.H.resize(n_ualpha, n_H);
        global_operators.H.setZero();
        global_operators.H.setFromTriplets(H_coeffs.begin(), H_coeffs.end());
        global_operators.HT = global_operators.H.transpose();
    };
    // assemble mass/stiffness matrix
    void assemble_invM(FieldHelper& field_helper)
    {
        StorageIndex n_ualpha = global_operators.Linear_U.size();

        std::vector<int> invM_row(n_ualpha, 0);
        std::vector<int> invM_col(n_ualpha, 0);
        std::vector<T> invM_var(n_ualpha, 0);
        std::vector<IJK> invM_coeffs(n_ualpha);

        auto assemble_invM = [&](const IV& I) {
            if (get_idu(field_helper, I) >= 0) {
                StorageIndex idu = get_idu(field_helper, I);
                T M = 0.0;
                field_helper.iterateKernel(
                    [&](const IV& I, const IV& adj_I) {
                        if (active_int_cell(field_helper, adj_I)) {
                            IntBSplineWeights<T, BSplineDegree_> IntWeights;
                            T dyadic_product = 1;
                            for (int d = 0; d < dim; d++)
                                dyadic_product *= IntWeights.IntB[adj_I[d] - I[d] + 1];
                            M += dyadic_product * get_mass(field_helper, adj_I);
                        }
                    },
                    I, -1, 2);
                assertm(M > 0, "negative or zero lumped mass");
                for (int d = 0; d < dim; d++) {
                    invM_row[idu * dim + d] = idu * dim + d;
                    invM_col[idu * dim + d] = idu * dim + d;
                    invM_var[idu * dim + d] = (T)1 / M;
                }
            }
        };
        field_helper.iterateGridParallel(assemble_invM, 1);
        for (int i = 0; i < n_ualpha; ++i)
            invM_coeffs[i] = IJK(invM_row[i], invM_col[i], invM_var[i]);

        global_operators.invM.resize(n_ualpha, n_ualpha);
        global_operators.invM.setZero();
        global_operators.invM.setFromTriplets(invM_coeffs.begin(),
            invM_coeffs.end());
    };
    void assemble_invS(FieldHelper& field_helper)
    {
        StorageIndex n_P = global_operators.Linear_P.size();
        // diag invStiffness
        std::vector<int> invS_row(n_P, 0);
        std::vector<int> invS_col(n_P, 0);
        std::vector<T> invS_var(n_P, 0);
        std::vector<IJK> invS_coeffs(n_P);

        auto assemble_invS = [&](const IV& I) {
            if (get_idP(field_helper, I) >= 0) {
                StorageIndex idP = get_idP(field_helper, I);
                T coeff = 0.0;
                field_helper.iterateKernel(
                    [&](const IV& I, const IV& adj_I) {
                        if (active_int_cell(field_helper, adj_I))
                            coeff += std::pow(0.5, dim);
                    },
                    I, -1, 1);
                T invS = coeff / get_stiffness(field_helper, I);
                invS_row[idP] = idP;
                invS_col[idP] = idP;
                invS_var[idP] = invS;
            }
        };
        field_helper.iterateGridParallel(assemble_invS, 1);
        for (int i = 0; i < n_P; ++i)
            invS_coeffs[i] = IJK(invS_row[i], invS_col[i], invS_var[i]);

        global_operators.invS.resize(n_P, n_P);
        global_operators.invS.setZero();
        global_operators.invS.setFromTriplets(invS_coeffs.begin(),
            invS_coeffs.end());
    };
    // assemble RHS
    void assemble_RHS_invS_P()
    {
        global_operators.RHS_invS_P = global_operators.invS * global_operators.Linear_P;
    };
    void assemble_RHS_GT_U()
    {
        global_operators.RHS_GT_U = global_operators.GT * global_operators.Linear_U;
    };
    void assemble_RHS_YT_U(
        FieldHelper& field_helper,
        Field<std::tuple<IV, int, int, T>>& moving_Y_interfaces_override)
    {
        global_operators.RHS_YT_U = global_operators.YT * global_operators.Linear_U;

        for (const auto& it_mark : moving_Y_interfaces_override) {
            auto [I, d, normal, inteface_normal_vel] = it_mark;
            // 1. check if this is a valid interface override: i.e. normal >0 cell
            // type should be active_cell|bound_cell
            // 2. fix RHS by -= normal*inter_vel/(2**(dim-1))
            IV cell_I = I;
            if (normal > 0)
                cell_I[d] -= 1;
            if (active_int_cell(field_helper, cell_I)) {
                field_helper.iterateKernel(
                    [&](const IV& I, const IV& adj_I) {
                        if (I(d) != adj_I(d))
                            return;
                        StorageIndex idY = get_idY(field_helper, adj_I);
                        global_operators.RHS_YT_U[idY] -= (T)normal * std::pow(0.5, (T)dim - 1) * inteface_normal_vel;
                    },
                    I, 0, 2);
            }
        }
    };
    void assemble_RHS_HT_U()
    {
        global_operators.RHS_HT_U = global_operators.HT * global_operators.Linear_U;
    };
    // main assemble
    void
    assemble(FieldHelper& field_helper,
        Field<std::tuple<IV, int, T>>& Y_interfaces,
        Field<std::tuple<IV, int, T>>& H_interfaces,
        Field<std::tuple<IV, int, int, T>>& moving_Y_interfaces_override,
        bool assemble_RHS_only = false)
    {
        // BOW_TIMER_FLAG("Assemble" + sys_name);
        // Logging::info("Assemble" + sys_name);
        if (!assemble_RHS_only) {
            assemble_G(field_helper);
            // Logging::debug("assembled G");
            assemble_Y(field_helper, Y_interfaces);
            // Logging::debug("assembled B");
            assemble_H(field_helper, H_interfaces);
            // Logging::debug("assembled H");
            assemble_invM(field_helper);
            // Logging::debug("assembled M-1");
            assemble_invS(field_helper);
            // Logging::debug("assembled S-1");
        }
        assemble_RHS_invS_P();
        // Logging::debug("assembled RHS U");
        assemble_RHS_GT_U();
        // Logging::debug("assembled RHS P");
        assemble_RHS_YT_U(field_helper, moving_Y_interfaces_override);
        // Logging::debug("assembled RHS B");
        assemble_RHS_HT_U();
        // Logging::debug("assembled RHS H");
    };
};
}
} // namespace ZenEulerGas::LinearProjection

#endif
