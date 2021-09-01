#ifndef EULER_GAS_DRIVE_BASIC_H
#define EULER_GAS_DRIVE_BASIC_H

#include "IdealGas.h"
#include "StateDense.h"
#include "Types.h"
#include <Eigen/Sparse>
#include <tbb/tbb.h>

namespace ZenEulerGas {

class AbstractOp {
};

template <class T, int dim, class StorageIndex, bool XFastestSweep>
class ExtrapolateOp : public AbstractOp {
public:
    // Inputs
    // FieldHelperSPGrid<T, dim, StorageIndex>& field_helper;
    FieldHelperDense<T, dim, StorageIndex, XFastestSweep>& field_helper;
    const Array<T, dim + 2, 1>& q_amb;

    // Outputs:
    inline void operator()()
    {
        // BOW_TIMER_FLAG("Extrapolate");
        // Logging::info("Extrapolate");
        // zero all none gas qs
        field_helper.iterateGridSerial(
            [&](const Vector<int, dim>& I) {
                int idx = field_helper.grid[I].idx;
                if (field_helper.cell_type[idx] != CellType::GAS && field_helper.cell_type[idx] != CellType::INLET)
                    field_helper.q[idx].setZero();
            },
            field_helper.grid.ghost_layer);
        // extrapolate
        auto ops = [&](const Vector<int, dim>& I) {
            int idx = field_helper.grid[I].idx;
            int iduf = field_helper.grid[I].iduf;
            int cell_type = field_helper.cell_type[idx];
            if (iduf >= 0 && cell_type != CellType::GAS && cell_type != CellType::INLET) {
                Array<T, dim + 2, 1> sum_Qs = Array<T, dim + 2, 1>::Zero();
                int sum_Qs_n = 0;
                field_helper.iterateKernel(
                    [&](const Vector<int, dim>& I, const Vector<int, dim>& adj_I) {
                        int adj_idx = field_helper.grid[adj_I].idx;
                        int adj_iduf = field_helper.grid[adj_I].iduf;
                        int adj_cell_type = field_helper.cell_type[adj_idx];
                        if (adj_iduf >= 0 && (adj_cell_type == CellType::GAS || adj_cell_type == CellType::INLET)) {
                            sum_Qs += field_helper.q[adj_idx];
                            sum_Qs_n++;
                        }
                    },
                    I, -1, 2);
                if (sum_Qs_n > 0) {
                    sum_Qs /= (T)sum_Qs_n;
                    field_helper.q[idx] = sum_Qs;
                }
                else {
                    // Logging::warn("a gost cell has no neighbor");
                    std::cout << "a gost cell has no neighbor" << std::endl;
                    field_helper.q[idx] = q_amb;
                }
            }
        };
        field_helper.iterateGridParallel(ops, 1);
    };
};

template <class T, int dim, class StorageIndex, bool XFastestSweep>
class ConvertMomentumToU : public AbstractOp {
public:
    // Inputs
    // FieldHelperSPGrid<T, dim, StorageIndex>& field_helper;
    FieldHelperDense<T, dim, StorageIndex, XFastestSweep>& field_helper;
    const T& gamma;

    // Outputs:
    inline void operator()()
    {
        // BOW_TIMER_FLAG("Convert to U");
        // Logging::info("Convert to U");
        auto ops = [&](const Vector<int, dim>& I) {
            // we dont use the getPfromQ function here as that in EOS pass
            // as it tested if the internal energy is legitmate
            // but this pass is called before fixing the total energy in the
            // projection step so the tot energy is smaller than the final value
            // leading to a temporally negative internal energy we cant tell if that's
            // wrong or not until the end of fixing total energy
            if (field_helper.grid[I].iduf >= 0) {
                int idx = field_helper.grid[I].idx;
                auto q = field_helper.q[idx];
                field_helper.rhof[idx] = q(0);
                field_helper.uf[idx] = (q.template tail<dim>() / q(0)).matrix();
            }
        };

        field_helper.iterateGridParallel(ops, 1);
    };
};

template <class T, int dim, class StorageIndex, bool XFastestSweep>
class EOS : public AbstractOp {
public:
    // Inputs
    // FieldHelperSPGrid<T, dim, StorageIndex>& field_helper;
    FieldHelperDense<T, dim, StorageIndex, XFastestSweep>& field_helper;
    const T& gamma;
    const T& P_amb;

    // Outputs:
    inline void operator()()
    {
        // BOW_TIMER_FLAG("EOS");
        // Logging::info("EOS");
        auto ops = [&](const Vector<int, dim>& I) {
            if (field_helper.grid[I].idPf >= 0) {
                auto& g = field_helper.grid[I];
                T sum_P = 0;
                int sum_P_n = 0;
                field_helper.iterateKernel(
                    [&](const Vector<int, dim>& I, const Vector<int, dim>& adj_I) {
                        auto& adj_g = field_helper.grid[adj_I];
                        if (adj_g.iduf >= 0) {
                            auto dP = ZenEulerGas::ConstitutiveModel::IdealGas::getPFromQ<T, dim>(
                                field_helper.q[adj_g.idx], gamma);
                            if (dP(1) > 0) {
                                sum_P += dP(1);
                                sum_P_n++;
                            }
                        }
                    },
                    I, -1, 1);
                field_helper.Pf[g.idx] = (sum_P_n > 0 ? sum_P / (T)sum_P_n : P_amb);
            }
        };
        field_helper.iterateGridParallel(ops, 1);
    };
};

template <class T, int dim, class StorageIndex, bool XFastestSweep>
class MarkOp : public AbstractOp {
public:
    // Inputs
    FieldHelperDense<T, dim, StorageIndex, XFastestSweep>& field_helper;

    // Outputs:
    inline void operator()()
    {
        // BOW_TIMER_FLAG("Mark cell type");
        // Logging::info("Mark cell type");
        // 0.reset
        field_helper.B_interfaces.clear();
        field_helper.Bs_interfaces.clear();
        field_helper.H_interfaces.clear();
        field_helper.Hs_interfaces.clear();
        auto reset_ids = [&](const Vector<int, dim>& I) {
            auto& g = field_helper.grid[I];
            g.iduf = -1;
            g.idPf = -1;
            g.idYf = -1;
            g.idH = -1;
            g.idus = -1;
            g.idPs = -1;
            g.idYs = -1;
        };
        field_helper.iterateGridSerial(reset_ids, field_helper.grid.ghost_layer);
        {
            // fluid info
            int active_u_node = 0;
            int active_P_node = 0;
            int active_Y_node = 0;
            // int active_H_node = 0;
            // 1.mark normal u
            auto mark_normal_idu = [&](const Vector<int, dim>& I) {
                auto& g = field_helper.grid[I];
                if (field_helper.cell_type[g.idx] == CellType::GAS)
                    field_helper.grid[I].iduf = active_u_node++;
            };
            field_helper.iterateGridSerial(mark_normal_idu);
            // 2.mark bound ghost u
            auto mark_bound_idu = [&](const Vector<int, dim>& I) {
                auto& g = field_helper.grid[I];
                if (field_helper.cell_type[g.idx] == CellType::BOUND || field_helper.cell_type[g.idx] == CellType::FREE || field_helper.cell_type[g.idx] == CellType::INLET) {
                    bool have_adj_gas = false;
                    field_helper.iterateKernel(
                        [&](const Vector<int, dim>& I, const Vector<int, dim>& adj_I) {
                            auto& adj_g = field_helper.grid[adj_I];
                            if (field_helper.cell_type[adj_g.idx] == CellType::GAS)
                                have_adj_gas = true;
                        },
                        I, -1, 2);
                    if (have_adj_gas)
                        field_helper.grid[I].iduf = active_u_node++;
                }
            };
            field_helper.iterateGridSerial(mark_bound_idu, 1);
            // 3.mark ghost u
            auto mark_ghost_idu = [&](const Vector<int, dim>& I) {
                auto& g = field_helper.grid[I];
                if (field_helper.cell_type[g.idx] == CellType::SOLID) {
                    std::cout << "found a solid cell in marking phase" << std::endl;
                    bool have_adj_gas = false;
                    field_helper.iterateKernel(
                        [&](const Vector<int, dim>& I, const Vector<int, dim>& adj_I) {
                            auto& adj_g = field_helper.grid[adj_I];
                            if (field_helper.cell_type[adj_g.idx] == CellType::GAS)
                                have_adj_gas = true;
                        },
                        I, -1, 2);
                    if (have_adj_gas)
                        field_helper.grid[I].iduf = active_u_node++;
                }
            };
            field_helper.iterateGridSerial(mark_ghost_idu);
            // 4.mark PYH
            auto mark_Ps = [&](const Vector<int, dim>& I) {
                auto& g = field_helper.grid[I];
                bool have_adj_P = false;
                bool have_adj_Y = false;
                // bool have_adj_H = false;
                field_helper.iterateKernel(
                    [&](const Vector<int, dim>& I, const Vector<int, dim>& adj_I) {
                        auto& adj_g = field_helper.grid[adj_I];
                        if (field_helper.cell_type[adj_g.idx] == CellType::GAS)
                            have_adj_P = true;
                        if (field_helper.cell_type[adj_g.idx] == CellType::BOUND || field_helper.cell_type[adj_g.idx] == CellType::FREE || field_helper.cell_type[adj_g.idx] == CellType::INLET || field_helper.cell_type[adj_g.idx] == CellType::SOLID)
                            have_adj_Y = true;
                    },
                    I, -1, 1);
                if (have_adj_P)
                    g.idPf = active_P_node++;
                if (have_adj_P && have_adj_Y)
                    g.idYf = active_Y_node++;
            };
            field_helper.iterateGridSerial(mark_Ps, 1);

            field_helper.nuf = active_u_node;
            field_helper.nPf = active_P_node;
            field_helper.nYf = active_Y_node;

            // Logging::info("active_u_node ", field_helper.nuf);
            // Logging::info("active_P_node ", field_helper.nPf);
            // Logging::info("active_Y_node ", field_helper.nYf);
        }
        // 5. mark selected interfaces
        auto mark_Interfaces = [&](const Vector<int, dim>& I) {
            auto IT = field_helper.interface_type(I);
            for (int d = 0; d < dim; d++) {
                if (IT(d) == InterfaceType::GAS_BOUND || IT(d) == InterfaceType::BOUND_GAS || IT(d) == InterfaceType::GAS_FREE || IT(d) == InterfaceType::FREE_GAS || IT(d) == InterfaceType::GAS_INLET || IT(d) == InterfaceType::INLET_GAS || IT(d) == InterfaceType::GAS_SOLID || IT(d) == InterfaceType::SOLID_GAS) {
                    T normal = ((IT(d) == InterfaceType::GAS_BOUND || IT(d) == InterfaceType::GAS_FREE || IT(d) == InterfaceType::GAS_INLET || IT(d) == InterfaceType::GAS_SOLID) ? 1 : -1);
                    auto it_mark = std::make_tuple(I, d, normal);
                    field_helper.B_interfaces.push_back(it_mark);
                }
            }
        };
        field_helper.iterateGridSerial(mark_Interfaces, 1);
    };
};

template <class T, int dim, class StorageIndex, bool XFastestSweep>
class AddSourceTermOp : public AbstractOp {
public:
    // Inputs
    FieldHelperDense<T, dim, StorageIndex, XFastestSweep>& field_helper;

    // Outputs:
    inline void operator()(const T dt)
    {
        // BOW_TIMER_FLAG("Source term");
        // Logging::info("Source term");
        // zero all none gas qs
        field_helper.iterateGridParallel([&](const Vector<int, dim>& I) {
            StorageIndex idx = field_helper.grid[I].idx;
            field_helper.q[idx] += field_helper.source[idx] * dt;
        });
    };
};
} // namespace ZenEulerGas

#endif
