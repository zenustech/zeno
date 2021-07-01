#ifndef EULER_GAS_DRIVE_BASIC_H
#define EULER_GAS_DRIVE_BASIC_H

#include "IdealGas.h"
#include "StateDense.h"
#include <Bow/Macros.h>
#include <Bow/Types.h>
#include <Bow/Utils/Timer.h>
#include <Eigen/Sparse>
#include <tbb/tbb.h>

namespace Bow {
namespace EulerGas {

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
        BOW_TIMER_FLAG("Extrapolate");
        Logging::info("Extrapolate");
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
                    Logging::warn("a gost cell has no neighbor");
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
        BOW_TIMER_FLAG("Convert to U");
        Logging::info("Convert to U");
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
        BOW_TIMER_FLAG("EOS");
        Logging::info("EOS");
        auto ops = [&](const Vector<int, dim>& I) {
            if (field_helper.grid[I].idPf >= 0) {
                auto& g = field_helper.grid[I];
                T sum_P = 0;
                int sum_P_n = 0;
                field_helper.iterateKernel(
                    [&](const Vector<int, dim>& I, const Vector<int, dim>& adj_I) {
                        auto& adj_g = field_helper.grid[adj_I];
                        if (adj_g.iduf >= 0) {
                            auto dP = Bow::ConstitutiveModel::IdealGas::getPFromQ<T, dim>(
                                field_helper.q[adj_g.idx], gamma);
                            // if (dP(1) <= 0) {
                            //     std::cout << "adj I is " << std::endl;
                            //     std::cout << adj_I.transpose() << std::endl;
                            //     std::cout << "q is " << std::endl;
                            //     std::cout << field_helper.q[adj_g.idx].transpose() <<
                            //     std::endl; std::cout << "p is " << std::endl; std::cout
                            //     << dP.transpose() << std::endl;
                            //     // BOW_ASSERT_INFO(dP(1) > 0, "negative or zero
                            //     pressure");
                            // }
                            if (dP(1) > 0) {
                                sum_P += dP(1);
                                sum_P_n++;
                            }
                        }
                    },
                    I, -1, 1);
                // BOW_ASSERT_INFO(sum_P_n > 0, "an active P node has no neighbor.");
                field_helper.Pf[g.idx] = (sum_P_n > 0 ? sum_P / (T)sum_P_n : P_amb);
            }
        };

        // field_helper.iterateGridSerial(ops, 1);
        field_helper.iterateGridParallel(ops, 1);
    };
};

template <class T, int dim, class StorageIndex, bool XFastestSweep>
class MarkOp : public AbstractOp {
public:
    // Inputs
    // FieldHelperSPGrid<T, dim, StorageIndex>& field_helper;
    FieldHelperDense<T, dim, StorageIndex, XFastestSweep>& field_helper;
    // const Array<int, dim, 1>& bbmin;
    // const Array<int, dim, 1>& bbmax;

    // Outputs:
    inline void operator()()
    {
        BOW_TIMER_FLAG("Mark cell type");
        Logging::info("Mark cell type");
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
            // solid info
            int active_us_node = 0;
            int active_Ps_node = 0;
            int active_Ys_node = 0;
            // 1.mark normal u
            auto mark_normal_idu = [&](const Vector<int, dim>& I) {
                auto& g = field_helper.grid[I];
                if (field_helper.cell_type[g.idx] == CellType::SOLID)
                    field_helper.grid[I].idus = active_us_node++;
            };
            field_helper.iterateGridSerial(mark_normal_idu);
            // 2.mark bound ghost u
            auto mark_bound_idu = [&](const Vector<int, dim>& I) {
                auto& g = field_helper.grid[I];
                if (field_helper.cell_type[g.idx] == CellType::BOUND || field_helper.cell_type[g.idx] == CellType::FREE || field_helper.cell_type[g.idx] == CellType::INLET) {
                    bool have_adj_solid = false;
                    field_helper.iterateKernel(
                        [&](const Vector<int, dim>& I, const Vector<int, dim>& adj_I) {
                            auto& adj_g = field_helper.grid[adj_I];
                            if (field_helper.cell_type[adj_g.idx] == CellType::SOLID)
                                have_adj_solid = true;
                        },
                        I, -1, 2);
                    if (have_adj_solid)
                        field_helper.grid[I].idus = active_us_node++;
                }
            };
            field_helper.iterateGridSerial(mark_bound_idu, 1);
            // 3.mark ghost u
            auto mark_ghost_idu = [&](const Vector<int, dim>& I) {
                auto& g = field_helper.grid[I];
                if (field_helper.cell_type[g.idx] == CellType::GAS) {
                    bool have_adj_solid = false;
                    field_helper.iterateKernel(
                        [&](const Vector<int, dim>& I, const Vector<int, dim>& adj_I) {
                            auto& adj_g = field_helper.grid[adj_I];
                            if (field_helper.cell_type[adj_g.idx] == CellType::SOLID)
                                have_adj_solid = true;
                        },
                        I, -1, 2);
                    if (have_adj_solid)
                        field_helper.grid[I].idus = active_us_node++;
                }
            };
            field_helper.iterateGridSerial(mark_ghost_idu);
            // 4.mark PYH
            auto mark_Ps = [&](const Vector<int, dim>& I) {
                auto& g = field_helper.grid[I];
                bool have_adj_P = false;
                bool have_adj_Y = false;
                field_helper.iterateKernel(
                    [&](const Vector<int, dim>& I, const Vector<int, dim>& adj_I) {
                        auto& adj_g = field_helper.grid[adj_I];
                        if (field_helper.cell_type[adj_g.idx] == CellType::SOLID)
                            have_adj_P = true;
                        if (field_helper.cell_type[adj_g.idx] == CellType::BOUND || field_helper.cell_type[adj_g.idx] == CellType::FREE || field_helper.cell_type[adj_g.idx] == CellType::INLET)
                            have_adj_Y = true;
                    },
                    I, -1, 1);
                if (have_adj_P)
                    g.idPs = active_Ps_node++;
                if (have_adj_P && have_adj_Y)
                    g.idYs = active_Ys_node++;
                // H marked in gas phase
            };
            field_helper.iterateGridSerial(mark_Ps, 1);

            field_helper.nus = active_us_node;
            field_helper.nPs = active_Ps_node;
            field_helper.nYs = active_Ys_node;

            Logging::debug("active_us_node ", field_helper.nus);
            Logging::debug("active_Ps_node ", field_helper.nPs);
            Logging::debug("active_Ys_node ", field_helper.nYs);
        }
        {
            // fluid info
            int active_u_node = 0;
            int active_P_node = 0;
            int active_Y_node = 0;
            int active_H_node = 0;
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
                bool have_adj_H = false;
                field_helper.iterateKernel(
                    [&](const Vector<int, dim>& I, const Vector<int, dim>& adj_I) {
                        auto& adj_g = field_helper.grid[adj_I];
                        if (field_helper.cell_type[adj_g.idx] == CellType::GAS)
                            have_adj_P = true;
                        if (field_helper.cell_type[adj_g.idx] == CellType::BOUND || field_helper.cell_type[adj_g.idx] == CellType::FREE || field_helper.cell_type[adj_g.idx] == CellType::INLET)
                            have_adj_Y = true;
                        if (field_helper.cell_type[adj_g.idx] == CellType::SOLID)
                            have_adj_H = true;
                    },
                    I, -1, 1);
                if (have_adj_P)
                    g.idPf = active_P_node++;
                if (have_adj_P && have_adj_Y)
                    g.idYf = active_Y_node++;
                if (have_adj_P && have_adj_H)
                    g.idH = active_H_node++;
            };
            field_helper.iterateGridSerial(mark_Ps, 1);

            field_helper.nuf = active_u_node;
            field_helper.nPf = active_P_node;
            field_helper.nYf = active_Y_node;
            field_helper.nHf = active_H_node;

            Logging::debug("active_u_node ", field_helper.nuf);
            Logging::debug("active_P_node ", field_helper.nPf);
            Logging::debug("active_Y_node ", field_helper.nYf);
            Logging::debug("active_H_node ", field_helper.nHf);
        }
        // 5. mark selected interfaces
        auto mark_Interfaces = [&](const Vector<int, dim>& I) {
            auto IT = field_helper.interface_type(I);
            for (int d = 0; d < dim; d++) {
                if (IT(d) == InterfaceType::GAS_BOUND || IT(d) == InterfaceType::BOUND_GAS || IT(d) == InterfaceType::GAS_FREE || IT(d) == InterfaceType::FREE_GAS || IT(d) == InterfaceType::GAS_INLET || IT(d) == InterfaceType::INLET_GAS) {
                    T normal = ((IT(d) == InterfaceType::GAS_BOUND || IT(d) == InterfaceType::GAS_FREE || IT(d) == InterfaceType::GAS_INLET)
                            ? 1
                            : -1);
                    auto it_mark = std::make_tuple(I, d, normal);
                    field_helper.B_interfaces.push_back(it_mark);
                }
                else if (IT(d) == InterfaceType::SOLID_BOUND || IT(d) == InterfaceType::BOUND_SOLID || IT(d) == InterfaceType::SOLID_FREE || IT(d) == InterfaceType::FREE_SOLID || IT(d) == InterfaceType::SOLID_INLET || IT(d) == InterfaceType::INLET_SOLID) {
                    T normal = ((IT(d) == InterfaceType::SOLID_BOUND || IT(d) == InterfaceType::SOLID_FREE || IT(d) == InterfaceType::SOLID_INLET)
                            ? 1
                            : -1);
                    auto it_mark = std::make_tuple(I, d, normal);
                    field_helper.Bs_interfaces.push_back(it_mark);
                }
                else if (IT(d) == InterfaceType::GAS_SOLID || IT(d) == InterfaceType::SOLID_GAS) {
                    T normal_gas = (IT(d) == InterfaceType::GAS_SOLID ? 1 : -1);
                    T normal_solid = -normal_gas;
                    auto it_mark_gas = std::make_tuple(I, d, normal_gas);
                    auto it_mark_solid = std::make_tuple(I, d, normal_solid);
                    field_helper.H_interfaces.push_back(it_mark_gas);
                    field_helper.Hs_interfaces.push_back(it_mark_solid);
                }
            }
        };
        field_helper.iterateGridSerial(mark_Interfaces, 1);
        BOW_ASSERT_INFO(field_helper.H_interfaces.size() == field_helper.Hs_interfaces.size(),
            "interface number not consistent for two phases");
        // 6. mark moving Ys override to compensate the cases when the solid grid is
        // moving away from the bound if a solid grid is not 'entering' one
        // interface, then no constraint should be applied here to reach this
        // effect, the RHS(the divergence/penalty term) here should be modified to
        // cancel out the RHS_Ys then after solving the velocity is unchanged
        // equivlantly saying, if a Ys interface is b-s, normal (-1), and the solid
        // velocity is +, then mark this Ys override with the same velocity as that
        // solid grid if a Ys interface is s-b, normal (+1), and the solid velocity
        // is -, then do the same thing
        if (false) {
            field_helper.moving_Ys_interfaces_override.clear();
            for (const auto& it_mark : field_helper.Bs_interfaces) {
                auto [I, d, normal] = it_mark;
                if (normal > 0) {
                    auto I_solid = I;
                    I_solid(d) -= 1;
                    int idx = field_helper.grid[I_solid].idx;
                    T solid_vel = field_helper.us[idx](d);
                    if (solid_vel < 0) {
                        auto it_mark_moving_solid = std::make_tuple(I, d, normal, solid_vel);
                        field_helper.moving_Ys_interfaces_override.push_back(
                            it_mark_moving_solid);
                    }
                }
                else if (normal < 0) {
                    auto I_solid = I;
                    int idx = field_helper.grid[I_solid].idx;
                    T solid_vel = field_helper.us[idx](d);
                    if (solid_vel > 0) {
                        auto it_mark_moving_solid = std::make_tuple(I, d, normal, solid_vel);
                        field_helper.moving_Ys_interfaces_override.push_back(
                            it_mark_moving_solid);
                    }
                }
            }
        }
    };
};

template <class T, int dim, class StorageIndex, bool XFastestSweep>
class AddSourceTermOp : public AbstractOp {
public:
    // Inputs
    // FieldHelperSPGrid<T, dim, StorageIndex>& field_helper;
    FieldHelperDense<T, dim, StorageIndex, XFastestSweep>& field_helper;

    // Outputs:
    inline void operator()(const T dt)
    {
        BOW_TIMER_FLAG("Source term");
        Logging::info("Source term");
        // zero all none gas qs
        field_helper.iterateGridParallel([&](const Vector<int, dim>& I) {
            StorageIndex idx = field_helper.grid[I].idx;
            field_helper.q[idx] += field_helper.source[idx] * dt;
        });
    };
};

}
} // namespace Bow::EulerGas

#endif
