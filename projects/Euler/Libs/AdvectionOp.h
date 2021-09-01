#ifndef EULER_GAS_DRIVE_ADVECTION_H
#define EULER_GAS_DRIVE_ADVECTION_H

#include "BasicOp.h"
#include "IdealGas.h"
#include "StateDense.h"
#include "TVDRK.h"
#include "WENO.h"
#include "Types.h"
#include <Eigen/Sparse>
#include <tbb/tbb.h>

namespace ZenEulerGas {

template <class T, class DerivedV, class DerivedE>
DerivedV mixed_bc_flux(std::array<DerivedV, 4>& Q, std::array<T, 4>& U,
    const std::array<int, 4>& cell_types, DerivedE eps,
    T i_frac = 0)
{
    // 1. For the velocity/cell-type stencil, fill in the value with the correct
    // cell type
    // 2. For qs, fill the corresponding value for the gas cell only, others just
    // leave as zero
    // 3. Calculate the eps and passable ratio beforehand

    using ZenEulerGas::CellType;
    DerivedV flux = (T)0 * Q[0];
    if (cell_types[1] == CellType::GAS && cell_types[2] == CellType::GAS) {
        // std::cout << "gas-gas" << std::endl;
        // 1,2 are gas cells
        // check the types of 0 and 3, extrapolate if needed
        if (cell_types[0] != CellType::GAS) {
            U[0] = U[1];
            Q[0] = Q[1];
        }
        if (cell_types[3] != CellType::GAS) {
            U[3] = U[2];
            Q[3] = Q[2];
        }
        flux = ZenEulerGas::Math::RPSolver::WENO2_LLF(U, Q, eps);
    }
    else if (cell_types[1] == CellType::GAS && cell_types[2] == CellType::FREE) {
        // std::cout << "gas-free" << std::endl;
        // check the types of 0, extrapolate if needed
        if (cell_types[0] != CellType::GAS) {
            U[0] = U[1];
            Q[0] = Q[1];
        }
        // all the free is extrapolated
        Q[2] = Q[1];
        Q[3] = Q[1];
        U[2] = U[1];
        U[3] = U[1];
        flux = ZenEulerGas::Math::RPSolver::WENO2_LLF(U, Q, eps);
    }
    else if (cell_types[1] == CellType::FREE && cell_types[2] == CellType::GAS) {
        // std::cout << "free-gas" << std::endl;
        // check the types of 3, extrapolate if needed
        if (cell_types[3] != CellType::GAS) {
            U[3] = U[2];
            Q[3] = Q[2];
        }
        // all the free is extrapolated
        Q[1] = Q[2];
        Q[0] = Q[2];
        U[1] = U[2];
        U[0] = U[2];
        flux = ZenEulerGas::Math::RPSolver::WENO2_LLF(U, Q, eps);
    }
    else if (cell_types[1] == CellType::GAS && cell_types[2] == CellType::INLET) {
        // check the types of 0, extrapolate if needed
        if (cell_types[0] != CellType::GAS) {
            U[0] = U[1];
            Q[0] = Q[1];
        }
        // 2 is inlet, and user given
        // extrapolate 3
        Q[3] = Q[2];
        U[3] = U[2];
        flux = ZenEulerGas::Math::RPSolver::WENO2_LLF(U, Q, eps);
    }
    else if (cell_types[1] == CellType::INLET && cell_types[2] == CellType::GAS) {
        // check the types of 3, extrapolate if needed
        if (cell_types[3] != CellType::GAS) {
            U[3] = U[2];
            Q[3] = Q[2];
        }
        // 1 is inlet, and user given
        // extrapolate 0
        Q[0] = Q[1];
        U[0] = U[1];
        flux = ZenEulerGas::Math::RPSolver::WENO2_LLF(U, Q, eps);
    }
    else if (cell_types[1] == CellType::GAS && (cell_types[2] == CellType::SOLID || cell_types[2] == CellType::BOUND)) {
        // 1 is gas, 2,3 will be ghost
        // check the type of 0, extrapolate if needed
        if (cell_types[0] != CellType::GAS) {
            U[0] = U[1];
            Q[0] = Q[1];
        }
        // create reflective/passable stencil
        std::array<DerivedV, 4> Qri = Q;
        std::array<T, 4> Ur = U, Ui = U;
        // modify reflective stencil
        T u_interface = U[2];
        Ur[2] = (T)2 * u_interface - U[1];
        Ur[3] = (T)2 * u_interface - U[0];
        Qri[2] = Q[1];
        Qri[3] = Q[0];
        // modify passable stencil
        Ui[2] = U[1];
        Ui[3] = U[0];
        // blend
        flux = ((T)1 - i_frac) * ZenEulerGas::Math::RPSolver::WENO2_LLF(Ur, Qri, eps) + i_frac * ZenEulerGas::Math::RPSolver::WENO2_LLF(Ui, Qri, eps);
    }
    else if ((cell_types[1] == CellType::SOLID || cell_types[1] == CellType::BOUND) && cell_types[2] == CellType::GAS) {
        // 2 is gas, 0,1 will be ghost
        // check the type of 3, extrapolate if needed
        if (cell_types[3] != CellType::GAS) {
            U[3] = U[2];
            Q[3] = Q[2];
        }
        // create reflective/passable stencil
        std::array<DerivedV, 4> Qri = Q;
        std::array<T, 4> Ur = U, Ui = U;
        // modify reflective stencil
        T u_interface = U[1];
        Ur[0] = (T)2 * u_interface - U[3];
        Ur[1] = (T)2 * u_interface - U[2];
        Qri[0] = Q[3];
        Qri[1] = Q[2];
        // modify passable stencil
        Ui[0] = U[3];
        Ui[1] = U[2];
        // blend
        flux = ((T)1 - i_frac) * ZenEulerGas::Math::RPSolver::WENO2_LLF(Ur, Qri, eps) + i_frac * ZenEulerGas::Math::RPSolver::WENO2_LLF(Ui, Qri, eps);
    }
    return flux;
}

template <class T, int dim, class StorageIndex, bool XFastestSweep>
class AdvectionOp : public AbstractOp {
public:
    // Inputs
    FieldHelperDense<T, dim, StorageIndex, XFastestSweep>& field_helper;
    const T& inv_dx;
    const T& lowest_rho;
    const T& lowest_int_e_by_rho;

    Array<T, dim + 2, dim> eps;

    // Helper Functions:

    // Outputs:
    inline void operator()(const T& dt, const int& substep)
    {
        // BOW_TIMER_FLAG("Flux update");
        // Logging::info("Flux update");
        eps = 1e-6 * Array<T, dim + 2, dim>::Ones();

        auto get_eps = [&](const Vector<int, dim>& I) {
            if (field_helper.cell_type[field_helper.grid[I].idx] == CellType::GAS) {
                int idx = field_helper.grid[I].idx;
                Vector<T, dim> u = field_helper.uf[idx];
                Array<T, dim + 2, 1> q = field_helper.q[idx];
                for (int d = 0; d < dim; d++)
                    eps.col(d) = eps.col(d).max(1e-6 * u(d) * u(d) * q * q);
            }
        };

        auto get_flux = [&](const Vector<int, dim>& I) {
            if (field_helper.grid[I].iduf >= 0) {
                int idx = field_helper.grid[I].idx;
                Array<T, dim + 2, dim> temp_flux = Array<T, dim + 2, dim>::Zero();
                for (int d = 0; d < dim; d++) {
                    Vector<int, dim> pos1 = I;
                    pos1(d) = pos1(d) - 2;
                    Vector<int, dim> pos2 = I;
                    pos2(d) = pos2(d) - 1;
                    Vector<int, dim> pos3 = I;
                    Vector<int, dim> pos4 = I;
                    pos4(d) = pos4(d) + 1;

                    // prevent sampling pos outof bbox
                    int IT = field_helper.interface_type(I)(d);
                    bool active_interface = field_helper.cell_type[field_helper.grid[pos2].idx] == CellType::GAS || field_helper.cell_type[field_helper.grid[pos3].idx] == CellType::GAS;
                    if (!active_interface)
                        continue;

                    std::array<int, 4> IDX{
                        { field_helper.grid[pos1].idx, field_helper.grid[pos2].idx,
                            field_helper.grid[pos3].idx, field_helper.grid[pos4].idx }
                    };
                    // q value is all obtained from gas field
                    std::array<Array<T, dim + 2, 1>, 4> Q{
                        { field_helper.q[IDX[0]], field_helper.q[IDX[1]],
                            field_helper.q[IDX[2]], field_helper.q[IDX[3]] }
                    };
                    std::array<int, 4> cell_types{
                        { field_helper.cell_type[IDX[0]], field_helper.cell_type[IDX[1]],
                            field_helper.cell_type[IDX[2]], field_helper.cell_type[IDX[3]] }
                    };
                    // U is obtained from two phases respectively
                    std::array<T, 4> U;
                    // std::cout << "hello" << std::endl;
                    for (int i = 0; i < 4; i++) {
                        if (cell_types[i] == CellType::GAS || cell_types[i] == CellType::FREE || cell_types[i] == CellType::INLET)
                            U[i] = field_helper.uf[IDX[i]](d);
                        else if (cell_types[i] == CellType::SOLID) {
                            // std::cout << "correctly got interface vel ";
                            U[i] = field_helper.us[IDX[i]](d);
                            // std::cout << U[i] << std::endl;
                        }
                        else if (cell_types[i] == CellType::BOUND)
                            U[i] = 0;
                        else
                            U[i] = 0;
                    }
                    T i_frac = 0;
                    // apply stencil
                    temp_flux.col(d) = mixed_bc_flux(Q, U, cell_types, eps.col(d), i_frac);
                }
                field_helper.flux[idx] = temp_flux;
            }
        };

        auto flux_based_update = [&](const Vector<int, dim>& I) {
            if (field_helper.cell_type[field_helper.grid[I].idx] == CellType::GAS) {
                auto RK_coeffs = ZenEulerGas::Math::TimeIntegration::TVDRK3<T, int>(substep);
                int idx = field_helper.grid[I].idx;
                Array<T, dim + 2, 1> div_flux = Array<T, dim + 2, 1>::Zero();
                // get flux div by Mac kernel
                for (int d = 0; d < dim; d++) {
                    Vector<int, dim> pos_r = I;
                    pos_r(d) = pos_r(d) + 1;
                    int idx_r = field_helper.grid[pos_r].idx;
                    div_flux += inv_dx * (field_helper.flux[idx_r].col(d) - field_helper.flux[idx].col(d));
                }
                field_helper.q[idx] = RK_coeffs(0) * field_helper.q_backup[idx] + RK_coeffs(1) * field_helper.q[idx] - RK_coeffs(2) * dt * div_flux;

                // clamp the density and the internal energy
                // 1. clamp the density smaller than a threshold value (if needed)
                if (field_helper.q[idx](0) < lowest_rho) {
                    // Logging::warn("clamped density");
                    // fix the artifical internal enerygy increase (kinematic energy
                    // decrease)
                    if (field_helper.q[idx](0) > 0) {
                        T artificial_kinematic_energy_decrease = 0.5 * (field_helper.q[idx].template tail<dim>() * field_helper.q[idx].template tail<dim>()).sum() * ((T)1 / field_helper.q[idx](0) - (T)1 / lowest_rho);
                        field_helper.q[idx](1) -= artificial_kinematic_energy_decrease;
                    }
                    field_helper.q[idx](0) = lowest_rho;
                }
                // 2. calculate the internal energy, devide by rho(to get temperature),
                // if lower than a value, clamp to it the corresponding d_int_e should
                // be added to the total energy
                T int_E = (field_helper.q[idx](1) - 0.5 * (field_helper.q[idx].template tail<dim>() * field_helper.q[idx].template tail<dim>()).sum() / field_helper.q[idx](0));
                if (int_E - field_helper.q[idx](0) * lowest_int_e_by_rho < 0) {
                    // Logging::warn("clamped energy");
                    T delta_E = field_helper.q[idx](0) * lowest_int_e_by_rho - int_E;
                    field_helper.q[idx](1) += delta_E;
                }
            }
        };

        field_helper.iterateGridSerial(get_eps);
        field_helper.iterateGridParallel(get_flux, 1);
        {
            // calculate flux for moving bound
            for (const auto& it_mark : field_helper.moving_Yf_interfaces_override) {
                auto [I, d, normal, inteface_normal_vel] = it_mark;
                Vector<int, dim> pos1 = I;
                pos1(d) = pos1(d) - 2;
                Vector<int, dim> pos2 = I;
                pos2(d) = pos2(d) - 1;
                Vector<int, dim> pos3 = I;
                Vector<int, dim> pos4 = I;
                pos4(d) = pos4(d) + 1;

                // only calculate at the interface of moving bound/solid
                // in other cases, i.e., solid takes the original gas cell, then this
                // flux is not needed
                int IT = field_helper.interface_type(I)(d);
                bool active_interface = (IT == InterfaceType::GAS_BOUND) || (IT == InterfaceType::BOUND_GAS);
                if (!active_interface)
                    continue;
                // assert the normal is consistent with the celltypes here
                assertm((IT == InterfaceType::GAS_BOUND) == (normal > 0),
                    "wrong marked moving interface override")
                    assertm((IT == InterfaceType::BOUND_GAS) == (normal < 0),
                        "wrong marked moving interface override")

                        std::array<int, 4>
                            IDX{
                                { field_helper.grid[pos1].idx, field_helper.grid[pos2].idx,
                                    field_helper.grid[pos3].idx, field_helper.grid[pos4].idx }
                            };
                std::array<Array<T, dim + 2, 1>, 4> Q{
                    { field_helper.q[IDX[0]], field_helper.q[IDX[1]],
                        field_helper.q[IDX[2]], field_helper.q[IDX[3]] }
                };
                std::array<int, 4> cell_types{
                    { field_helper.cell_type[IDX[0]], field_helper.cell_type[IDX[1]],
                        field_helper.cell_type[IDX[2]], field_helper.cell_type[IDX[3]] }
                };
                // U is obtained from two phases respectively
                std::array<T, 4> U;
                for (int i = 0; i < 4; i++) {
                    if (cell_types[i] == CellType::GAS)
                        U[i] = field_helper.uf[IDX[i]](d);
                    else if (cell_types[i] == CellType::SOLID)
                        U[i] = field_helper.us[IDX[i]](d);
                    else if (cell_types[i] == CellType::BOUND)
                        U[i] = 0;
                    else
                        U[i] = 0;
                }
                // fix velocity by override
                if (normal > 0) {
                    // gas on the left, all right cells are ghost
                    U[2] = inteface_normal_vel;
                    U[3] = inteface_normal_vel;
                }
                else {
                    // gas on the right, vise vesa
                    U[0] = inteface_normal_vel;
                    U[1] = inteface_normal_vel;
                }
                // pure-reflective
                // apply stencil
                int idx = field_helper.grid[I].idx;
                Array<T, dim + 2, 1> temp_flux = mixed_bc_flux(Q, U, cell_types, eps.col(d), (T)0);
                field_helper.flux[idx].col(d) = temp_flux;
            }
        }
        field_helper.iterateGridParallel(flux_based_update);
    };
};
} // namespace ZenEulerGas

#endif
