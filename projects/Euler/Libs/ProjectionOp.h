#ifndef EULER_GAS_DRIVE_PROJECTION_H
#define EULER_GAS_DRIVE_PROJECTION_H

#include "BasicOp.h"
#include "IdealGas.h"
#include "StateDense.h"
#include "TVDRK.h"
#include "Types.h"
#include "Types.h"

#include <Eigen/Sparse>
#include <tbb/tbb.h>

namespace ZenEulerGas {

template <class T, int dim, class StorageIndex, bool XFastestSweep>
class PostFixMuOp : public AbstractOp {
public:
    // Inputs
    FieldHelperDense<T, dim, StorageIndex, XFastestSweep>& field_helper;
    const T& inv_dx;

    // Outputs:

    inline void operator()(const T& dt, const int& substep)
    {
        // BOW_TIMER_FLAG("Fix Momentum");
        // Logging::info("Fix Momentum");
        auto flux_based_update = [&, dt](const Vector<int, dim>& I) {
            if (field_helper.cell_type[field_helper.grid[I].idx] == CellType::GAS) {
                auto RK_coeffs = ZenEulerGas::Math::TimeIntegration::TVDRK3<T, int>(substep);
                field_helper.q[field_helper.grid[I].idx].template tail<dim>() -= dt * RK_coeffs(2) * field_helper.cellPressureGrad(I, inv_dx).array();
            }
        };
        field_helper.iterateGridParallel(flux_based_update);
    };
};

template <class T, int dim, class StorageIndex, bool XFastestSweep>
class PostFixEOp : public AbstractOp {
public:
    // Inputs
    FieldHelperDense<T, dim, StorageIndex, XFastestSweep>& field_helper;
    const T& inv_dx;
    const T& lowest_int_e_by_rho;

    // Outputs:

    inline void operator()(const T& dt, const int& substep)
    {
        // BOW_TIMER_FLAG("Fix energy");
        // Logging::info("Fix energy");
        auto flux_based_update = [&](const Vector<int, dim>& I) {
            if (field_helper.cell_type[field_helper.grid[I].idx] == CellType::GAS) {
                auto RK_coeffs = ZenEulerGas::Math::TimeIntegration::TVDRK3<T, int>(substep);
                int idx = field_helper.grid[I].idx;
                Vector<T, dim> PU_div = field_helper.cellPressureXVelocityGrad(I, inv_dx);
                T sum_PU_div = PU_div.sum();
                field_helper.q[idx](1) -= dt * RK_coeffs(2) * sum_PU_div;
                // clamp the internal energy
                // 1. calculate the internal energy, devide by rho(to get temperature),
                // if lower than a value, clamp to it the corresponding d_int_e should
                // be added to the total energy
                T int_E = (field_helper.q[idx](1) - 0.5 * (field_helper.q[idx].template tail<dim>() * field_helper.q[idx].template tail<dim>()).sum() / field_helper.q[idx](0));
                if (int_E - field_helper.q[idx](0) * lowest_int_e_by_rho < 0) {
                    // Logging::warn("clamped energy");
                    std::cout << "clamped energy" << std::endl;
                    T delta_E = field_helper.q[idx](0) * lowest_int_e_by_rho - int_E;
                    field_helper.q[idx](1) += delta_E;
                }
            }
        };

        field_helper.iterateGridParallel(flux_based_update);
    };
};
} // namespace ZenEulerGas

#endif
