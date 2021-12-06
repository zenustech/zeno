#pragma once

#include "base_integrator.h"
#include <memory>

/**
 * @class<BackEulerIntegrator>
 * @brief the Backward Euler Integrator, simulating over position only , with coupling lenght equals 3.
 */
class BackEulerIntegrator : public BaseIntegrator
{
public:
    BackEulerIntegrator() : BaseIntegrator(3) {}
    ~BackEulerIntegrator(void) {};

    int EvalElmObj(const TetAttributes& attrs,
            const std::shared_ptr<BaseForceModel>& force_model,
            const std::shared_ptr<DiricletDampingModel>& damping_model,
            const std::vector<Vec12d>& elm_states,FEM_Scaler* obj) const override;

    int EvalElmObjDeriv(const TetAttributes& attrs,
            const std::shared_ptr<BaseForceModel>& force_model,
            const std::shared_ptr<DiricletDampingModel>& damping_model,
            const std::vector<Vec12d>& elm_states,FEM_Scaler* obj,Vec12d& elm_deriv) const override;

    int EvalElmObjDerivJacobi(const TetAttributes& attrs,
            const std::shared_ptr<BaseForceModel>& force_model,
            const std::shared_ptr<DiricletDampingModel>& damping_model,
            const std::vector<Vec12d>& elm_states,FEM_Scaler* obj,Vec12d& elm_deriv,Mat12x12d& elm_H,bool enforce_spd,bool debug = false) const override;
};