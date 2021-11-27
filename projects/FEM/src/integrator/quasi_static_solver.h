#pragma once
#include <base_integrator.h>

class QuasiStaticSolver : public BaseIntegrator {
public:
    QuasiStaticSolver() : BaseIntegrator(1) {}
    ~QuasiStaticSolver(void) {};

    int EvalElmObj(const TetAttributes tet_attribs,
            const std::shared_ptr<MuscleForceModel>& force_model,
            const std::shared_ptr<DiricletDampingModel>& damping_model,
            const std::vector<Vec12d>& elm_states,FEM_Scaler* obj) const override;

    int EvalElmObjDeriv(const TetAttributes tet_attribs,
            const std::shared_ptr<MuscleForceModel>& force_model,
            const std::shared_ptr<DiricletDampingModel>& damping_model,
            const std::vector<Vec12d>& elm_states,FEM_Scaler* obj,Vec12d& elm_deriv) const override;

    int EvalElmObjDerivJacobi(const TetAttributes tet_attribs,
            const std::shared_ptr<MuscleForceModel>& force_model,
            const std::shared_ptr<DiricletDampingModel>& damping_model,
            const std::vector<Vec12d>& elm_states,FEM_Scaler* obj,Vec12d& elm_deriv,Mat12x12d& elm_H,bool enforce_spd) const override;
};