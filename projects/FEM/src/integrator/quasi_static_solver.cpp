#include <quasi_static_solver.h>

int QuasiStaticSolver::EvalElmObj(const TetAttributes attrs,
    const std::shared_ptr<MuscleForceModel>& force_model,
    const std::shared_ptr<DiricletDampingModel>& damping_model,
    const std::vector<Vec12d>& elm_states,FEM_Scaler* elm_obj) const {
        Vec12d u0 = elm_states[0];
        FEM_Scaler vol = attrs._volume;
        FEM_Scaler m = vol * attrs._density / 4;

        Mat3x3d F;
        FEM_Scaler psiE;
        ComputeDeformationGradient(attrs._Minv,u0,F);
        force_model->ComputePhi(attrs._activation,
            attrs._fiberWeight,
            attrs._fiberOrient,
            attrs._E,
            attrs._nu,
            F,psiE);

        Vec12d gravity_force = _gravity.replicate(4,1) * m;
        *elm_obj = psiE * vol - u0.dot(gravity_force);

        return 0;
}

int QuasiStaticSolver::EvalElmObjDeriv(const TetAttributes attrs,
    const std::shared_ptr<MuscleForceModel>& force_model,
    const std::shared_ptr<DiricletDampingModel>& damping_model,
    const std::vector<Vec12d>& elm_states,FEM_Scaler* elm_obj,Vec12d& elm_deriv) const {
        Vec12d u0 = elm_states[0];
        FEM_Scaler vol = attrs._volume;
        FEM_Scaler m = vol * attrs._density / 4;

        Mat3x3d F;
        FEM_Scaler psiE;
        Vec9d dpsiE;
        ComputeDeformationGradient(attrs._Minv,u0,F);
        force_model->ComputePhiDeriv(attrs._activation,
            attrs._fiberWeight,
            attrs._fiberOrient,
            attrs._E,
            attrs._nu,
            F,psiE,dpsiE);

        const Mat9x12d& dFdX = attrs._dFdX;

        Vec12d gravity_force = _gravity.replicate(4,1) * m;
        *elm_obj = psiE * vol - u0.dot(gravity_force);
        elm_deriv = vol * dFdX.transpose() * dpsiE - gravity_force;

        return 0;
}

int QuasiStaticSolver::EvalElmObjDerivJacobi(const TetAttributes attrs,
    const std::shared_ptr<MuscleForceModel>& force_model,
    const std::shared_ptr<DiricletDampingModel>& damping_model,
    const std::vector<Vec12d>& elm_states,FEM_Scaler* elm_obj,Vec12d& elm_deriv,Mat12x12d& elm_H,bool enforce_spd) const {
        Vec12d u0 = elm_states[0];
        FEM_Scaler vol = attrs._volume;
        FEM_Scaler m = vol * attrs._density / 4;

        Mat3x3d F;
        FEM_Scaler psiE;
        Vec9d dpsiE;
        Mat9x9d ddpsiE;
        ComputeDeformationGradient(attrs._Minv,u0,F);
        force_model->ComputePhiDerivHessian(attrs._activation,
            attrs._fiberWeight,
            attrs._fiberOrient,
            attrs._E,
            attrs._nu,
            F,psiE,dpsiE,ddpsiE,enforce_spd);

        const Mat9x12d& dFdX = attrs._dFdX;

        Vec12d gravity_force = _gravity.replicate(4,1) * m;
        *elm_obj = psiE * vol - u0.dot(gravity_force);
        elm_deriv = vol * dFdX.transpose() * dpsiE - gravity_force;

        elm_H = vol * dFdX.transpose() * ddpsiE * dFdX;

        return 0;
}