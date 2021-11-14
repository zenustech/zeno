#include <quasi_static_solver.h>

int QuasiStaticSolver::EvalElmObj(const TetAttributes attrs,
    const std::shared_ptr<BaseForceModel>& force_model,
    const std::shared_ptr<DiricletDampingModel>& damping_model,
    const std::vector<Vec12d>& elm_states,FEM_Scaler* elm_obj) const {
        Vec12d u0 = elm_states[0];
        FEM_Scaler vol = attrs._volume;
        FEM_Scaler m = vol * attrs._density / 4;

        Mat3x3d F;
        FEM_Scaler psiE;
        ComputeDeformationGradient(attrs._Minv,u0,F);

        if(dynamic_cast<ElasticModel*>(force_model.get())){
            auto model = dynamic_cast<ElasticModel*>(force_model.get());
            model->ComputePhi(attrs.emp,F,psiE);
        }else{
            throw std::runtime_error("PLASTIC MODEL NOT DEFINED");
        }

        Vec12d gravity_force = _gravity.replicate(4,1) * m;
        *elm_obj = psiE * vol - u0.dot(gravity_force);

        return 0;
}

int QuasiStaticSolver::EvalElmObjDeriv(const TetAttributes attrs,
    const std::shared_ptr<BaseForceModel>& force_model,
    const std::shared_ptr<DiricletDampingModel>& damping_model,
    const std::vector<Vec12d>& elm_states,FEM_Scaler* elm_obj,Vec12d& elm_deriv) const {
        Vec12d u0 = elm_states[0];
        FEM_Scaler vol = attrs._volume;
        FEM_Scaler m = vol * attrs._density / 4;

        Mat3x3d F;
        FEM_Scaler psiE;
        Vec9d dpsiE;
        ComputeDeformationGradient(attrs._Minv,u0,F);

        if(dynamic_cast<ElasticModel*>(force_model.get())){
            auto model = dynamic_cast<ElasticModel*>(force_model.get());
            model->ComputePhiDeriv(attrs.emp,F,psiE,dpsiE);
        }else{
            throw std::runtime_error("PLASTIC MODEL NOT DEFINED");
        }

        const Mat9x12d& dFdX = attrs._dFdX;

        Vec12d gravity_force = _gravity.replicate(4,1) * m;
        *elm_obj = psiE * vol - u0.dot(gravity_force);
        elm_deriv = vol * dFdX.transpose() * dpsiE - gravity_force;

        return 0;
}

int QuasiStaticSolver::EvalElmObjDerivJacobi(const TetAttributes attrs,
    const std::shared_ptr<BaseForceModel>& force_model,
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

        if(dynamic_cast<ElasticModel*>(force_model.get())){
            auto model = dynamic_cast<ElasticModel*>(force_model.get());
            model->ComputePhiDerivHessian(attrs.emp,F,psiE,dpsiE,ddpsiE,enforce_spd);
        }else{
            throw std::runtime_error("PLASTIC MODEL NOT DEFINED");
        }

        const Mat9x12d& dFdX = attrs._dFdX;

        Vec12d gravity_force = _gravity.replicate(4,1) * m;
        *elm_obj = psiE * vol - u0.dot(gravity_force);
        elm_deriv = vol * dFdX.transpose() * dpsiE - gravity_force;

        elm_H = vol * dFdX.transpose() * ddpsiE * dFdX;

        return 0;
}