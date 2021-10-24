#include "backward_euler_integrator.h"
#include <iostream>
#include <ctime>
#include <numeric>

#include "diriclet_damping.h"

int BackEulerIntegrator::EvalElmObj(const TetAttributes attrs,
            const std::shared_ptr<BaseForceModel>& force_model,
            const std::shared_ptr<DiricletDampingModel>& damping_model,
            const std::vector<Vec12d>& elm_states,FEM_Scaler* elm_obj) const {
    Vec12d u2 = elm_states[2];
    Vec12d u1 = elm_states[1];
    Vec12d u0 = elm_states[0];
    FEM_Scaler h = _dt;
    Vec12d v2 = (u2 - u1)/h;

    FEM_Scaler PhiI,PsiE,PsiD;

    FEM_Scaler vol = attrs._volume;
    FEM_Scaler m = vol * attrs._density / 4;  // nodal mass
   
    // inertial term
    Vec12d y = u2 - 2*u1 + u0 - h*h*_gravity.replicate(4,1); 
    PhiI = y.squaredNorm() * m / 2 / h;

    Mat3x3d F,L;
    ComputeDeformationGradient(attrs._Minv,u2,F);
    ComputeDeformationGradient(attrs._Minv,v2,L);
    force_model->ComputePhi(attrs._activation,
        attrs._fiberWeight,
        attrs._fiberOrient,
        attrs._E,
        attrs._nu,
        F,
        PsiE);       
    damping_model->ComputePhi(attrs._d,L,PsiD);

    *elm_obj = PhiI + h*PsiE*vol + h*h*PsiD*vol;

    return 0;
}

int BackEulerIntegrator::EvalElmObjDeriv(const TetAttributes attrs,
            const std::shared_ptr<BaseForceModel>& force_model,
            const std::shared_ptr<DiricletDampingModel>& damping_model,
            const std::vector<Vec12d>& elm_states,FEM_Scaler* elm_obj,Vec12d& elm_deriv) const {
    Vec12d u2 = elm_states[2];
    Vec12d u1 = elm_states[1];
    Vec12d u0 = elm_states[0];
    FEM_Scaler h = _dt;
    Vec12d v2 = (u2 - u1)/h;

    FEM_Scaler PhiI,PsiE,PsiD;
    Vec9d dPsiE,dPsiD;

    FEM_Scaler vol = attrs._volume;
    FEM_Scaler m = vol * attrs._density / 4;  // nodal mass

    Mat3x3d F,L;
    ComputeDeformationGradient(attrs._Minv,u2,F);
    ComputeDeformationGradient(attrs._Minv,v2,L);

    force_model->ComputePhiDeriv(attrs._activation,
        attrs._fiberWeight,
        attrs._fiberOrient,
        attrs._E,
        attrs._nu,
        F,
        PsiE,dPsiE);
    damping_model->ComputePhiDeriv(attrs._d,L,PsiD,dPsiD);

    Vec12d y = u2 - 2*u1 + u0 - h*h*_gravity.replicate(4,1);
    PhiI = y.squaredNorm() * m / 2 / h;

    const Mat9x12d& dFdX = attrs._dFdX;

    *elm_obj = PhiI + h*PsiE*vol + h*h*PsiD*vol;   

    elm_deriv = m*y/h + h * dFdX.transpose() * dPsiE*vol + h * dFdX.transpose() * dPsiD * vol;

    return 0;                
}

int BackEulerIntegrator::EvalElmObjDerivJacobi(const TetAttributes attrs,
        const std::shared_ptr<BaseForceModel>& force_model,
        const std::shared_ptr<DiricletDampingModel>& damping_model,
        const std::vector<Vec12d>& elm_states,
        FEM_Scaler* elm_obj,Vec12d& elm_deriv,Mat12x12d& elm_H,bool filtering) const{
    Vec12d u2 = elm_states[2];
    Vec12d u1 = elm_states[1];
    Vec12d u0 = elm_states[0];
    FEM_Scaler h = _dt;
    Vec12d v2 = (u2 - u1)/h;

    FEM_Scaler PhiI,PsiE,PsiD;
    Vec9d dPsiE,dPsiD;
    Mat9x9d ddPsiE,ddPsiD;
         
    FEM_Scaler vol = attrs._volume;
    FEM_Scaler m = vol * attrs._density / 4;  // nodal mass

    Mat3x3d F,L;
    ComputeDeformationGradient(attrs._Minv,u2,F);
    ComputeDeformationGradient(attrs._Minv,v2,L);

    force_model->ComputePhiDerivHessian(attrs._activation,
        attrs._fiberWeight,
        attrs._fiberOrient,
        attrs._E,
        attrs._nu,
        F,
        PsiE,dPsiE,ddPsiE,filtering);
    damping_model->ComputePhiDerivHessian(attrs._d,L,PsiD,dPsiD,ddPsiD);

    Vec12d y = u2 - 2*u1 + u0 - h*h*_gravity.replicate(4,1);
    PhiI = y.squaredNorm() * m / 2 / h;
    *elm_obj = PhiI + h * PsiE * vol + h * h * PsiD * vol;   

    const Mat9x12d& dFdX = attrs._dFdX;
    elm_deriv = m * y / h + dFdX.transpose() * (h*dPsiE + h*dPsiD) * vol;

    elm_H = Mat12x12d::Identity() * m / h + vol * dFdX.transpose() * (h*ddPsiE + ddPsiD) * dFdX;

    return 0;
}