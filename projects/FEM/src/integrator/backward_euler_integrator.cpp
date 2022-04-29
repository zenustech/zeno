#include "backward_euler_integrator.h"
#include <iostream>
#include <ctime>
#include <numeric>

#include <stable_isotropic_NH.h>
#include <stable_Stvk.h>

#include "diriclet_damping.h"

int BackEulerIntegrator::EvalElmObj(const TetAttributes& attrs,
            const std::shared_ptr<BaseForceModel>& force_model,
            const std::shared_ptr<DiricletDampingModel>& damping_model,
            const std::vector<Vec12d>& elm_states,FEM_Scaler* elm_obj) const {
    Vec12d u2 = elm_states[2];
    Vec12d u1 = elm_states[1];
    Vec12d u0 = elm_states[0];
    FEM_Scaler h = _dt;
    Vec12d v2 = (u2 - u1)/h;

    FEM_Scaler psiI,psi,psiD;

    FEM_Scaler vol = attrs._volume;
    FEM_Scaler m = vol * attrs._density / 4;

    const auto& nf = attrs._nodal_force;

    Vec12d y = (u2 -2*u1 + u0) - h*h*attrs._volume_force_accel.replicate(4,1) - h*h*nf/m;
    psiI = y.squaredNorm();

    Mat3x3d F,L;
    ComputeDeformationGradient(attrs._Minv,u2,F);
    ComputeDeformationGradient(attrs._Minv,v2,L);

    force_model->ComputePsi(attrs,F,psi);     
    damping_model->ComputePsi(attrs,L,psiD);

    *elm_obj = (psiI*m/h/h + (psi + h*psiD)*vol);

    return 0;
}

int BackEulerIntegrator::EvalElmObjDeriv(const TetAttributes& attrs,
            const std::shared_ptr<BaseForceModel>& force_model,
            const std::shared_ptr<DiricletDampingModel>& damping_model,
            const std::vector<Vec12d>& elm_states,FEM_Scaler* elm_obj,Vec12d& elm_deriv) const {
    Vec12d u2 = elm_states[2];
    Vec12d u1 = elm_states[1];
    Vec12d u0 = elm_states[0];
    FEM_Scaler h = _dt;
    Vec12d v2 = (u2 - u1)/h;

    FEM_Scaler psiI,psi,psiD;
    Vec9d dpsi,dpsiD;

    FEM_Scaler vol = attrs._volume;
    FEM_Scaler m = vol * attrs._density / 4;  // nodal mass

    const auto& nf = attrs._nodal_force;

    Mat3x3d F,L;
    ComputeDeformationGradient(attrs._Minv,u2,F);
    ComputeDeformationGradient(attrs._Minv,v2,L);

    force_model->ComputePsiDeriv(attrs,F,psi,dpsi); 
    damping_model->ComputePsiDeriv(attrs,L,psiD,dpsiD);

    Vec12d y = (u2 -2*u1 + u0) - h*h*attrs._volume_force_accel.replicate(4,1) - h*h*nf/m;
    psiI = y.squaredNorm();

    const Mat9x12d& dFdX = attrs._dFdX;

    *elm_obj = (psiI*m/h/h + (psi + h*psiD)*vol);
    elm_deriv = (y*m/h/h + dFdX.transpose() * (dpsi + dpsiD) * vol);

    return 0;                
}

int BackEulerIntegrator::EvalElmObjDerivJacobi(const TetAttributes& attrs,
        const std::shared_ptr<BaseForceModel>& force_model,
        const std::shared_ptr<DiricletDampingModel>& damping_model,
        const std::vector<Vec12d>& elm_states,
        FEM_Scaler* elm_obj,Vec12d& elm_deriv,Mat12x12d& elm_H,bool filtering,bool debug) const{
    Vec12d u2 = elm_states[2];
    Vec12d u1 = elm_states[1];
    Vec12d u0 = elm_states[0];
    FEM_Scaler h = _dt;
    Vec12d v2 = (u2 - u1)/h;

    FEM_Scaler psiI,psi,psiD;
    Vec9d dpsi,dpsiD;
    Mat9x9d ddpsi,ddpsiD;
         
    FEM_Scaler vol = attrs._volume;
    FEM_Scaler m = vol * attrs._density / 4;  // nodal mass

    const auto& nf = attrs._nodal_force;

    Mat3x3d F,L;
    ComputeDeformationGradient(attrs._Minv,u2,F);
    ComputeDeformationGradient(attrs._Minv,v2,L);

    force_model->ComputePsiDerivHessian(attrs,F,psi,dpsi,ddpsi,filtering);
    // force_model->ComputePsiDerivHessian(attrs.emp,F,psi,dpsi,ddpsi,filtering);
    damping_model->ComputePsiDerivHessian(attrs,L,psiD,dpsiD,ddpsiD);

    Vec12d y = (u2 -2*u1 + u0) - h*h*attrs._volume_force_accel.replicate(4,1) - h*h*nf/m;
    psiI = y.squaredNorm();
    const Mat9x12d& dFdX = attrs._dFdX;
    *elm_obj = (psiI*m/h/h + (psi + h*psiD)*vol);
    elm_deriv = (y*m/h/h + dFdX.transpose() * (dpsi + dpsiD) * vol);
    elm_H = Mat12x12d::Identity()*m/h/h + dFdX.transpose() * (ddpsi + ddpsiD/h) * dFdX * vol;

    return 0;
}
