#include "backward_euler_integrator.h"
#include <iostream>
#include <ctime>
#include <numeric>

#include "diriclet_damping.h"

int BackEulerIntegrator::EvalElmObj(const TetAttributes attrs,
            const std::shared_ptr<MuscleForceModel>& force_model,
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
            const std::shared_ptr<MuscleForceModel>& force_model,
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
        const std::shared_ptr<MuscleForceModel>& force_model,
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


    // bool debug = false;
    // if(debug){
    //     Mat3x3d Ftest = Mat3x3d::Random();
    //     Ftest = Ftest.transpose() * Ftest;
    //     Mat3x3d fiber = attrs._fiberOrient;
    //     Vec3d w = attrs._fiberWeight;

    //     Vec3d act = Vec3d::Random();
    //     act = act.cwiseAbs();
    //     Mat3x3d Act = fiber.transpose() * act.asDiagonal() * fiber;

    //     FEM_Scaler E = attrs._E;
    //     FEM_Scaler nu = attrs._nu;

    //     Mat9x9d ddpsi_cmp = Mat9x9d::Zero();
    //     Vec9d dpsi_cmp = Vec9d::Zero();
    //     FEM_Scaler psi_cmp = 0;

    //     force_model->ComputePhiDerivHessian(Act,w,fiber,E,nu,Ftest,psi_cmp,dpsi_cmp,ddpsi_cmp,false);

    //     Mat9x9d ddpsi_fd = Mat9x9d::Zero();
    //     Vec9d dpsi_fd = Vec9d::Zero();
    //     FEM_Scaler ratio = 1e-6;
    //     for(size_t i = 0;i < 9;++i){
    //             Mat3x3d F_tmp = Ftest;
    //             Vec9d f_tmp = MatHelper::VEC(F_tmp);
    //             FEM_Scaler step = f_tmp[i] * ratio;
    //             step = fabs(step) < ratio ? ratio : step;
    //             f_tmp[i] += step;
    //             F_tmp = MatHelper::MAT(f_tmp);

    //             FEM_Scaler psi_tmp;
    //             Vec9d dpsi_tmp;
    //             force_model->ComputePhiDeriv(Act,w,fiber,E,nu,F_tmp,psi_tmp,dpsi_tmp);

    //             dpsi_fd[i] = (psi_tmp - psi_cmp) / step;
    //             ddpsi_fd.col(i) = (dpsi_tmp - dpsi_cmp) / step;
    //     }

    //     FEM_Scaler ddpsi_error = (ddpsi_fd - ddpsi_cmp).norm() / ddpsi_fd.norm();
    //     FEM_Scaler dpsi_error = (dpsi_fd - dpsi_cmp).norm() / dpsi_fd.norm();
    //     if(ddpsi_error > 1e-5 || dpsi_error > 1e-3){
    //             std::cout << "FAIL WITH FAct = " << std::endl << Ftest << std::endl << Act << std::endl;
    //             std::cerr << "dpsi_error : " << dpsi_error << std::endl;
    //             std::cerr << "dpsi_fd : \t" << dpsi_fd.transpose() << std::endl;
    //             std::cerr << "dpsi : \t" << dpsi_cmp.transpose() << std::endl; 
    //             std::cerr << "ddpsi_error : " << ddpsi_error << std::endl;
    //             std::cout << "ddpsi : " << std::endl << ddpsi_cmp << std::endl;
    //             std::cout << "ddpsi_fd : " << std::endl << ddpsi_fd << std::endl;
    //             throw std::runtime_error("ddpsi_error");
    //     }
    // }


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