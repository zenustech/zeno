#include "backward_euler_integrator.h"
#include <iostream>
#include <ctime>
#include <numeric>

#include <stable_isotropic_NH.h>
#include <stable_Stvk.h>

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
    FEM_Scaler m = vol * attrs._density / 4;

    const auto& ext_f = attrs._ext_f;

    Vec12d y = u2 - 2*u1 + u0 - h*h*_gravity.replicate(4,1) - h*h*ext_f/m; 
    PhiI = y.squaredNorm() * m / 2 / h;

    Mat3x3d F,L;
    ComputeDeformationGradient(attrs._Minv,u2,F);
    ComputeDeformationGradient(attrs._Minv,v2,L);

    if(dynamic_cast<ElasticModel*>(force_model.get())){
        auto model = dynamic_cast<ElasticModel*>(force_model.get());
        model->ComputePhi(attrs.emp,F,PsiE);
    }else{
        auto model = dynamic_cast<PlasticForceModel*>(force_model.get());
        model->ComputePsi(attrs.pmp,attrs.emp,F,PsiE);
    }
     
    damping_model->ComputePhi(attrs.v,L,PsiD);

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

    const auto& ext_f = attrs._ext_f;

    Mat3x3d F,L;
    ComputeDeformationGradient(attrs._Minv,u2,F);
    ComputeDeformationGradient(attrs._Minv,v2,L);

    if(dynamic_cast<ElasticModel*>(force_model.get())){
        auto model = dynamic_cast<ElasticModel*>(force_model.get());
        model->ComputePhiDeriv(attrs.emp,F,PsiE,dPsiE);
    }else{
        auto model = dynamic_cast<PlasticForceModel*>(force_model.get());
        model->ComputePsiDeriv(attrs.pmp,attrs.emp,F,PsiE,dPsiE);
    }

    damping_model->ComputePhiDeriv(attrs.v,L,PsiD,dPsiD);

    Vec12d y = u2 - 2*u1 + u0 - h*h*_gravity.replicate(4,1) - h*h*ext_f/m;
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
        FEM_Scaler* elm_obj,Vec12d& elm_deriv,Mat12x12d& elm_H,bool filtering,bool debug) const{
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

    const auto& ext_f = attrs._ext_f;

    Mat3x3d F,L;
    ComputeDeformationGradient(attrs._Minv,u2,F);
    ComputeDeformationGradient(attrs._Minv,v2,L);

    if(debug){
        std::cout << "DEBUG" << std::endl;
        Mat3x3d Ftest = Mat3x3d::Random();
        Ftest = Ftest.transpose() * Ftest;

        Vec3d forient;
        FEM_Scaler E,nu;


        if(dynamic_cast<PlasticForceModel*>(force_model.get())){
            std::cout << "PlasticModel" << std::endl;
            forient = attrs.emp.forient;
            E = attrs.emp.E;
            nu = attrs.emp.nu;
        }else{
            std::cout << "ElasticModel" << std::endl;
            forient = attrs.emp.forient;
            E = attrs.emp.E;
            nu = attrs.emp.nu;
        }


        Vec3d act = Vec3d::Random();
        act = act.cwiseAbs();

        Mat3x3d R = MatHelper::Orient2R(forient);

        // Mat3x3d Act = R * act.asDiagonal() * R.transpose();
        Mat3x3d Act = Mat3x3d::Identity();

        Mat9x9d ddpsi_cmp = Mat9x9d::Zero();
        Vec9d dpsi_cmp = Vec9d::Zero();
        FEM_Scaler psi_cmp = 0;

        dynamic_cast<PlasticForceModel*>(force_model.get())->ComputePsiDerivHessian(attrs.pmp,attrs.emp,Ftest,psi_cmp,dpsi_cmp,ddpsi_cmp,false);

        Mat9x9d ddpsi_fd = Mat9x9d::Zero();
        Vec9d dpsi_fd = Vec9d::Zero();
        FEM_Scaler ratio = 1e-6;
        for(size_t i = 0;i < 9;++i){
                Mat3x3d F_tmp = Ftest;
                Vec9d f_tmp = MatHelper::VEC(F_tmp);
                FEM_Scaler step = f_tmp[i] * ratio;
                step = fabs(step) < ratio ? ratio : step;
                f_tmp[i] += step;
                F_tmp = MatHelper::MAT(f_tmp);

                FEM_Scaler psi_tmp;
                Vec9d dpsi_tmp;
                dynamic_cast<PlasticForceModel*>(force_model.get())->ComputePsiDeriv(attrs.pmp,attrs.emp,F_tmp,psi_tmp,dpsi_tmp);

                dpsi_fd[i] = (psi_tmp - psi_cmp) / step;
                ddpsi_fd.col(i) = (dpsi_tmp - dpsi_cmp) / step;
        }

        FEM_Scaler ddpsi_error = (ddpsi_fd - ddpsi_cmp).norm() / ddpsi_fd.norm();
        FEM_Scaler dpsi_error = (dpsi_fd - dpsi_cmp).norm() / dpsi_fd.norm();
        if(ddpsi_error > 1e-5 || dpsi_error > 1e-3){
                FEM_Scaler lambda = ElasticModel::Enu2Lambda(attrs.emp.E,attrs.emp.nu);
                FEM_Scaler mu = ElasticModel::Enu2Mu(attrs.emp.E,attrs.emp.nu);
                // Vec9d eig_vals;
                // Vec9d eig_vecs[9];

                // dynamic_cast<StableStvk*>(force_model.get())->ComputeIsoEigenSystem(lambda,mu,Ftest,eig_vals,eig_vecs);
                // std::cout << "Check Eigen !!!!!: " << std::endl;
                // for(size_t i = 0;i < 9;++i){
                //     Vec9d Ax = ddpsi_fd * eig_vecs[i];
                //     Vec9d lx = eig_vals[i] * eig_vecs[i];
                //     std::cout << "IDX : " << i << std::endl;
                //     for(size_t j = 0;j < 9;++j){
                //         std::cout << Ax[j] /  lx[j] << "\t";std::cout.flush();
                //     }
                //     std::cout << std::endl;
                // }

                std::cout << "FAIL WITH FAct = " << std::endl << Ftest << std::endl << Act << std::endl;
                std::cerr << "dpsi_error : " << dpsi_error << std::endl;
                std::cerr << "dpsi_fd : \t" << dpsi_fd.transpose() << std::endl;
                std::cerr << "dpsi : \t" << dpsi_cmp.transpose() << std::endl; 
                std::cerr << "ddpsi_error : " << ddpsi_error << std::endl;
                std::cout << "ddpsi : " << std::endl << ddpsi_cmp << std::endl;
                std::cout << "ddpsi_fd : " << std::endl << ddpsi_fd << std::endl;


                std::cout << "pstrain : " << attrs.pmp.plastic_strain.transpose() << std::endl;
                std::cout << "kinimatic_hardening_shift : " << attrs.pmp.kinematic_hardening_shift.transpose() << std::endl;

                throw std::runtime_error("ddpsi_error");
        }

    }


    if(dynamic_cast<ElasticModel*>(force_model.get())){
        auto model = dynamic_cast<ElasticModel*>(force_model.get());
        model->ComputePhiDerivHessian(attrs.emp,F,PsiE,dPsiE,ddPsiE,filtering);
    }else{
        auto model = dynamic_cast<PlasticForceModel*>(force_model.get());
        model->ComputePsiDerivHessian(attrs.pmp,attrs.emp,F,PsiE,dPsiE,ddPsiE,filtering);
    }

    // force_model->ComputePhiDerivHessian(attrs.emp,F,PsiE,dPsiE,ddPsiE,filtering);
    damping_model->ComputePhiDerivHessian(attrs.v,L,PsiD,dPsiD,ddPsiD);

    Vec12d y = u2 - 2*u1 + u0 - h*h*_gravity.replicate(4,1) - h*h*ext_f/m;
    PhiI = y.squaredNorm() * m / 2 / h;
    *elm_obj = PhiI + h * PsiE * vol + h * h * PsiD * vol;   

    const Mat9x12d& dFdX = attrs._dFdX;
    elm_deriv = m * y / h + dFdX.transpose() * (h*dPsiE + h*dPsiD) * vol;

    elm_H = Mat12x12d::Identity() * m / h + vol * dFdX.transpose() * (h*ddPsiE + ddPsiD) * dFdX;

    return 0;
}
